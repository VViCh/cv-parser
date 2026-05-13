import json
import glob
import tempfile
from pathlib import Path

import streamlit as st
import torch
from model_ner import ResumeNERModel
from job_fit_scoring import calculate_fit
from transformers import AutoTokenizer
from main import predict_cv_info
from read_pdf import extract_text_from_pdf

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("Automated Resume Screening")
st.write("Deep learning-based resume screening system.")

st.sidebar.header("Job Requirements")

requirements_path = Path("job_requirements.json")
requirements_data = {}
if requirements_path.exists():
    with requirements_path.open("r", encoding="utf-8") as handle:
        requirements_data = json.load(handle)

requirement_keys = ["Custom"] + sorted(requirements_data.keys())
default_req_key = "DEFAULT" if "DEFAULT" in requirements_data else "Custom"

if "req_choice" not in st.session_state:
    st.session_state.req_choice = default_req_key
if "req_skills" not in st.session_state:
    st.session_state.req_skills = "Python, Machine Learning, NLP, SQL"
if "req_exp" not in st.session_state:
    st.session_state.req_exp = "AI Research, Data Scientist"
if "req_edu" not in st.session_state:
    st.session_state.req_edu = "Bachelor Degree in Computer Science"

req_choice = st.sidebar.selectbox("Requirement profile", requirement_keys, index=requirement_keys.index(st.session_state.req_choice))

if req_choice != "Custom" and req_choice in requirements_data:
    selected_req = requirements_data[req_choice]
    st.session_state.req_skills = selected_req.get("skills", "")
    st.session_state.req_exp = selected_req.get("experience", "")
    st.session_state.req_edu = selected_req.get("education", "")
    st.session_state.req_choice = req_choice

req_skills = st.sidebar.text_area("Required Skills", st.session_state.req_skills)
req_exp = st.sidebar.text_area("Experience", st.session_state.req_exp)
req_edu = st.sidebar.text_area("Education", st.session_state.req_edu)

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    model_name = "microsoft/deberta-v3-base"
    weights_path = "models/resume_ner_model_final_microsoft_deberta-v3-base.pth"
    
    needs_prefix = True if "roberta" in model_name.lower() else False
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=needs_prefix)
    model = ResumeNERModel(model_name=model_name, num_labels=9).to(device)
    
    if Path(weights_path).exists():
        try:
            weights = torch.load(weights_path, map_location=device)
            try:
                model.load_state_dict(weights, strict=False)
            except Exception as e:
                st.caption(f"Error: Failed to load weights: {e}")
        except Exception as e:
            st.caption(f"Error: Could not read weights file: {e}")
    
    model.float()
    model.eval()
    return model, tokenizer, device, model_name, weights_path

model, tokenizer, device, model_name, weights_path = load_models()
st.caption(f"Model: {model_name} | Weights: {weights_path or 'uninitialized'}")

st.subheader("Upload Candidate CV")
if "folder_pdfs" not in st.session_state:
    st.session_state.folder_pdfs = []

source_mode = st.radio("Input source", ["Upload PDF", "Local folder"], horizontal=True)

uploaded_file = None
selected_path = None

if source_mode == "Upload PDF":
    uploaded_file = st.file_uploader("Select candidate PDF file", type=["pdf"])
else:
    folder_path = st.text_input("Local folder path", "cv/data")
    if st.button("Scan folder"):
        st.session_state.folder_pdfs = sorted(
            glob.glob(str(Path(folder_path) / "**" / "*.pdf"), recursive=True)
        )
    if st.session_state.folder_pdfs:
        selected_path = st.selectbox("Pick a PDF", st.session_state.folder_pdfs)

run_scan = st.button("Parse & score")

if run_scan:
    if source_mode == "Upload PDF" and not uploaded_file:
        st.warning("Please upload a PDF first.")
        st.stop()
    if source_mode == "Local folder" and not selected_path:
        st.warning("Please scan a folder and select a PDF.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    status.write("Step 1/4: Reading PDF...")
    progress.progress(10)

    pdf_path = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            pdf_path = temp_pdf.name
    else:
        pdf_path = selected_path

    debug_info = {}

    text_cv = extract_text_from_pdf(pdf_path)
    debug_info["text_len"] = len(text_cv)
    debug_info["text_preview"] = text_cv[:300]
    progress.progress(40)

    if not text_cv:
        st.error("No text extracted from the PDF.")
        st.stop()

    status.write("Step 2/4: Extracting entities...")
    # Run extraction without verbose debug prints in the UI
    candidate_data = predict_cv_info(text_cv, model, tokenizer, device, verbose=False)
    debug_info["candidate_data"] = candidate_data
    
    # Fallback: if key scoring fields (skills, experience, education) are empty, use full text
    has_skills = bool(candidate_data.get("skills", "").strip())
    has_experience = bool(candidate_data.get("experience", "").strip())
    has_education = bool(candidate_data.get("education", "").strip())
    
    if not (has_skills or has_experience or has_education):
        st.info("Note: NER model could not extract skills/experience/education. Falling back to full-text similarity scoring.")
        candidate_data = {
            "skills": text_cv,
            "experience": text_cv,
            "education": text_cv,
            "designation": candidate_data.get("designation", "")
        }
        debug_info["fallback_used"] = True
    else:
        debug_info["fallback_used"] = False
    
    progress.progress(70)

    status.write("Step 3/4: Scoring against requirements...")
    job_data = {"skills": req_skills, "experience": req_exp, "education": req_edu}
    score, details = calculate_fit(candidate_data, job_data)
    debug_info["score"] = score
    debug_info["details"] = details
    progress.progress(90)

    status.write("Step 4/4: Rendering results...")
    progress.progress(100)

    # Debug UI commented out to reduce on-screen noise
    # with st.expander("Debug Info", expanded=False):
    #     st.write(f"**Model:** {model_name}")
    #     st.write(f"**Weights:** {weights_path}")
    #     st.write(f"**Text length:** {debug_info['text_len']}")
    #     st.write(f"**Text preview:**\n{debug_info['text_preview']}")
    #     st.write(f"**Fallback used:** {debug_info.get('fallback_used', False)}")
    #     st.write(f"**Candidate data:**")
    #     st.json(debug_info["candidate_data"])
    #     st.write(f"**Score:** {debug_info['score']}")
    #     st.write(f"**Details:** {debug_info['details']}")

    st.divider()
    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric("Overall Match", f"{score:.2%}")
        if score >= 0.75:
            st.success("Result: Strong match")
        elif score >= 0.50:
            st.warning("Result: Potential match")
        else:
            st.info("Result: Needs review")

    with c2:
        st.subheader("Analysis Breakdown")
        for category, value in details.items():
            safe_value = float(value)
            st.write(f"**{category.capitalize()} Score:** {safe_value:.1%}")
            st.progress(safe_value)

st.sidebar.markdown("---")
st.write("System status: Active")