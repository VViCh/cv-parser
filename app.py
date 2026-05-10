import streamlit as st
import torch
from model_ner import ResumeNERModel
from job_fit_scoring import calculate_fit
from transformers import AutoTokenizer
from main import predict_cv_info

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
req_skills = st.sidebar.text_area("Required Skills", "Python, Machine Learning, NLP, SQL")
req_exp = st.sidebar.text_area("Experience", "AI Research, Data Scientist")
req_edu = st.sidebar.text_area("Education", "Bachelor Degree in Computer Science")

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = ResumeNERModel(num_labels=9).to(device)
    try:
        model.load_state_dict(torch.load("resume_ner_model.pth", map_location=device))
    except FileNotFoundError:
        pass
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_models()

st.subheader("Upload Candidate CV")
uploaded_file = st.file_uploader("Select candidate PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner('Analyzing CV...'):
        # Mock data simulation
        sample_cv_text = "I have 3 years of experience as an AI Research Assistant. My skills include Python, Machine Learning, and NLP. I have a Bachelor Degree in Computer Science."
        
        candidate_data = predict_cv_info(sample_cv_text, model, tokenizer, device)
        job_data = {'skills': req_skills, 'experience': req_exp, 'education': req_edu}
        
        score, details = calculate_fit(candidate_data, job_data)
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.metric("Overall Match", f"{score:.2%}")
            if score > 0.70:
                st.success("Result: Passed")
            else:
                st.error("Result: Rejected")
        
        with c2:
            st.subheader("Analysis Breakdown")
            for category, value in details.items():
                st.write(f"**{category.capitalize()} Score:** {value:.1%}")
                st.progress(value)

st.sidebar.markdown("---")
st.write("System status: Active")