import os
import glob
import json
import argparse
from pathlib import Path

from read_pdf import extract_text_from_pdf
from job_fit_scoring import calculate_fit

try:
    import torch
    from transformers import AutoTokenizer
    from model_ner import ResumeNERModel
    from main import predict_cv_info
    HAS_MODEL_STACK = True
except Exception:
    HAS_MODEL_STACK = False

try:
    import ssl
    ssl.create_default_context = ssl._create_unverified_context
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SEMANTIC = True
except Exception:
    HAS_SEMANTIC = False


def load_model(model_name, weights_path=None, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    needs_prefix = True if "roberta" in model_name.lower() else False
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=needs_prefix)
    model = ResumeNERModel(model_name=model_name, num_labels=9).to(device)
    
    if weights_path:
        try:
            weights = torch.load(weights_path, map_location=device)
            # Check if this is a partial NER weights dict or full state_dict
            if "classifier.weight" in weights:
                # New format: only NER weights (classifier + CRF)
                model.classifier.weight.data = weights["classifier.weight"]
                model.classifier.bias.data = weights["classifier.bias"]
                model.crf.transitions.data = weights["crf.transitions"]
                model.crf.start_transitions.data = weights["crf.start_transitions"]
                model.crf.end_transitions.data = weights["crf.end_transitions"]
            else:
                # Old format: full state_dict
                model.load_state_dict(weights, strict=False)
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
    
    model.float()
    model.eval()
    return model, tokenizer, device


def prepare_candidate_from_text(text):
    # Fallback: use full resume text for each field to exercise the scoring function
    return {"skills": text, "experience": text, "education": text}


def default_job_criteria():
    return {
        "skills": "Python, Machine Learning, NLP, PyTorch",
        "experience": "AI or Research experience",
        "education": "Bachelor in Computer Science"
    }


def load_job_requirements(req_path):
    if not req_path:
        return {}
    req_file = Path(req_path)
    if not req_file.exists():
        return {}
    with req_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_best_model(training_results_path):
    if not training_results_path:
        return None, None
    results_file = Path(training_results_path)
    if not results_file.exists():
        return None, None
    with results_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    best_name = None
    best_f1 = -1.0
    best_path = None
    for name, metrics in data.items():
        if "error" in metrics:
            continue
        f1 = metrics.get("best_f1", -1.0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_path = metrics.get("model_path")
    return best_name, best_path


def split_sentences(text):
    normalized = " ".join(text.replace("\n", " ").split())
    chunks = normalized.replace("?", ".").replace("!", ".").split(".")
    sentences = [chunk.strip() for chunk in chunks if chunk.strip()]
    return [sentence for sentence in sentences if len(sentence) >= 20]


def extract_semantic_highlights(text, job_criteria, model, top_k):
    sentences = split_sentences(text)
    if not sentences:
        return {}

    sentence_embeddings = model.encode(sentences)
    highlights = {}
    for key in ("skills", "experience", "education"):
        req_text = job_criteria.get(key, "")
        if not req_text:
            continue
        req_embedding = model.encode([req_text])
        scores = cosine_similarity(sentence_embeddings, req_embedding).reshape(-1)
        top_indices = scores.argsort()[-top_k:][::-1]
        highlights[key] = [
            {"sentence": sentences[index], "score": float(scores[index])}
            for index in top_indices
        ]
    return highlights


def process_pdfs(pdf_dir, out_path=None, use_model=False, criteria=None, requirements=None, category=None, max_per_category=None, model_name=None, weights_path=None, semantic_highlight=False, highlight_top_k=3):
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(glob.glob(str(pdf_dir / "**" / "*.pdf"), recursive=True))
    results = []
    category_scores = {}

    semantic_model = None
    if semantic_highlight and HAS_SEMANTIC:
        try:
            semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            semantic_model = None

    model = tokenizer = device = None
    if use_model and HAS_MODEL_STACK:
        try:
            model, tokenizer, device = load_model(model_name, weights_path=weights_path)
            print(f"Loaded model on {device}")
        except Exception as e:
            print(f"Failed to initialize model stack: {e}")
            model = tokenizer = device = None

    for p in pdfs:
        rel = Path(p).relative_to(pdf_dir)
        cv_category = rel.parts[0] if rel.parts else "UNKNOWN"
        if category and cv_category.lower() != category.lower():
            continue
        if max_per_category and len(category_scores.get(cv_category, [])) >= max_per_category:
            continue

        text = extract_text_from_pdf(p)
        if not text:
            print(f"{p}: no text extracted, skipping")
            continue

        if model and tokenizer:
            try:
                candidate = predict_cv_info(text, model, tokenizer, device)
            except Exception as e:
                print(f"Model extraction failed for {p}: {e}, using fallback text")
                candidate = prepare_candidate_from_text(text)
        else:
            candidate = prepare_candidate_from_text(text)

        job_criteria = criteria or (requirements.get(cv_category) if requirements else None) or default_job_criteria()
        score, details = calculate_fit(candidate, job_criteria)
        score = float(score)
        details = {key: float(value) for key, value in details.items()}

        highlights = {}
        if semantic_highlight:
            if not semantic_model:
                print("Semantic highlight unavailable; missing sentence-transformers or model load failed.")
            else:
                highlights = extract_semantic_highlights(text, job_criteria, semantic_model, highlight_top_k)

        print(f"{os.path.relpath(p)} -> Score: {score:.2%} | Details: {{'skills': {details['skills']:.2%}, 'experience': {details['experience']:.2%}, 'education': {details['education']:.2%}}}")
        results.append({
            "file": str(p),
            "category": cv_category,
            "score": score,
            "details": details,
            "highlights": highlights
        })

        category_scores.setdefault(cv_category, []).append(score)
        if max_per_category and len(category_scores[cv_category]) >= max_per_category:
            continue

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    if category_scores:
        print("\nCategory averages:")
        for name, scores in sorted(category_scores.items()):
            avg_score = sum(scores) / max(len(scores), 1)
            print(f"{name}: {avg_score:.2%} (n={len(scores)})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test job fit scoring over a directory of CV PDFs")
    parser.add_argument("--pdf-dir", default="cv/data", help="Root folder containing PDF resumes")
    parser.add_argument("--out", default="job_fit_results.json", help="Output JSON file")
    parser.add_argument("--use-model", action="store_true", help="Attempt to run the NER model before falling back to text similarity")
    parser.add_argument("--category", help="Filter to a single category folder (e.g., INFORMATION-TECHNOLOGY)")
    parser.add_argument("--max-per-category", type=int, help="Limit number of CVs processed per category")
    parser.add_argument("--requirements-file", default="job_requirements.json", help="Path to job requirements JSON")
    parser.add_argument("--model-name", help="Transformer model name to load (e.g., bert-base-uncased)")
    parser.add_argument("--weights", help="Path to model weights .pth")
    parser.add_argument("--training-results", default="training_results.json", help="Path to training results JSON for best model selection")
    parser.add_argument("--semantic-highlight", action="store_true", help="Include top-matching resume sentences for each requirement section")
    parser.add_argument("--highlight-top-k", type=int, default=3, help="Number of sentences to keep per section")
    args = parser.parse_args()

    requirements = load_job_requirements(args.requirements_file)
    model_name = args.model_name
    weights_path = args.weights
    if args.use_model and (not model_name or not weights_path):
        best_name, best_path = pick_best_model(args.training_results)
        model_name = model_name or best_name or "distilbert-base-uncased"
        weights_path = weights_path or best_path

    process_pdfs(
        args.pdf_dir,
        out_path=args.out,
        use_model=args.use_model,
        requirements=requirements,
        category=args.category,
        max_per_category=args.max_per_category,
        model_name=model_name,
        weights_path=weights_path,
        semantic_highlight=args.semantic_highlight,
        highlight_top_k=args.highlight_top_k,
    )
