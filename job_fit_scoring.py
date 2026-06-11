import ssl
ssl.create_default_context = ssl._create_unverified_context

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading semantic model (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2') 

def compute_similarity(cand_text, req_text):
    if not cand_text or not req_text:
        return 0.0
    
    cand_items = [item.strip() for item in cand_text.split(',')]
    req_items = [item.strip() for item in req_text.split(',')]
    
    cand_items = [item for item in cand_items if item]
    req_items = [item for item in req_items if item]
    
    if not cand_items or not req_items:
        return 0.0
        
    cand_embeddings = sbert_model.encode(cand_items)
    req_embeddings = sbert_model.encode(req_items)
    
    sim_matrix = cosine_similarity(req_embeddings, cand_embeddings)
    max_sims = sim_matrix.max(axis=1)
    
    return float(max_sims.mean())

def calculate_fit(cv_data, hr_criteria):
    skills_sim = compute_similarity(cv_data['skills'], hr_criteria['skills'])
    exp_sim = compute_similarity(cv_data['experience'], hr_criteria['experience'])
    edu_sim = compute_similarity(cv_data['education'], hr_criteria['education'])
    
    weights = {'skills': 0.5, 'experience': 0.3, 'education': 0.2}
    
    final_score = (skills_sim * weights['skills'] +
                   exp_sim * weights['experience'] +
                   edu_sim * weights['education'])
    
    details = {
        'skills': skills_sim,
        'experience': exp_sim,
        'education': edu_sim
    }
    return final_score, details

if __name__ == "__main__":
    sample_cv = {
        'skills': "Python, Machine Learning, TensorFlow, NLP",
        'experience': "3 years Data Scientist",
        'education': "Bachelor in Computer Science"
    }

    sample_hr = {
        'skills': "Python, Deep Learning, Pytorch, NLP",
        'experience': "Data Scientist experience",
        'education': "Computer Science Degree"
    }

    print("Calculating fit score...")
    score, report = calculate_fit(sample_cv, sample_hr)

    print("\n--- Fit Score Report ---")
    for k, v in report.items():
        print(f"{k.capitalize()}: {v:.2%}")
    print(f"Overall Fit: {score:.2%}")