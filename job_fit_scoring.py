from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading semantic model (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2') 

def compute_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    embedding1 = sbert_model.encode([text1])
    embedding2 = sbert_model.encode([text2])
    sim = cosine_similarity(embedding1, embedding2)[0][0]
    return sim

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