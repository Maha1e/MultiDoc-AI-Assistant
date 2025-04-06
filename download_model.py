# download_model.py
from sentence_transformers import SentenceTransformer

# Load the model from Hugging Face and save locally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("./models/miniLM")

print("✅ Model downloaded and saved locally to ./models/miniLM")
