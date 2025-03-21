import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found! Check your .env file.")

# Authenticate
login(token=HUGGINGFACE_TOKEN)
print("✅ Successfully logged in to Hugging Face!")
