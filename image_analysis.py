from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import streamlit as st

# Forcer le téléchargement et le cache local



def load_image(image_path):
    """
    Cette fonction permet de charger une image à partir du chemin fourni.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        print("🖼️ Image chargée avec succès.")
        return image
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement de l'image: {e}")

def generate_caption(image):
    """
    Cette fonction permet d'utiliser le modèle BLIP pour générer une description de l'image.
    """

    processor = BlipProcessor.from_pretrained("C:/Users/CYTech Student/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/snapshots/<hash>")
    model = BlipForConditionalGeneration.from_pretrained("C:/Users/CYTech Student/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/snapshots/<hash>")


    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption écrite générée: {caption}")
    return caption

def summarize_caption(caption):
    """
    Cette fonction permet de résumer ou extraire les points clés de la caption avec un modèle de NLP.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(caption, max_length=50, min_length=20, do_sample=False)[0]["summary_text"]
    print(f"Résumé: {summary}")
    return summary

