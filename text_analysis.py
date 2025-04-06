from transformers import pipeline

def analyze_text(text, model_name="sshleifer/distilbart-cnn-12-6"):
    """
    Cette fonction résume un texte en utilisant un modèle de summarization Hugging Face.

    Args:
    text : Le texte à résumer.
    model_name : Nom du modèle pré-entraîné à utiliser.

    """
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


