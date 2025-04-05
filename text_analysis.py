from transformers import pipeline

def analyze_text(text, model_name="facebook/bart-large-cnn"):
    """
    Cette fonction résume un texte en utilisant un modèle de summarization Hugging Face.

    Args:
    text : Le texte à résumer.
    model_name : Nom du modèle pré-entraîné à utiliser.

    """
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


