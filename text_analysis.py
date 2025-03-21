from transformers import pipeline

def analyze_text(text, model_name="facebook/bart-large-cnn"):
    """
    Génère un résumé du texte fourni en entrée en utilisant un modèle de summarization.
    :param text: Texte à analyser.
    :param model_name: Modèle Hugging Face utilisé pour le résumé.
    :return: Résumé du texte.
    """
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Test rapide
if __name__ == "__main__":
    example_text = """
    Hugging Face est une plateforme qui fournit des modèles d'intelligence artificielle avancés 
    pour le traitement du langage naturel, la vision par ordinateur et d'autres tâches d'apprentissage automatique.
    """
    print("Résumé :", analyze_text(example_text))
