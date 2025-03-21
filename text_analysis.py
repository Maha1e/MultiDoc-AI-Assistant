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
    Vous avez 35 ans et vous êtes responsable d'une équipe de 10 personnes.
Marc Drapier est un de vos collaborateurs, il fait partie de cette équipe depuis
15 ans, avant votre nomination. Il a 45 ans et travaille à temps partiel (30 h par
semaine), après une longue période d'absence pour dépression.
Vous avez remarqué qu'à plusieurs reprises, Marc Drapier venait vous
soumettre une difficulté qu'il rencontrait dans son travail, sans apporter d'idée
de solutions. Quand vous lui en avez fait la remarque, Marc Drapier n'a pas
apprécié : il s'estime défavorisé par rapport à ses collègues, du fait de son temps
partiel et trouve normal que votre soutien soit plus intensif le concernant.
Il ne recherche pas l'avis de ses collègues, car il supporte très difficilement leurs
critiques, même quand elles sont justifiées. Quand il est présent aux réunions,
il semble trouver le temps long, ne pose pas de questions et ne fait jamais de
suggestions d'amélioration. Ses relations de travail à l'intérieur de l'équipe sont
médiocres.
Il répond quelquefois sèchement à certains collègues d'autres services,
s'estimant être dérangé dans son travail quand ces derniers lui demandent un
renseignement qui leur est nécessaire.
Vous l'avez déjà reçu en entretien, à plusieurs reprises, pour lui demander
d'améliorer sa communication.
Hier, en fin de journée, il a passé, très sèchement à un de ses jeunes collègues,
qu'il estime favorisé (25 ans, jeune diplômé en poste depuis un an) une
communication avec un interlocuteur d'un autre service qu'il trouvait "collant".
Ce jeune collègue a repris sa communication, a géré très aimablement
l'interlocuteur mais n'a pas apprécié le comportement de Marc Drapier et s'est
accroché avec lui, le lendemain matin.
    """
    print("Résumé :", analyze_text(example_text))
