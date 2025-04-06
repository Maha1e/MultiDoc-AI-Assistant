import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import numpy as np

def extract_text_from_pdf(pdf_path):
    """
    Cette fonction permet d'extraire le texte d'un fichier PDF page par page.
    
    Args:
    pdf_path : Le chemin du fichier PDF.
    
    Return:
    Liste contenant le texte de chaque page.
    """
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")  # Ajoute un texte vide si None
    return text

def chunk_text(text_list, chunk_size=400):
    """
    Cette fonction permet de découper le texte en segments (chunks) de taille définie.
    
    Args:
    text_list: Liste de textes extraits du PDF (depuis "extract_text_from_pdf")
    chunk_size: Nombre de caractères par segment (chunk).

    Return:
    La liste des segments de texte (chunks).
    """
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks


def compute_embeddings(chunks, model_name="paraphrase-MiniLM-L3-v2"):
    """
    Cette fonction permet de calculer les embeddings pour chaque chunk de texte.

    Args:
    chunks: Liste des segments de texte.
    model_name: Modèle SentenceTransformers utilisé.

    Return:
    La liste des Embeddings (sous forme de numpy array).
    """
    model = SentenceTransformer(model_name)
    #model = SentenceTransformer(model_name, cache_folder="./models")

    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)

#NB : Le modèle "all-MiniLM-L6-v2" génère des embeddings de taille 384 (chaque chunk est codé par un vecteur de 384 nombres flottants)

def create_faiss_index(embeddings):
    """
    Cette fonction permet de créer un index FAISS pour la recherche par similarité entre les embeddings.
    L'index est basé sur la distance L2 (a.k.a. Euclidean distance)
    
    Args:
    embeddings: Embeddings des chunks.

    Return:
    Index FAISS prêt à l'emploi par la suite.
    """

    #Vérification des embeddings
    if embeddings.shape[0] == 0:
        raise ValueError("Les embeddings sont vides ! Vérifiez que le texte est bien extrait.")
    
    # Dimension des embeddings (384 pour ce modèle "MiniLM")
    d = embeddings.shape[1]  
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def answer_question(question, chunks, index, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
  
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode([question], convert_to_tensor=False)

    # Debug
    print(f"Question Embedding Shape: {question_embedding.shape}")
    print(f"FAISS Index Shape: {index.ntotal} vectors of dimension {index.d}")

    # FAISS search
    question_embedding = np.array(question_embedding).reshape(1, -1)
    distances, closest_chunk_idx = index.search(question_embedding, k=1)

    closest_chunk = chunks[closest_chunk_idx[0][0]]
    print(f" Le chunk le plus important: {closest_chunk}")

    # Load Q&A model
    qa_pipeline = pipeline("question-answering", model=model_name)

    #Réponse générée
    result = qa_pipeline(question=question, context=closest_chunk)

    # Vérifier si la réponse est vide ou trop courte
    if result["answer"].strip() == "" or len(result["answer"]) < 5:
        print("Réponse peu fiable, essayez de reformuler la question.")
        return "Réponse non trouvée. Essayez de reformuler votre question."

    print(f"Réponse générée: {result}")
    return result["answer"]

