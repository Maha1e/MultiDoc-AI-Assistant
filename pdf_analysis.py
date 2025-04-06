import PyPDF2
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss


def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF page par page.
    """
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return text


def chunk_text(text_list, chunk_size=400):
    """
    Découpe le texte en segments de taille définie.
    """
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
    return chunks


def compute_embeddings(chunks):
    """
    Calcule les embeddings pour chaque chunk via SentenceTransformers en ligne.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)


def create_faiss_index(embeddings):
    """
    Crée un index FAISS pour recherche par similarité.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Les embeddings sont vides !")

    d = embeddings.shape[1]  # Dimension des embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def answer_question(question, chunks, index, model_id="deepset/roberta-base-squad2"):
    """
    Répond à la question en utilisant le modèle Q&A via transformers.pipeline.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    question_embedding = model.encode([question], convert_to_tensor=False)
    question_embedding = np.array(question_embedding).reshape(1, -1)

    # Find the most relevant chunk using FAISS
    distances, closest_chunk_idx = index.search(question_embedding, k=1)
    closest_chunk = chunks[closest_chunk_idx[0][0]]

    print(f"\n🔍 Contexte sélectionné: {closest_chunk}\n")

    # Use the transformers pipeline for Q&A
    qa_pipeline = pipeline("question-answering", model=model_id)

    result = qa_pipeline(question=question, context=closest_chunk)
    answer = result.get("answer", "")

    if not answer or len(answer.strip()) < 5:
        print("Réponse peu fiable, essayez de reformuler la question.")
        return "Réponse non trouvée. Essayez de reformuler votre question."

    print(f"✅ Réponse: {answer}")
    return answer