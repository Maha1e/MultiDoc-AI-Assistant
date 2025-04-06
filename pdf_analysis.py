import PyPDF2
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


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

def create_index_sklearn(embeddings):
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(embeddings)
    return nn

def answer_question(question, chunks, index, model_id="deepset/roberta-base-squad2"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    question_embedding = model.encode([question])
    
    _, closest_chunk_idx = index.kneighbors(question_embedding)
    closest_chunk = chunks[closest_chunk_idx[0][0]]
    
    print(f"\n🔍 Contexte sélectionné: {closest_chunk}\n")
    
    qa_pipeline = pipeline("question-answering", model=model_id)
    result = qa_pipeline(question=question, context=closest_chunk)
    
    return result.get("answer", "Pas de réponse.")