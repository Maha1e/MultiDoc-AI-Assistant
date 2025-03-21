import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import numpy as np
def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF page par page.

    :param pdf_path: Chemin du fichier PDF.
    :return: Liste contenant le texte de chaque page.
    """
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")  # Ajoute un texte vide si None
    return text

def chunk_text(text_list, chunk_size=400):
    """
    Découpe un texte en segments (chunks) de taille définie.

    :param text_list: Liste de textes extraits d'un PDF.
    :param chunk_size: Nombre de caractères par chunk.
    :return: Liste de chunks.
    """
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks
def compute_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Calcule les embeddings pour chaque chunk de texte.

    :param chunks: Liste des segments de texte.
    :param model_name: Modèle SentenceTransformers utilisé.
    :return: Embeddings sous forme de numpy array.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)
def create_faiss_index(embeddings):
    """
    Crée un index FAISS pour la recherche par similarité.

    :param embeddings: Embeddings des chunks.
    :return: Index FAISS prêt à l'emploi.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Les embeddings sont vides ! Vérifiez que le texte est bien extrait.")

    d = embeddings.shape[1]  # Dimension des embeddings (384 pour MiniLM)
    print(f"🟢 FAISS Index Created with {embeddings.shape[0]} vectors of dimension {d}")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def answer_question(question, chunks, index, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
  
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode([question], convert_to_tensor=False)

    # 🔍 Debug
    print(f"🔵 Question Embedding Shape: {question_embedding.shape}")
    print(f"🟢 FAISS Index Shape: {index.ntotal} vectors of dimension {index.d}")

    # FAISS search
    question_embedding = np.array(question_embedding).reshape(1, -1)
    distances, closest_chunk_idx = index.search(question_embedding, k=1)

    closest_chunk = chunks[closest_chunk_idx[0][0]]
    print(f"📄 Most relevant chunk: {closest_chunk}")

    # Load Q&A model
    qa_pipeline = pipeline("question-answering", model=model_name)

    # Generate the answer
    result = qa_pipeline(question=question, context=closest_chunk)

    # Vérifier si la réponse est vide ou trop courte
    if result["answer"].strip() == "" or len(result["answer"]) < 5:
        print("⚠️ Réponse peu fiable, essayez de reformuler la question.")
        return "Réponse non trouvée. Essayez de reformuler votre question."

    print(f"🤖 Answer generated: {result}")
    return result["answer"]

# Test rapide
if __name__ == "__main__":
    pdf_text = extract_text_from_pdf("data/example.pdf")
    chunks = chunk_text(pdf_text, chunk_size=400)
    embeddings = compute_embeddings(chunks)
    index = create_faiss_index(embeddings)

    question = "Combien il y a de cochons?"
    response = answer_question(question, chunks, index)
    print(f"🤖 Réponse : {response}")