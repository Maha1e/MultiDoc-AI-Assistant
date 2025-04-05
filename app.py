

# Imports des modules
from text_analysis import analyze_text
from pdf_analysis import extract_text_from_pdf, chunk_text, compute_embeddings, create_faiss_index, answer_question
from image_analysis import load_image, generate_caption, summarize_caption
import streamlit as st
import tempfile
from PIL import Image
import torch
import sys
import types
# Fix pour éviter l'erreur "__path__._path" dans torch.classes sur Windows avec Streamlit
if isinstance(torch.classes, types.ModuleType) and not hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []
st.set_page_config(page_title="Analyseur de contenu", layout="centered")
# 🌟 Titre personnalisé et stylé (remplace st.title)
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>
        🧠 MultiDoc AI Assistant : Texte, PDF, Image
    </h1>
""", unsafe_allow_html=True)

#st.title("🧠 MultiDoc AI Assistant : Texte, PDF, Image")

#option = st.sidebar.radio("Choisissez un type de contenu à analyser :", ["Texte", "PDF", "Image"])
with st.sidebar:
    st.markdown("## 📂 Sélection du contenu")
    st.markdown("---")
    st.markdown("### 📌 Type d’analyse")
    option = st.radio("", ["📝 Texte", "📄 PDF", "🖼️ Image"])


# Option 1 :  Analyse de texte 
if option == "📝 Texte":
    text_input = st.text_area("Entrez votre texte ici :")
    if st.button("Analyser le texte"):
        if text_input.strip():
            summary = analyze_text(text_input)
            st.success("Résumé généré :")
            st.write(summary)
        else:
            st.warning("Merci d'entrer un texte à analyser.")

# Option 2 : Analyse de PDF
elif option == "📄 PDF":
    uploaded_pdf = st.file_uploader("Téléversez un fichier PDF", type="pdf")
    question = st.text_input("Posez une question sur le contenu du PDF :")

    if uploaded_pdf and question.strip():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        st.info("🔍 Extraction et traitement du PDF en cours...")
        pdf_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pdf_text)
        embeddings = compute_embeddings(chunks)
        index = create_faiss_index(embeddings)
        answer = answer_question(question, chunks, index)
        st.success("Réponse à votre question :")
        st.write(answer)

# Option 3 : Analyse d’image
elif option == "🖼️ Image":
    uploaded_img = st.file_uploader("Téléversez votre image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = load_image(uploaded_img)
        st.image(image, caption="Image chargée", use_column_width=True)

        if st.button("Analyser l'image"):
            caption = generate_caption(image)
            st.info(f"Description générée : {caption}")
            summary = summarize_caption(caption)
            st.success("Résumé de la description :")
            st.write(summary)
