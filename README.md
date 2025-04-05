# 🧠 MultiDoc AI Assistant : Texte, PDF, Image

**Projet IA - CYTech**\
Application Streamlit permettant d’analyser automatiquement des contenus textuels, documents PDF et images à l’aide de modèles open source (Hugging Face).

---

## Fonctionnalités

🔹 **Analyse de texte**

- Résumé automatique de textes en français ou anglais
- Basé sur le modèle `facebook/bart-large-cnn`

🔹 **Analyse de documents PDF**

- Extraction des textes page par page
- Découpage en chunks, vectorisation avec `all-MiniLM-L6-v2`
- Recherche par similarité (FAISS) + Q&R interface

🔹 **Analyse d’images**

- Génération de description automatique via BLIP
- Résumé ou extraction d’information depuis la caption générée

---

## 💻 Interface utilisateur



- Interface simple via `Streamlit`
- Sidebar de sélection du type de contenu
- Résultat affiché instantanément

---

## 🛠️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Nicolas-Usson/big-data-cy3-RAG.git
cd big-data-cy3-RAG
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---



## ▶️ Lancer l'application

```bash
streamlit run app.py
```

---


## Modèles utilisés

| Tâche         | Modèle                                                  |
| ------------- | ------------------------------------------------------- |
| Résumé texte  | `facebook/bart-large-cnn`                               |
| Embeddings    | `all-MiniLM-L6-v2`                                      |
| Q&A sur PDF   | `bert-large-uncased-whole-word-masking-finetuned-squad` |
| Caption image | `Salesforce/blip-image-captioning-base`                 |

---

## Structure du projet

```
📆 BIG-DATA-CY3-RAG/
├── app.py
├── text_analysis.py
├── pdf_analysis.py
├── image_analysis.py
├── requirements.txt
└── README.md
```

---

##  Auteur

- CYTech Student – 2025
- Projet dans le cadre du cours **LLM & Applications Open Source**

---


