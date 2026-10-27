# NLP Text Processor (Streamlit)

A small Streamlit NLP app with five selectable operations:
- Tokenization
- Stemming
- Lemmatization
- Punctuation Removal
- One-Hot Encoding

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

NLTK WordNet and POS-tagging resources are downloaded on first run when needed. If those downloads are unavailable, the app uses a simple fallback for lemmatization and clearly labels it.


Footer: Developed by Shoaib Alam Khan
