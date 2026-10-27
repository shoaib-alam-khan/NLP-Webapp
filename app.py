import re
import string
from collections import OrderedDict

import nltk
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer, WordNetLemmatizer

st.set_page_config(
    page_title="NLP Text Processor",
    page_icon="🧠",
    layout="wide",
)

# NLTK data is downloaded on demand so the app can run on a fresh deployment.
@st.cache_resource
def prepare_nltk():
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    }
    available = {}
    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
            available[package] = True
        except LookupError:
            try:
                available[package] = bool(nltk.download(package, quiet=True))
            except Exception:
                available[package] = False
    return available

resource_status = prepare_nltk()

def tokenize_words(text):
    """Return word tokens while keeping contractions as one token."""
    return re.findall(r"\b[\w]+(?:['’][\w]+)*\b", text, flags=re.UNICODE)

def tokenize_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]

def remove_punctuation(text):
    # Unicode-aware punctuation removal, retaining letters, digits, whitespace and apostrophes.
    return "".join(ch for ch in text if not (unicodedata_category(ch) == "P"))

def unicodedata_category(ch):
    import unicodedata
    return unicodedata.category(ch)

def wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    if resource_status.get("averaged_perceptron_tagger_eng") or resource_status.get("averaged_perceptron_tagger"):
        try:
            tagged = nltk.pos_tag(words)
            return [lemmatizer.lemmatize(word, wordnet_pos(tag)) for word, tag in tagged], True
        except Exception:
            pass
    # Fallback keeps the app usable if NLTK corpora/tagger downloads are unavailable.
    # This is intentionally simple and is not a full linguistic lemmatizer.
    def simple_lemma(word):
        low = word.lower()
        if len(low) > 4 and low.endswith("ies"):
            return low[:-3] + "y"
        if len(low) > 4 and low.endswith("ing"):
            stem = low[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        if len(low) > 3 and low.endswith("ed"):
            return low[:-2]
        if len(low) > 3 and low.endswith("s") and not low.endswith(("ss", "us", "is")):
            return low[:-1]
        return low
    return [simple_lemma(word) for word in words], False

def make_one_hot(tokens):
    """Create one row per token and one column per unique vocabulary item."""
    vocabulary = list(OrderedDict.fromkeys(tokens))
    rows = []
    for token in tokens:
        rows.append([1 if token == vocab_word else 0 for vocab_word in vocabulary])
    return vocabulary, rows

st.title("🧠 NLP Text Processor")
st.write("Enter text, choose an NLP operation, and view the processed output.")
st.caption("Operations: Tokenization · Stemming · Lemmatization · Punctuation Removal · One-Hot Encoding")

with st.container(border=True):
    user_text = st.text_area(
        "Enter your text",
        placeholder="Example: The students are studying and playing games!",
        height=160,
        key="user_text",
    )
    operation = st.selectbox(
        "Choose an operation",
        [
            "Tokenization",
            "Stemming",
            "Lemmatization",
            "Punctuation Removal",
            "One-Hot Encoding",
        ],
    )
    process = st.button("⚙️ Process Text", type="primary", use_container_width=True)

if process:
    if not user_text.strip():
        st.warning("Please enter some text before processing.")
    else:
        st.subheader(f"Results: {operation}")
        if operation == "Tokenization":
            words = tokenize_words(user_text)
            sentences = tokenize_sentences(user_text)
            left, right = st.columns(2)
            with left:
                st.metric("Word tokens", len(words))
                st.write(words)
            with right:
                st.metric("Sentences", len(sentences))
                st.write(sentences)

        elif operation == "Stemming":
            words = tokenize_words(user_text)
            stemmer = PorterStemmer()
            results = [stemmer.stem(word) for word in words]
            st.dataframe(
                pd.DataFrame({"Original token": words, "Stem": results}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Stemming trims word endings; the result is not always a valid dictionary word.")

        elif operation == "Lemmatization":
            words = tokenize_words(user_text)
            results, full_nltk = lemmatize_words(words)
            st.dataframe(
                pd.DataFrame({"Original token": words, "Lemma": results}),
                use_container_width=True,
                hide_index=True,
            )
            if full_nltk:
                st.success("Used NLTK WordNet lemmatization with part-of-speech tagging.")
            else:
                st.warning("NLTK language resources were unavailable. A simple fallback was used; results may be less accurate.")

        elif operation == "Punctuation Removal":
            cleaned = "".join(ch for ch in user_text if not unicodedata_category(ch).startswith("P"))
            st.text_area("Text without punctuation", value=cleaned, height=140)
            st.caption("Letters, numbers, spaces, and other non-punctuation characters are retained.")

        elif operation == "One-Hot Encoding":
            tokens = tokenize_words(user_text)
            if not tokens:
                st.info("No word tokens were found to encode.")
            elif len(tokens) > 100:
                st.warning("For readability, One-Hot Encoding is limited to 100 tokens per run. Please shorten the input.")
            else:
                normalized = [token.lower() for token in tokens]
                vocabulary, matrix = make_one_hot(normalized)
                st.write("Each row represents a token occurrence. Each column represents a unique vocabulary word.")
                st.metric("Token occurrences", len(normalized))
                st.metric("Unique vocabulary words", len(vocabulary))
                table = pd.DataFrame(matrix, columns=vocabulary)
                table.insert(0, "Token", normalized)
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.caption("Repeated words share the same one-hot vector. This matrix records which vocabulary item each token represents.")

        # Offer a downloadable text summary for the selected operation.
        if operation == "Tokenization":
            output = "Word tokens:\n" + "\n".join(tokenize_words(user_text)) + "\n\nSentences:\n" + "\n".join(tokenize_sentences(user_text))
        elif operation == "Stemming":
            words = tokenize_words(user_text)
            output = "\n".join(f"{w}\t{PorterStemmer().stem(w)}" for w in words)
        elif operation == "Lemmatization":
            words = tokenize_words(user_text)
            lemmas, _ = lemmatize_words(words)
            output = "\n".join(f"{w}\t{lemma}" for w, lemma in zip(words, lemmas))
        elif operation == "Punctuation Removal":
            output = "".join(ch for ch in user_text if not unicodedata_category(ch).startswith("P"))
        else:
            tokens = [t.lower() for t in tokenize_words(user_text)]
            if len(tokens) <= 100:
                vocab, matrix = make_one_hot(tokens)
                table = pd.DataFrame(matrix, columns=vocab)
                table.insert(0, "Token", tokens)
                output = table.to_csv(index=False)
            else:
                output = "Input exceeded the 100-token limit for One-Hot Encoding."
        st.download_button(
            "⬇️ Download results (.txt)",
            data=output,
            file_name="nlp_results.txt",
            mime="text/plain",
        )

st.divider()
st.caption("Developed by Shoaib Alam Khan")
