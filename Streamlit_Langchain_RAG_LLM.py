import os
import pandas as pd
import numpy as np
from typing import List

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# -----------------------------------
# Global Config
# -----------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # for vector embedding
LLM_MODEL = "gpt2"  # GPT-2 is publicly available; no login needed
CSV_PATH = "hospital_charges.csv"

# -----------------------------------
# Data + FAISS Index
# -----------------------------------

def generate_mock_data(csv_path: str):
    """Generate a small synthetic dataset resembling hospital charge data."""
    if not os.path.exists(csv_path):
        data = {
            "Provider_Name": [
                "General Hospital A",
                "Specialized Cardiac Center B",
                "Rural Community Clinic C",
                "Urban Medical Facility D"
            ],
            "DRG_Definition": [
                "Cardiac Procedures",
                "Orthopedic Procedures",
                "Cardiac Procedures",
                "Pediatric Procedures"
            ],
            "Average_Covered_Charges": [15000, 12000, 18000, 8000]
        }
        df_mock = pd.DataFrame(data)
        df_mock.to_csv(csv_path, index=False)
        print(f"Mock data created at {csv_path}.")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset as a pandas DataFrame."""
    if not os.path.exists(csv_path):
        generate_mock_data(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df = df.dropna(axis=0, how='any')
    return df


def create_faiss_index(df: pd.DataFrame, text_cols: List[str]):
    """Create a FAISS index from relevant text columns."""
    corpus = df[text_cols].astype(str).agg(' '.join, axis=1).tolist()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(corpus, show_progress_bar=False).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    ids = np.array(range(len(corpus)))
    index.add_with_ids(embeddings, ids)

    return index, corpus

# -----------------------------------
# LLM + RAG Query
# -----------------------------------

def load_llm():
    """Load GPT-2 model & tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

    # GPT-2 does not have a pad token, so we set pad_token_id = eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def rag_query(query: str, index: faiss.IndexIDMap, corpus: List[str], top_k: int, tokenizer, model) -> str:
    """
    RAG pipeline:
      1. Embed query
      2. Retrieve top-k from FAISS
      3. Prompt GPT-2 with context + user query
      4. Generate + return answer
    """
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_emb = embedder.encode([query], show_progress_bar=False).astype('float32')

    distances, ids = index.search(query_emb, top_k)

    retrieved_texts = [corpus[i] for i in ids[0]]
    context = "\n".join(retrieved_texts)

    prompt = (
        f"Below is some hospital charge data context from the top {top_k} relevant entries:\n"
        f"{context}\n\n"
        f"Answer the user's query in one short sentence: {query}\n"
        f"Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# -----------------------------------
# Streamlit App
# -----------------------------------

def main():
    st.title("RAG + GPT-2 Demo (Healthcare Example)")
    st.write("""This app demonstrates a simple retrieval-augmented generation pipeline.
    **No online accounts** or API keys required.
    """)

    # Lazy load data/index/model in session_state
    if "df" not in st.session_state:
        st.session_state.df = load_data(CSV_PATH)
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index, st.session_state.corpus = create_faiss_index(
            st.session_state.df,
            text_cols=["Provider_Name", "DRG_Definition", "Average_Covered_Charges"]
        )
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer, st.session_state.model = load_llm()

    user_query = st.text_input("Enter your question:", "Which provider has the highest average covered charges for cardiac procedures?")
    top_k = st.slider("Number of relevant results to retrieve (top_k)", min_value=1, max_value=5, value=2)

    if st.button("Submit Query"):
        with st.spinner("Generating response..."):
            answer = rag_query(
                query=user_query,
                index=st.session_state.faiss_index,
                corpus=st.session_state.corpus,
                top_k=top_k,
                tokenizer=st.session_state.tokenizer,
                model=st.session_state.model
            )
        st.subheader("Answer")
        st.write(answer)

    st.write("\n\n---")
    st.write("**Data Preview**")
    st.dataframe(st.session_state.df)

if __name__ == "__main__":
    main()
