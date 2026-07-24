import os

import pandas as pd
import streamlit as st

from src.search.hybrid_search import HybridSearchEngine
from src.utils.config import settings

st.set_page_config(page_title="Enterprise Semantic Search", layout="wide")

DOCUMENTS_CSV = os.path.join("input", "documents.csv")


@st.cache_resource
def load_engine():
    engine = HybridSearchEngine()
    engine.build_index(DOCUMENTS_CSV)
    return engine


st.title("Enterprise Semantic Search Platform")
st.caption("Ask a natural-language question and find the most relevant document sections.")

engine = load_engine()

with st.sidebar:
    st.header("Search Settings")
    mode = st.radio("Search mode", ["hybrid", "semantic"], index=0)
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    if mode == "hybrid":
        alpha = st.slider(
            "Vector weight (alpha)", min_value=0.0, max_value=1.0, value=settings.HYBRID_ALPHA
        )
    st.metric("Indexed chunks", engine.store.size)

query = st.text_input("Enter your search query", placeholder="What is the leave policy?")

if st.button("Search", type="primary") or query:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        if mode == "hybrid":
            results = engine.search(query, top_k=top_k, alpha=alpha)
        else:
            results = engine.semantic_search(query, top_k=top_k)

        if not results:
            st.info("No matching results found.")
        else:
            table_rows = [
                {
                    "Rank": i + 1,
                    "Document": r["title"],
                    "Category": r["category"],
                    "Similarity Score": round(r["score"], 4),
                    "Matching Text": r["text"][:200],
                }
                for i, r in enumerate(results)
            ]
            df = pd.DataFrame(table_rows)
            st.dataframe(df, width="stretch", hide_index=True)

            st.download_button(
                "Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="search_results.csv",
                mime="text/csv",
            )

            st.subheader("Details")
            for i, r in enumerate(results):
                with st.expander(f"#{i + 1} — {r['title']} (score: {round(r['score'], 4)})"):
                    st.write(r["text"])
