"""Streamlit user interface for WordWave."""

from __future__ import annotations

import streamlit as st

from src.artifacts import load_artifacts
from src.config import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_GENERATION_LENGTH,
    DEFAULT_SEED_TEXT,
    MAX_GENERATION_LENGTH,
)
from src.generation import beam_search_decoder, evaluate_bleu
from src.metrics import evaluate_model_metrics


st.set_page_config(page_title="WordWave - Next Word Prediction")


@st.cache_resource(show_spinner=True)
def get_runtime():
    return load_artifacts()


@st.cache_data(show_spinner=True)
def get_evaluation_metrics(model, tokenizer, max_len):
    return evaluate_model_metrics(model, tokenizer, max_len)


def render_app():
    model, tokenizer, max_len, _ = get_runtime()

    with st.spinner("Evaluating model metrics..."):
        top_5_acc, perplexity_score = get_evaluation_metrics(model, tokenizer, max_len)

    st.title("WordWave: Next Word Prediction")
    st.write("Generate coherent text using a trained BiLSTM + Attention model.")

    st.sidebar.header("Model Evaluation")
    st.sidebar.metric("Top-5 Accuracy", f"{top_5_acc * 100:.2f}%")
    st.sidebar.metric("Perplexity", f"{perplexity_score:.2f}")
    st.sidebar.caption("Evaluated on a subset of 5,000 examples")

    seed_text = st.text_input("Enter your seed text", value=DEFAULT_SEED_TEXT)
    next_words = st.slider(
        "How many words to generate?",
        min_value=1,
        max_value=MAX_GENERATION_LENGTH,
        value=DEFAULT_GENERATION_LENGTH,
    )

    if st.button("Generate"):
        generated = beam_search_decoder(
            model,
            tokenizer,
            seed_text,
            beam_width=DEFAULT_BEAM_WIDTH,
            next_words=next_words,
            max_len=max_len,
        )

        st.markdown("### Generated Text")
        st.success(generated)

        reference = st.text_input("Optional: Enter reference sentence to compute BLEU score")
        if reference:
            bleu = evaluate_bleu(reference, generated)
            st.metric("BLEU Score", f"{bleu:.4f}")
        else:
            st.info("BLEU Score skipped (no reference provided)")
