"""Streamlit user interface for WordWave."""

from __future__ import annotations

import streamlit as st

from src.artifacts import load_artifacts
from src.generation import beam_search_decoder, evaluate_bleu
from src.settings import load_settings

SETTINGS = load_settings()


st.set_page_config(page_title="WordWave - Next Word Prediction")


@st.cache_resource(show_spinner=True)
def get_runtime():
    return load_artifacts()


def render_app():
    try:
        model, vocabulary, max_len, metrics, _ = get_runtime()
    except FileNotFoundError:
        st.title("WordWave: Next Word Prediction")
        st.warning(
            "Train the model first so the PyTorch artifacts exist before launching the app."
        )
        return

    st.title("WordWave: Next Word Prediction")
    st.write("Generate coherent text using a trained BiLSTM + attention PyTorch model.")

    st.sidebar.header("Model Evaluation")
    st.sidebar.metric(
        "Top-5 Accuracy", f"{metrics.get('validation_top_k', 0.0) * 100:.2f}%"
    )
    st.sidebar.metric(
        "Perplexity", f"{metrics.get('validation_perplexity', float('inf')):.2f}"
    )
    st.sidebar.caption("Evaluated on the saved validation split during training")

    seed_text = st.text_input(
        "Enter your seed text", value=SETTINGS.runtime.default_seed_text
    )
    next_words = st.slider(
        "How many words to generate?",
        min_value=1,
        max_value=SETTINGS.runtime.max_generation_length,
        value=SETTINGS.runtime.default_generation_length,
    )

    if st.button("Generate"):
        generated = beam_search_decoder(
            model,
            vocabulary,
            seed_text,
            beam_width=SETTINGS.runtime.default_beam_width,
            next_words=next_words,
            max_len=max_len,
        )

        st.markdown("### Generated Text")
        st.success(generated)

        reference = st.text_input(
            "Optional: Enter reference sentence to compute BLEU score"
        )
        if reference:
            bleu = evaluate_bleu(reference, generated)
            st.metric("BLEU Score", f"{bleu:.4f}")
        else:
            st.info("BLEU Score skipped (no reference provided)")
