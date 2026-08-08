"""Streamlit user interface for WordWave."""

from __future__ import annotations

import streamlit as st

from src.artifacts import load_artifacts
from src.generation import evaluate_bleu, generate_text
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

    strategy_options = ["beam_search", "sample"]
    default_strategy = (
        SETTINGS.runtime.default_decoding_strategy
        if SETTINGS.runtime.default_decoding_strategy in strategy_options
        else "beam_search"
    )
    decoding_strategy = st.selectbox(
        "Decoding strategy",
        options=strategy_options,
        index=strategy_options.index(default_strategy),
        format_func=lambda value: {
            "beam_search": "Beam search",
            "sample": "Sampling (Top-k/Top-p/Temp)",
        }[value],
    )
    beam_width = st.slider(
        "Beam width",
        min_value=1,
        max_value=10,
        value=SETTINGS.runtime.default_beam_width,
        disabled=decoding_strategy != "beam_search",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=SETTINGS.runtime.default_sampling_temperature,
        step=0.1,
        disabled=decoding_strategy == "beam_search",
    )
    top_p = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=SETTINGS.runtime.default_top_p,
        step=0.05,
        disabled=decoding_strategy == "beam_search",
    )

    top_k = st.slider(
        "Top-k",
        min_value=0,
        max_value=100,
        value=SETTINGS.runtime.default_top_k,
        disabled=decoding_strategy == "beam_search",
    )
    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=3.0,
        value=SETTINGS.runtime.default_repetition_penalty,
        step=0.1,
    )
    no_repeat_ngram_size = st.slider(
        "No-Repeat N-Gram Size",
        min_value=0,
        max_value=5,
        value=SETTINGS.runtime.default_no_repeat_ngram_size,
    )

    if st.button("Generate"):
        with st.spinner("Generating sequence..."):
            generated = generate_text(
                model,
                vocabulary,
                seed_text,
                next_words=next_words,
                max_len=max_len,
                beam_width=beam_width,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                strategy=decoding_strategy,
            )
        st.session_state["generated_text"] = generated

    if "generated_text" in st.session_state:
        st.markdown("### Generated Text")
        st.success(st.session_state["generated_text"])

        reference = st.text_input(
            "Optional: Enter reference sentence to compute BLEU score"
        )
        if reference:
            bleu = evaluate_bleu(reference, st.session_state["generated_text"])
            st.metric("BLEU Score", f"{bleu:.4f}")
        else:
            st.info("BLEU Score skipped (no reference provided)")
