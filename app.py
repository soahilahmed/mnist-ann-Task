import streamlit as st

st.title("TensorFlow Test")

try:
    import tensorflow as tf
    st.success(f"TensorFlow installed: {tf.__version__}")
except Exception as e:
    st.error(str(e))
