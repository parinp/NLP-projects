import streamlit as st
from summarization import extractive_summary
from abstractive_summarization import abstractive_summary

# Streamlit UI
st.title("📝 NLP Summarization App")

# Text Input
text_input = st.text_area("Enter text to summarize:", height=200)

# File Upload Option
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

# Summary Type Selection
summary_type = st.radio("Select Summarization Method:", ("Extractive", "Abstractive"))

# Summary Generation
if st.button("Generate Summary"):
    if text_input:
        if summary_type == "Extractive":
            summary = extractive_summary(text_input, num_sentences=3)
        else:
            summary = abstractive_summary(text_input, max_length=100, min_length=30)

        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text or upload a file!")
