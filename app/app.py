import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("model/codementor-flan")
    tokenizer = AutoTokenizer.from_pretrained("model/codementor-flan")
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit page config
st.set_page_config(page_title="CodeMentor AI", page_icon="💻", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>CodeMentor AI</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Your AI Coding Interview Assistant</p>",
    unsafe_allow_html=True
)

# Sidebar info
with st.sidebar:
    st.title("About CodeMentor AI")
    st.info(
        "This assistant is fine-tuned on 20k+ coding problems. "
        "Ask any Data Structures, Algorithms, or Python/Java coding question!"
    )
    st.markdown("---")
    st.markdown("Created by Chetan")

# Chat interface
user_input = st.text_area("Ask your coding question here:", height=150)

if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            prompt = f"### Question:\n{user_input}\n\n### Answer:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=256)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.split("### Answer:")[-1].strip()
            st.success("Response:")
            st.code(answer, language="python")
