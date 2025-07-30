# CodeMentor AI – ChatGPT for Coding Interviews (Fine-Tuned Flan-T5)

CodeMentor AI is a fine-tuned language model specialized for solving **coding interview questions**, built on top of **TinyLlama-1.1B-Chat**, trained with 20K+ prompts, and deployed with a sleek **ChatGPT-style UI using Streamlit**.

---

##  Features

-  Fine-tuned LLM using HuggingFace Transformers
-  Trained on 20K+ high-quality coding problems (CodeAlpaca dataset)
-  Clean ChatGPT-style frontend built with Streamlit
-  Docker-ready for easy deployment
-  Optimized for local + cloud usage
-  Can run inference via terminal or web UI

---

##  Tech Stack

- `Flan-T5-small` (HuggingFace)
- `Transformers` + `Datasets`
- `Streamlit`
- `Docker` for packaging
- `Render` or `HuggingFace Spaces` for deployment

---

##  Training Details

| Config         | Value                   |
|----------------|-------------------------|
| Model          | `google/flan-t5-small`  |
| Epochs         | 6                       |
| Batch Size     | 1 (with gradient accumulation) |
| Learning Rate  | 5e-5                    |
| Max Length     | 512 tokens              |
| GPU            | GTX 1650 (4GB VRAM)     |
| Total Samples  | ~20,000 examples        |
| Training Time  | ~4 hours                |

---

##  Folder Structure

CodeMentor-AI/
│
├── data/ # Raw + Processed Datasets
├── model/codementor-flan/ # Saved fine-tuned model
├── train/ # Preprocessing + Training scripts
├── app/app.py # Streamlit Chat UI
├── requirements.txt # All dependencies
├── Dockerfile # Docker config
├── render.yaml # Optional Render deployment config


---

##  to Run Locally

```bash
git clone https://github.com/chetan10510/CodeMentor-AI.git
cd CodeMentor-AI
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app/app.py
