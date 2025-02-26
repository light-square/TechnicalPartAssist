# TechnicalPartAssist
Find similar parts and get recommendations. Example: "Find me a similar part to A10"
# Technical Part Matching Chatbot

A GenAI-powered chatbot that combines LLM inference (Mistral-7B) with technical part similarity matching. Users can:
- Chat naturally with the LLM
- Find alternative parts based on technical specifications
- View detailed comparisons of parts

[![Demo GIF](demo.gif)](https://drive.google.com/file/d/128LQeDj_GK95Oig3s2ou1MINGYuJLwi9/view?usp=share_link)

![Screenshot](/Technical%20Parts%20Assitant.png)

## Features
- **Chatbot Interface**:
  - Mistral-7B GGUF model inference
  - GPU acceleration via Apple Metal
  - Streamlit UI with chat history
- **Part Matching**:
  - Hybrid similarity (40% text, 60% specs)
  - Detailed comparison of original and matched parts
  - CSV dataset integration (`Parts.csv`)

## Tech Stack
- **LLM**: Mistral-7B-Instruct (Q4_K_M quantized)
- **Frameworks**: `llama-cpp-python`, Streamlit
- **NLP**: Sentence Transformers, scikit-learn
- **Hardware**: Optimized for Apple Silicon M1/M2/M3 GPUs

## Setup

### Prerequisites
- Python 3.10+
- Xcode Command Line Tools (Mac)
- [Mistral-7B GGUF Model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/light-square/TechnicalPartAssist.git
   cd TechnicalPartAssist

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download the Mistral-7B GGUF model:
   ```bash
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O model/mistral-7b.Q4_K_M.gguf

### Running the App
```bash
python -m streamlit run app.py