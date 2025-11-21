# TAICA Final Project - Generative AI: Text and Image Synthesis Principles and Practice

## Overview

This project implements a **RAG (Retrieval-Augmented Generation) Chatbot** that can answer questions based on PDF documents. It leverages embeddings, vector stores, and a large language model to provide context-aware responses.

---

## Installation

#### 1. Create and activate a conda environment:

```bash
conda create -n RAGChatbot python=3.13 -y
conda activate RAGChatbot
```
#### 2. Install dependencies:
```bash
pip install -q uv
uv pip install -r requirements.txt
```
#### 3. Install PyTorch based on your CUDA version
(Example below uses CUDA 12.4):
```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Usage  

#### 1.	Prepare PDF files:  
Place all relevant PDF files in the `pdf_files` folder. The program will automatically read and index them for RAG.  

#### 2.	Set API Key:  
Create or update `API_KEY.json` with your API key for LLM and embedding access.  

#### 3.	Run the chatbot:  
```bash
python app.py
```
#### 4.	Adjust settings:  
You can configure parameters such as top_k for similarity search, embedding model choice, and RAG prompt templates inside `app.py`.  

## Folder Structure
```bash
├── API_KEY.json      # Your API key configuration
├── app.py            # Main chatbot application
├── chroma_db_ollama
├── gen_results
├── pdf_files/        # Place your PDF documents here
├── README.md
├── requirement.txt
└── src
    ├── __pycache__
    ├── inference.py
    ├── pipeline.py
    ├── RAG.py
    └── utils.py
```

⸻

> Notes  
•	Ensure your PDF documents are readable and well-formatted for optimal RAG performance.  
•	Check app.py for advanced configuration options.  
•	Recommended Python version: 3.13.  
