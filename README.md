# Credit Scoring RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for credit scoring analysis using the **Flan-T5-Base** model.

## Setup Instructions

### 1. Environment
Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Model Setup (Important!)
Since valid model files are large (~1GB) and excluded from git, you must download them locally before running the app.
We have provided a script to automate this:

```bash
python setup_model.py
```

This script will:
1. Download `google/flan-t5-base` from Hugging Face.
2. Save it to the `./models/flan-t5-base` directory structure expected by the chatbot.

### 4. Data
Place your knowledge base documents (`.txt` or `.docx`) in the `./data` directory.

## Running the Application
Start the API server:
```bash
python server.py
```
The API will be available at `http://localhost:8088`.
