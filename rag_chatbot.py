import os
import csv
import json
import requests
from datetime import datetime
from collections import deque
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Load the Knowledge Base
print("Loading data from ./data directory...")
documents = []
data_folder = "./data"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"Created {data_folder} directory. Please add your documents there.")

for file in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file)
    
    try:
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded: {file}")
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded: {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not documents:
    print("No documents found in ./data folder!")

# 2. Split text into chunks (better for retrieval)
text_splitter = CharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=40
    )
docs = text_splitter.split_documents(documents)

# 3. Create Embeddings
# We use a small, efficient embedding model compatible with local setups
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create Vector Store (FAISS)
# This indexes the documents so we can search them
if docs:
    print("Updating Vector Store...")
    db = FAISS.from_documents(docs, embeddings)
else:
    db = None



# 5. Initialize Memory
# memory is now a dict: {user_id: deque(maxlen=6)}
user_memories = {}

# ... [Load models] ...
# Initialize Flan-T5-Base
print("Checking for local model...")
local_model_path = "./models/flan-t5-large"

# Check if looks like a model directory (has config.json)
if os.path.isdir(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
    print(f"Loading from local folder: {local_model_path}")
    model_id = local_model_path
else:
    raise FileNotFoundError(f"Model not found in {local_model_path}. Please place your model files there.")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,      
    truncation=True,
    do_sample=True,          
    temperature=0.2,         
    top_p=0.9,
    num_beams=1
)


llm = HuggingFacePipeline(pipeline=pipe)

# --- LLM CONFIGURATION ---
# Choose your LLM: 'ollama', 'gemini', or 'local'
LLM_CHOICE = 'local'  # Recommended: Use Ollama for best performance

# Ollama Configuration
OLLAMA_BASE_URL = "http://10.10.0.147:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"  # Your downloaded model

# Gemini Cloud Configuration
GEMINI_API_KEY = 'AIzaSyDPjeUkZ2_oFice-A2FBA5uvwDMYqmvBzo'
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

def get_ollama_response(prompt):
    """
    Calls the local Ollama API to generate a response using Llama 3.2.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower temperature for more focused responses
            "top_p": 0.9,
            "num_predict": 512   # Max tokens to generate
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "response" in data:
            return data["response"].strip()
        else:
            return "Error: Ollama API returned no content."
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running (try 'ollama serve' in terminal)."
    except Exception as e:
        return f"Ollama API Error: {str(e)}"

def get_gemini_response(prompt):
    """
    Calls the Google Gemini API to generate a response.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return "Error: Gemini API returned no content."
            
    except Exception as e:
        return f"Gemini API Error: {str(e)}"
# -----------------------------------

def get_user_memory(user_id):
    if user_id not in user_memories:
        user_memories[user_id] = deque(maxlen=6)
    return user_memories[user_id]

def log_interaction(user_id, question, answer):
    """
    Logs the user interaction to a CSV file for performance monitoring.
    """
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "chat_history.csv")
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not file_exists:
                writer.writerow(["Timestamp", "User ID", "Question", "Answer"])
            
            writer.writerow([datetime.now().isoformat(), user_id, question, answer])
    except Exception as e:
        print(f"Failed to log interaction: {e}")

# Define the Retrieval Function
def ask_credit_bot(query, user_id="default"):
    print(f"\nQuery: {query} (User: {user_id})")
    
    # Retrieve relevant docs
    # Check if db exists (it might be None if no docs were found)
    context_text = ""
    if db:
        # Increased k to 5 to capture more context for multiple metrics
        relevant_docs = db.similarity_search(query, k=5)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
    else:
        context_text = "No knowledge base documents found."
    
    print(f"\n Context is : {context_text}")

    
    # Get specific user memory
    memory = get_user_memory(user_id)
    
    ##history_text = "\n".join(memory)

    ##print(f"\n History is : {history_text}")
    
    # Construct a prompt for RAG
    # Specifically instructing the model to handle metric-by-metric evaluation
    prompt = f"""You are a professional credit scoring analyst assistant.
    
    TASK:
    Analyze the 'Customer Scores' provided in the Question. 
    Use the Context below to determine if each metric score is 'Good', 'Average', or 'Bad' based on the defined thresholds.
    
    OUTPUT FORMAT:
    1. Provide a clear, concise paragraph summarizing the overall credit health of the customer.

    Context (Knowledge Base):
    {context_text}

    Question/Input:
    {query}

    Evaluation and Summary:"""    

    # OUTPUT FORMAT:
    # 1. Provide a brief evaluation for each metric mentioned.
    # 2. Conclude with a clear, concise paragraph summarizing the overall credit health of the customer.
    # 3. If a metric's threshold is not found in the Context, state 'Threshold not specified' for that metric.

    # Generate answer using the selected LLM
    if LLM_CHOICE == 'ollama':
        print("Model: Ollama (Llama 3.2)")
        response = get_ollama_response(prompt)
    elif LLM_CHOICE == 'gemini':
        print("Model: Gemini Cloud")
        response = get_gemini_response(prompt)
    else:  # 'local' or fallback
        print("Model: Local Flan-T5")
        response = llm.invoke(prompt)
    
    # Update memory
    memory.append(f"User: {query}")
    memory.append(f"Bot: {response}")
    
    # Log interaction to file
    log_interaction(user_id, query, response)

    print(f"\n Answer is : {response}")
    return response


if __name__ == "__main__":
    print("RAG Module Loaded.")

