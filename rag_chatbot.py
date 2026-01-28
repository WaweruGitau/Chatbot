import os
import csv
import json
import requests
import time
from datetime import datetime
from collections import deque
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- LLM CONFIGURATION ---
# Using Ollama Model
OLLAMA_BASE_URL = "http://10.10.0.147:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"  # Your downloaded model

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



# Initialize Memory
# memory is now a dict: {user_id: deque(maxlen=6)}
user_memories = {}

# (Configuration moved to top of file)

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
            "temperature": 0.1,  # Lowered for more consistent, concise output
            "top_p": 0.9,
            "num_predict": 256   # Reduced token limit to improve speed
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
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

def get_ollama_response_stream(prompt):
    """
    Yields chunks of the response from Ollama API provided stream=True.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 256
        }
    }
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    # Parse the JSON chunk
                    body = json.loads(line)
                    response_part = body.get("response", "")
                    if response_part:
                        yield response_part
                    
                    if body.get("done", False):
                        break
                        
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to Ollama. Make sure Ollama is running."
    except Exception as e:
        yield f"Ollama API Error: {str(e)}"



# -----------------------------------

def get_user_memory(user_id):
    if user_id not in user_memories:
        user_memories[user_id] = deque(maxlen=6)
    return user_memories[user_id]

def log_interaction(user_id, question, answer, metrics=None):
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
                writer.writerow(["Timestamp", "User ID", "Question", "Answer", "Embed Time", "Search Time", "Retrieval Time", "Prompt Time", "LLM Time", "Total Time"])
            
            row = [datetime.now().isoformat(), user_id, question, answer]
            if metrics:
                row.extend([
                    metrics.get("embed_time", 0),
                    metrics.get("search_time", 0),
                    metrics.get("retrieval_time", 0),
                    metrics.get("prompt_time", 0),
                    metrics.get("llm_time", 0),
                    metrics.get("total_time", 0)
                ])
            else:
                row.extend([0, 0, 0, 0, 0, 0])
                
            writer.writerow(row)
    except Exception as e:
        print(f"Failed to log interaction: {e}")

# Define the Retrieval Function
def ask_credit_bot(query, user_id="default"):
    start_total = time.time()
    print(f"\nQuery: {query} (User: {user_id})")
    
    # 1. Retrieval Stage (Split into Embedding and Search)
    context_text = ""
    retrieval_time = 0
    embed_time = 0
    search_time = 0
    
    if db:
        # Measure Embedding time
        start_embed = time.time()
        # We access the underlying embedding model to embed the query separately
        query_vec = embeddings.embed_query(query)
        embed_time = time.time() - start_embed
        
        # Measure Search time
        start_search = time.time()
        relevant_docs = db.similarity_search_by_vector(query_vec, k=5)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
        search_time = time.time() - start_search
        
        retrieval_time = embed_time + search_time
    else:
        context_text = "No knowledge base documents found."
    
    print(f"\nEmbedding took: {embed_time:.4f}s")
    print(f"Vector Search took: {search_time:.4f}s")
    # print(f"\n Context is : {context_text}")

    # 2. Memory and Prompt Stage
    start_prompt = time.time()
    memory = get_user_memory(user_id)
    
    # Updated Prompt for extreme brevity and structure
    prompt = f"""You are a professional credit scoring analyst assistant.
    Your task is to provide a highly concise evaluation. Do NOT explain every metric.
    Only highlight the overall health and the most critical findings.

    Rules:
    - Use bullet points.
    - Be direct. No filler phrases like "Based on the scores provided..."
    - Maximum 4-5 bullet points total.

    OUTPUT FORMAT:
    - **Overall Rating**: [Good/Average/Poor]
    - **Key Strengths**: (Short bullet points)
    - **Critical Risks**: (Short bullet points)
    - **Final Verdict**: (1 sentence summary)

    Context (Knowledge Base):
    {context_text}

    Customer Input:
    {query}

    Concise Evaluation:"""    
    prompt_time = time.time() - start_prompt

    # 3. LLM Generation Stage
    start_llm = time.time()
    print(f"Model: Ollama ({OLLAMA_MODEL})")
    response = get_ollama_response(prompt)
    llm_time = time.time() - start_llm
    
    total_time = time.time() - start_total

    print(f"LLM Generation took: {llm_time:.4f}s")
    print(f"Total time: {total_time:.4f}s")
    
    # Update memory
    memory.append(f"User: {query}")
    memory.append(f"Bot: {response}")
    
    # Log interaction to file (including timing)
    log_interaction(user_id, query, response, metrics={
        "embed_time": embed_time,
        "search_time": search_time,
        "retrieval_time": retrieval_time,
        "prompt_time": prompt_time,
        "llm_time": llm_time,
        "total_time": total_time
    })

    print(f"\nAnswer is: {response}")
    
    # Returning both response and metrics for potential API use
    return {
        "answer": response,
        "metrics": {
            "embedding_seconds": round(embed_time, 4),
            "search_seconds": round(search_time, 4),
            "prompt_seconds": round(prompt_time, 4),
            "llm_seconds": round(llm_time, 4),
            "total_seconds": round(total_time, 4)
        }
    }

def ask_credit_bot_stream(query, user_id="default"):
    """
    Streaming version of the ask_credit_bot function.
    Yields chunks of the answer and logs performance metrics.
    """
    start_total = time.time()
    
    # 1. Retrieval Stage
    start_embed = time.time()
    embed_time = 0
    search_time = 0
    retrieval_time = 0
    context_text = ""
    
    if db:
        # Measure Embedding time
        query_vec = embeddings.embed_query(query)
        embed_time = time.time() - start_embed
        
        # Measure Search time
        start_search = time.time()
        relevant_docs = db.similarity_search_by_vector(query_vec, k=5)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
        search_time = time.time() - start_search
        
        retrieval_time = embed_time + search_time
    else:
        context_text = "No knowledge base documents found."

    # 2. Memory and Prompt Stage
    start_prompt = time.time()
    memory = get_user_memory(user_id)
    
    prompt = f"""You are a professional credit scoring analyst assistant.
    Your task is to provide a highly concise evaluation. Do NOT explain every metric.
    Only highlight the overall health and the most critical findings.

    Rules:
    - Use bullet points.
    - Be direct. No filler phrases like "Based on the scores provided..."
    - Maximum 4-5 bullet points total.

    OUTPUT FORMAT:
    - **Overall Rating**: [Good/Average/Poor]
    - **Key Strengths**: (Short bullet points)
    - **Critical Risks**: (Short bullet points)
    - **Final Verdict**: (1 sentence summary)

    Context (Knowledge Base):
    {context_text}

    Customer Input:
    {query}

    Concise Evaluation:"""    
    prompt_time = time.time() - start_prompt

    # 3. LLM Generation Stage (Streaming)
    start_llm = time.time()
    full_response = ""
    for chunk in get_ollama_response_stream(prompt):
        full_response += chunk
        yield chunk
    
    llm_time = time.time() - start_llm
    total_time = time.time() - start_total
    
    # Update memory after stream matches
    memory.append(f"User: {query}")
    memory.append(f"Bot: {full_response}")
    
    # Log interaction to file (including timing)
    metrics = {
        "embed_time": embed_time,
        "search_time": search_time,
        "retrieval_time": retrieval_time,
        "prompt_time": prompt_time,
        "llm_time": llm_time,
        "total_time": total_time
    }
    
    log_interaction(user_id, query, full_response, metrics=metrics)



if __name__ == "__main__":
    print("RAG Module Loaded.")

