import os
import csv
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

# Create a generic HF pipeline (Faster)
# pipe = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=128,
#     truncation=True,
#     do_sample=False,
#     num_beams=1
# )

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,      
    truncation=True,
    do_sample=True,          
    temperature=0.7,         
    top_p=0.9,
    num_beams=1
)


llm = HuggingFacePipeline(pipeline=pipe)

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
        relevant_docs = db.similarity_search(query, k=1)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
    else:
        context_text = "No knowledge base documents found."
    
    print(f"\n Context is : {context_text}")

    
    # Get specific user memory
    memory = get_user_memory(user_id)
    history_text = "\n".join(memory)

    print(f"\n History is : {history_text}")
    
    # Construct a prompt for RAG
    # We improve the prompt to ask for complete sentences and consider history
    prompt = f"""You are a helpful credit scoring assistant. 
                Use the Context below to answer the Question in clear and complete sentences.
    

        Context:
        {context_text}

        History:
        {history_text}

        Question: {query}
        Answer:""" 
    

    # Generate answer
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

