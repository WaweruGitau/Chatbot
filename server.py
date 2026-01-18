from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag_chatbot import ask_credit_bot, documents
import uvicorn

app = FastAPI(title="Credit Scoring RAG API")

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "active", "message": "Credit Scoring Chatbot API is running"}

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    """
    Endpoint to interact with the RAG chatbot.
    """
    if not documents:
        raise HTTPException(status_code=500, detail="No documents loaded in the knowledge base.")
        
    try:
        # The ask_credit_bot function returns the answer string
        # We now pass specific user_id to maintain separate memories
        answer = ask_credit_bot(request.query, user_id=request.user_id)
        return QueryResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API server with hot reload enabled
    print("Starting API Server with Hot Reload...")
    uvicorn.run("server:app", host="0.0.0.0", port=8088, reload=True, reload_delay=5.0)
