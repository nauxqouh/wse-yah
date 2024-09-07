from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from rag_llm import RAGPipelineSetup

app = FastAPI()

# Cấu hình FastAPI với các thông số giống như trong ứng dụng Streamlit của bạn
EMBEDDINGS_MODEL_NAME = "BAAI/bge-m3"
QDRANT_URL = "https://88bb6378-66e7-49db-a5de-6bb17f0d664a.europe-west3-0.gcp.cloud.qdrant.io:6333"
NGROK_STATIC_DOMAIN = "WSE"
NGROK_TOKEN = "2knpVVzzj8s7zHdWN2HJa5CMKTm_3MR4C9ZFWL1uTX6Lr4zEe"
HUGGINGFACE_API_KEY = "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"
QDRANT_API_KEY = "cLGVHbp48h0CZayJIXdxVW-JJijODOKpBFlzIPm6nvLHxRE4B_nrFA"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
GROQ_API_KEY= "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"
DATABASE_TO_COLLECTION = {
    "Trường Đại học Khoa học Tự nhiên": "US_vectorDB",
    "Trường Đại học Công nghệ Thông tin": "UIT_vectorDB",
    "Trường Đại Học Khoa Học Xã Hội - Nhân Văn": "USSH_vectorDB",
    "Trường Đại Học Bách Khoa": "UT_vectorDB",
    "Trường Đại Học Quốc Tế": "IU_vectorDB",
    "Trường Đại Học Kinh tế - Luật": "UEL_vectorDB"
}

class PromptRequest(BaseModel):
    prompt: str
    database: str

@app.post("/api/query")
async def query(prompt_request: PromptRequest):
    selected_collection = DATABASE_TO_COLLECTION.get(prompt_request.database, "US_vectorDB")
    rag_setup = RAGPipelineSetup(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_collection_name=selected_collection,
        huggingface_api_key=HUGGINGFACE_API_KEY,
        embeddings_model_name=EMBEDDINGS_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
    )
    rag_pipeline = rag_setup.rag(source=selected_collection)
    response = rag_pipeline.run(prompt_request.prompt)
    return {"response": response}


@app.head("/api/query")
async def query_head(request: Request):
    # Respond with a 200 OK status for HEAD requests
    return {"status": "HEAD request successful"}

if __name__ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)
