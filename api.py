# it provides interface for interaction b/w server backened api and core files;
# Fast api required to provide interaction of these files with frontend

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from core.scorer import analyze_text
from fastapi.staticfiles import StaticFiles



limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Hallucination Detector API",
    description="Detects hallucinated claims in LLM-generated text using Wikipedia grounding, self-consistency, and embedding similarity.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
@limiter.limit("3/minute")
async def analyze(request: Request, input: TextInput):
    if len(input.text) > 500:
        raise HTTPException(status_code=400, detail="Input too long. Max 500 characters.")
    result = analyze_text(input.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}

# Mount the frontend directory to the root path
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)