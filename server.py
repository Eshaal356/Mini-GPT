import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

# Add parent dir to path to import mingpt
sys.path.append(os.path.dirname(__file__))

from mingpt.model import GPT
from mingpt.utils import set_seed, CfgNode as CN

app = FastAPI()

# Serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model configuration (should match training)
def get_model_config():
    C = CN()
    C.model_type = 'gpt-mini'
    C.n_layer = None
    C.n_head = None
    C.n_embd = None
    C.vocab_size = 65 
    C.block_size = 128
    C.embd_pdrop = 0.1
    C.resid_pdrop = 0.1
    C.attn_pdrop = 0.1
    return C

# Global model and dataset objects
model = None
stoi = {}
itos = {}

def load_context():
    global stoi, itos, model
    data_path = 'input.txt'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    text = open(data_path, 'r').read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    config = get_model_config()
    config.vocab_size = len(chars)
    model = GPT(config)
    
    ckpt_path = os.path.join('out', 'chargpt', 'model.pt')
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    else:
        print(f"Checkpoint not found at {ckpt_path}. Model will be untrained.")
    model.eval()
    return True

# Chat memory (Simple in-memory store for session)
sessions: Dict[str, List[str]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    global model, stoi, itos
    
    if model is None:
        if not load_context():
            raise HTTPException(status_code=500, detail="Model or data not found. Please wait for training to save a checkpoint.")

    # Memory logic: Build context from session
    if request.session_id not in sessions:
        sessions[request.session_id] = []
    
    # In a char-level model, we might just append the user message. 
    # For a chat interface, we might want a separator or just raw text.
    # We'll treat it as continuing the text.
    context_str = request.message
    
    # Construct prompt. For Shakespeare, maybe just the last few chars?
    # Or let's just use the message as the seed.
    prompt = context_str
    
    # Simple sampling
    x = torch.tensor([stoi.get(s, stoi.get(' ', 0)) for s in prompt], dtype=torch.long)[None,...]
    
    # Generate
    y = model.generate(x, 200, temperature=0.8, do_sample=True, top_k=10)[0]
    
    response = ""
    for i in y[len(prompt):]:
        char = itos[int(i)]
        response += char
    
    clean_response = response
    
    return {"reply": clean_response}

@app.get("/")
async def root():
    return {"message": "MiniGPT Server Running. Visit /static/index.html"}

if __name__ == "__main__":
    import uvicorn
    load_context()
    uvicorn.run(app, host="0.0.0.0", port=8000)
