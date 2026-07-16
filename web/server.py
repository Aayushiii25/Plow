import os
import subprocess
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import json

app = FastAPI(title="InferSpore UI")

# Paths - assume server is run from project root, or specify these in env
CLI_PATH = os.environ.get("INFER_SPORE_CLI", "./bin/inferspore_cli")
WEIGHTS_DIR = os.environ.get("INFER_SPORE_WEIGHTS", "./weights")
TOKENIZER_PATH = os.environ.get("INFER_SPORE_TOKENIZER", "./tokenizer.flm")

# Mount static files for the frontend
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class GenerateRequest(BaseModel):
    prompt: str

async def generate_tokens(prompt: str):
    """
    Generator that spawns the C++ CLI and yields its output character-by-character
    or token-by-token using SSE (Server-Sent Events) format.
    """
    if not os.path.exists(CLI_PATH):
        # Fallback for dev mode when CLI isn't built/available
        yield f"data: {json.dumps({'token': 'CLI not found. '})}\n\n"
        yield f"data: {json.dumps({'token': 'Please build inferspore_cli first.'})}\n\n"
        return

    # Start the C++ CLI process
    process = await asyncio.create_subprocess_exec(
        CLI_PATH, WEIGHTS_DIR, TOKENIZER_PATH, prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Read stdout token by token as they arrive
    while True:
        # Read 1 byte at a time since tokens might not end in newlines
        char = await process.stdout.read(1)
        if not char:
            break
            
        try:
            # Basic decoding, for multi-byte UTF-8 this might slice bytes, 
            # but since C++ outputs tokens, we can read chunks or buffer until valid utf8.
            # For simplicity, we try to read small chunks if a byte is a start of a multibyte char.
            # But await process.stdout.read(1024) is safer to grab whatever is available.
            pass
        except:
            pass

    # A better approach: read whatever is available in the buffer
    while True:
        try:
            # We use read(4096) to get available bytes without blocking for EOF
            # If the process is slow, it might return fewer bytes.
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
                
            text = chunk.decode("utf-8", errors="replace")
            # We send the text chunk as SSE data
            yield f"data: {json.dumps({'token': text})}\n\n"
            
            # Yield control back to event loop
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error reading stdout: {e}")
            break

    # Wait for process to finish
    await process.wait()
    
    if process.returncode != 0:
        stderr = await process.stderr.read()
        err_text = stderr.decode("utf-8", errors="replace")
        yield f"data: {json.dumps({'error': f'Process exited with code {process.returncode}', 'stderr': err_text})}\n\n"

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(
        generate_tokens(request.prompt),
        media_type="text/event-stream"
    )

@app.get("/")
async def root():
    # Redirect root to static index
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
