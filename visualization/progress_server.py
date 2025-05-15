import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

from inference import *

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/answer_progress")
async def get_answer_progress():
    """Serve real-time question answering progress."""
    try:
        with open(answer_progress_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "No progress data available"}
    
@app.get("/kg_progress")
async def get_progress():
    """Serve progress statistics."""
    try:
        with open(kg_progress_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return JSONResponse(content={"error": "No progress data available"}, status_code=404)

@app.get("/stats")
async def get_stats():
    """Serve detailed corpus processing stats."""
    try:
        with open("results/update_kg_logs.jsonl", "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        return JSONResponse(content={"error": "No logs available"}, status_code=404)

    return {"processed_corpora": len(logs), "logs": logs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="movie", help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    args = parser.parse_args()

    answer_progress_path = f"results/{args.model}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    kg_progress_path = f"results/update_{args.dataset}_kg_progress.json"

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7689)  # New port for QA tracking