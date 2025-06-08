"""FastAPI application for Kelpie-Carbon v1."""
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
