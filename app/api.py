"""HTTP API for the speech service."""

from fastapi import FastAPI

app = FastAPI(title="Pronunciation Speech Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

