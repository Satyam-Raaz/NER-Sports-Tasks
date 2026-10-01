import os

import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from uvicorn import run as app_run

from src.constant import *  # APP_HOST, PORT
from src.pipeline.train_pipeline import TrainPipeline

# Folder where the trained model is saved (change it, or set env var NER_MODEL_PATH)
NER_MODEL_PATH = "D:/Document/NLP_Project/NER/artifacts/ModelTrainerArtifacts/model-best"

app = FastAPI(title="Certificate NER API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- model loading
_nlp = None


def load_model(force: bool = False):
    """Load the spaCy model once; reload after training (force=True)."""
    global _nlp
    if _nlp is None or force:
        if not os.path.isdir(NER_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{NER_MODEL_PATH}'. Call /train first."
            )
        _nlp = spacy.load(NER_MODEL_PATH)
    return _nlp


def build_answer(grouped: dict) -> str:
    """Turn the extracted entities into a readable sentence."""
    sport = ", ".join(grouped.get("SPORTS_NAME", []))
    year = ", ".join(grouped.get("ORG_YEAR", []))
    position = ", ".join(grouped.get("WINNING_POSITION", []))

    if not (sport or year or position):
        return "No sport, year or winning position was found in this text."

    msg = (
        f"This certificate was awarded for securing {position} position"
        if position
        else "This certificate was awarded for participating"
    )
    if sport:
        msg += f" in {sport}"
    if year:
        msg += f" during the {year} competition"
    msg += "."

    missing = [
        name
        for name, val in [("sport", sport), ("year", year), ("position", position)]
        if not val
    ]
    if missing:
        msg += f" (Not found: {', '.join(missing)})"
    return msg


# ---------------------------------------------------------------- API
class PredictRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "ok", "endpoints": ["/train (GET)", "/predict (POST)"]}


@app.get("/train")
def training():
    # plain `def` -> FastAPI runs it in a thread, so the server is not blocked
    try:
        TrainPipeline().run_pipeline()
        try:
            load_model(force=True)  # use the freshly trained model
        except FileNotFoundError as e:
            return Response(f"Training finished, but {e}")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    try:
        nlp = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    doc = nlp(text)

    entities = [
        {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
        for e in doc.ents
    ]
    grouped: dict = {}
    for e in entities:
        grouped.setdefault(e["label"], []).append(e["text"])

    return {
        "text": text,
        "entities": entities,
        "grouped": grouped,
        "answer": build_answer(grouped),
    }


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=PORT)