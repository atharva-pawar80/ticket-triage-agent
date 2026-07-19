# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# load once at startup, not per-request — this matters for speed
model = joblib.load('urgency_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_urgency(request: TicketRequest):
    X = vectorizer.transform([request.text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)

    return {
        "urgency": prediction,
        "confidence": round(float(confidence), 3)
    }