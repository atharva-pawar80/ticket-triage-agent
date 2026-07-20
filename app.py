# app.py
import os
import json
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

# Load model once at startup
model = joblib.load('urgency_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class TicketRequest(BaseModel):
    text: str

def predict_urgency(text: str):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    return {"urgency": prediction, "confidence": round(float(confidence), 3)}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_urgency",
            "description": "Predicts the urgency of a customer support ticket. Returns 'urgency' (Low/Medium/High/Immediate) and 'confidence' (0 to 1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {"type": "string", "description": "The full text of the ticket"}
                },
                "required": ["ticket_text"]
            }
        }
    }
]

def run_agent(ticket_text: str):
    messages = [{
        "role": "user",
        "content": f"A new support ticket came in: '{ticket_text}'. Check its urgency and tell me what action we should take (auto-reply, escalate to human, or flag for review)."
    }]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024
    )
    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        result = predict_urgency(args["ticket_text"])

        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

        final = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            max_tokens=1024
        )
        return {"urgency_data": result, "agent_decision": final.choices[0].message.content}
    else:
        return {"urgency_data": None, "agent_decision": response_message.content}


@app.post("/predict")
def predict(request: TicketRequest):
    return predict_urgency(request.text)

@app.post("/agent")
def agent_endpoint(request: TicketRequest):
    return run_agent(request.text)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Ticket Triage Agent</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto;">
        <h2>Ticket Triage Agent</h2>
        <p>Paste a support ticket below and see how the agent handles it.</p>
        <textarea id="ticket" rows="4" style="width:100%;" placeholder="e.g. My payment failed twice, please fix this now"></textarea><br><br>
        <button onclick="submitTicket()">Check Ticket</button>
        <div id="result" style="margin-top:20px; white-space:pre-wrap;"></div>

        <script>
        async function submitTicket() {
            const text = document.getElementById('ticket').value;
            document.getElementById('result').innerText = "Thinking...";
            const res = await fetch('/agent', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            document.getElementById('result').innerText =
                "Model prediction: " + JSON.stringify(data.urgency_data) +
                "\\n\\nAgent decision:\\n" + data.agent_decision;
        }
        </script>
    </body>
    </html>
    """