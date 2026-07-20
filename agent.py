import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq

print("Script started")

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_urgency",
            "description": "Predicts the urgency of a customer support ticket based on its text. Returns 'urgency' (Low/Medium/High/Immediate) and 'confidence' (0 to 1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {
                        "type": "string",
                        "description": "The full text of the customer's complaint or support ticket"
                    }
                },
                "required": ["ticket_text"]
            }
        }
    }
]

def call_urgency_api(ticket_text):
    """This actually calls YOUR FastAPI server"""
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": ticket_text}
    )
    return response.json()

def run_agent(ticket_text):
    messages = [
        {
            "role": "user",
            "content": f"A new support ticket came in: '{ticket_text}'. Check its urgency and tell me what action we should take (auto-reply, escalate to human, or flag for review)."
        }
    ]

    response = client.chat.completions.create(
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
        ticket_text_arg = args["ticket_text"]

        # Actually call our API
        result = call_urgency_api(ticket_text_arg)
        print(f"[Model called] urgency={result['urgency']}, confidence={result['confidence']}")

        # Send the result back so the model can reason about it
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            max_tokens=1024
        )

        return final_response.choices[0].message.content
    else:
        return response_message.content


if __name__ == "__main__":
    print("Entering main block")
    try:
        ticket = "My payment failed twice and I need this fixed immediately"
        result = run_agent(ticket)
        print("\n--- Agent's Final Decision ---")
        print(result)
    except Exception as e:
        print("ERROR OCCURRED:", e)