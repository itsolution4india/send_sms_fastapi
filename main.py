# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
import httpx

app = FastAPI()

class SMSRequest(BaseModel):
    sender: str
    receiver: List[str]
    msgType: str
    requestType: str
    content: str

async def send_sms(receiver: str, sender: str, msgType: str, requestType: str, content: str, token: str):
    url = 'https://api.mobireach.com.bd/sms/send'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "sender": sender,
        "receiver": [receiver],
        "msgType": msgType,
        "requestType": requestType,
        "content": content,
        "contentType": 1 if msgType == "T" else 2
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Error: {response.text}")

@app.post("/send_sms")
async def handle_sms_request(sms_request: SMSRequest, background_tasks: BackgroundTasks):
    sender = sms_request.sender
    msgType = sms_request.msgType
    requestType = sms_request.requestType
    content = sms_request.content
    token = 'your_token_here'  # Get it dynamically in a real-world scenario

    # For bulk requests, process each number in the background
    for receiver in sms_request.receiver:
        background_tasks.add_task(send_sms, receiver, sender, msgType, requestType, content, token)

    return {"status": "SMS is being sent in background", "total_receivers": len(sms_request.receiver)}
