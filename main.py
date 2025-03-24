from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from typing import List
import httpx
from uuid import uuid4

# PostgreSQL database connection
DATABASE_URL = "postgresql://postgres:Solution%4097@217.145.69.172:5432/smsdb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI app instance
app = FastAPI()

# ReportDetails model for saving responses
class ReportDetails(Base):
    __tablename__ = 'sms_app_reportdetails'

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Use timezone-aware datetime
    user_id = Column(Integer, ForeignKey('customuser.id'))  # Assuming the CustomUser table exists
    campaign_id = Column(String(12), nullable=False)
    report_id = Column(String(12), unique=True, nullable=False)
    status = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    msgCount = Column(Integer, nullable=False)
    errorCode = Column(Integer, nullable=False)
    messageId = Column(String(255), nullable=False)
    receiver = Column(String(20), nullable=False)  # New field for receiver

# SMS request body schema
class SMSRequest(BaseModel):
    sender: str
    receiver: List[str]
    msgType: str
    requestType: str
    content: str
    token: str
    campaign_id: str

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to send SMS and save response in the database
async def send_sms(receiver: str, sender: str, msgType: str, requestType: str, content: str, token: str, campaign_id: str, db: Session):
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
        
        response_data = response.json()
        # Save the response to the database
        report = ReportDetails(
            user_id=1,  # Replace with actual user ID
            campaign_id=campaign_id,
            report_id=str(uuid4())[:12],  # Generate a unique report_id
            status=response_data.get('status', 'UNKNOWN'),
            description=response_data.get('description', ''),
            msgCount=response_data.get('msgCount', 0),
            errorCode=response_data.get('errorCode', 0),
            messageId=response_data.get('messageId', ''),
            receiver=receiver  # Store the receiver's number
        )
        db.add(report)
        db.commit()

        return response_data

# Endpoint to handle SMS request and process it
@app.post("/send_sms")
async def handle_sms_request(sms_request: SMSRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    sender = sms_request.sender
    msgType = sms_request.msgType
    requestType = sms_request.requestType
    content = sms_request.content
    token = sms_request.token
    campaign_id = sms_request.campaign_id

    responses = []

    # Process bulk request without looping, send the whole list at once
    receiver_list = sms_request.receiver
    response = await send_sms(receiver_list, sender, msgType, requestType, content, token, campaign_id, db)
    responses.append(response)

    return {
        "status": "SMS sent",
        "total_receivers": len(receiver_list),
        "responses": responses
    }

@app.get("/")
def root():
    return {"message": "Send SMS API Successful"}
