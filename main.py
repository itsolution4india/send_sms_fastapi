from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from typing import List
import httpx
from uuid import uuid4
import logging
import json
import os
from logging.handlers import RotatingFileHandler
import time

# Set up logging
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logger
logger = logging.getLogger("sms_api")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# File handler with rotation (10MB per file, max 5 files)
file_handler = RotatingFileHandler("logs/sms_api.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# PostgreSQL database connection
DATABASE_URL = "postgresql://postgres:Solution%4097@217.145.69.172:5432/smsdb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI app instance
app = FastAPI()

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    request_path = request.url.path
    request_method = request.method
    
    # Log request start
    logger.info(f"Request started - ID: {request_id} | {request_method} {request_path}")
    
    # Time the request
    start_time = time.time()
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log request completion
        logger.info(
            f"Request completed - ID: {request_id} | {request_method} {request_path} | "
            f"Status: {response.status_code} | Time: {process_time:.4f}s"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed - ID: {request_id} | {request_method} {request_path} | "
            f"Error: {str(e)} | Time: {process_time:.4f}s"
        )
        raise

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
async def send_sms(receiver: str, sender: str, msgType: str, requestType: str, content: str, token: str, campaign_id: str, user_id: int, db: Session):
    url = 'https://api.mobireach.com.bd/sms/send'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Create the request payload
    data = {
        "sender": sender,
        "receiver": [receiver],
        "msgType": msgType,
        "requestType": requestType,
        "content": content,
        "contentType": 1 if msgType == "T" else 2
    }
    
    # Log the outgoing request (masking sensitive data)
    masked_data = data.copy()
    # Mask token in logs
    masked_headers = headers.copy()
    masked_headers["Authorization"] = "Bearer [MASKED]"
    
    logger.info(f"Sending SMS request to {url} for campaign {campaign_id}")
    logger.debug(f"SMS request headers: {json.dumps(masked_headers)}")
    logger.debug(f"SMS request payload: {json.dumps(masked_data)}")
    
    # Send the request
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
            
            # Log the response
            logger.info(f"Received response from SMS API for campaign {campaign_id}: Status {response.status_code}")
            
            if response.status_code != 200:
                error_detail = f"SMS API Error: Status {response.status_code}, Response: {response.text}"
                logger.error(error_detail)
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            response_data = response.json()
            logger.debug(f"SMS API response data: {json.dumps(response_data)}")
            
            # Save the response to the database
            try:
                report = ReportDetails(
                    user_id=user_id,  # Replace with actual user ID
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
                logger.info(f"SMS response saved to database for campaign {campaign_id}, report_id: {report.report_id}")
            except Exception as db_error:
                logger.error(f"Database error while saving SMS response: {str(db_error)}")
                db.rollback()
                raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
            
            return response_data
    except httpx.RequestError as req_error:
        error_msg = f"HTTP Request error while sending SMS: {str(req_error)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Endpoint to handle SMS request and process it
@app.post("/send_sms")
async def handle_sms_request(sms_request: SMSRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    logger.info(f"Received SMS request for campaign {sms_request.campaign_id} with {len(sms_request.receiver)} {sms_request.receiver} recipients")
    
    sender = sms_request.sender
    msgType = sms_request.msgType
    requestType = sms_request.requestType
    content = sms_request.content
    token = sms_request.token
    campaign_id = sms_request.campaign_id
    user_id = sms_request.user_id

    responses = []

    try:
        # Process bulk request without looping, send the whole list at once
        receiver_list = sms_request.receiver
        response = await send_sms(receiver_list[0], sender, msgType, requestType, content, token, campaign_id,user_id, db)
        responses.append(response)
        
        logger.info(f"Successfully processed SMS request for campaign {campaign_id}")
        
        return {
            "status": "SMS sent",
            "total_receivers": len(receiver_list),
            "responses": responses
        }
    except Exception as e:
        logger.error(f"Error processing SMS request for campaign {campaign_id}: {str(e)}")
        raise

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Send SMS API Successful"}

# Add endpoint to check logs (for admin use)
@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

# Exception handler for internal server errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"detail": "Internal server error", "error_type": type(exc).__name__}

if __name__ == "__main__":
    import uvicorn
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info("Starting SMS API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)