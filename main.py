# main.py
import asyncio
import httpx
import json
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import csv
import io
from contextlib import asynccontextmanager

# Database models (using SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./sms_api.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SMS Log model
class SMSLog(Base):
    __tablename__ = "sms_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=func.now())
    sender = Column(String)
    receiver = Column(Text)  # JSON string of receivers
    content = Column(Text)
    msg_type = Column(String)  # T or P
    request_type = Column(String)  # S or B
    content_type = Column(Integer)  # 1 or 2
    status = Column(String)  # PENDING, SENT, FAILED
    response_data = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Limiter for concurrent tasks
class TaskLimiter:
    def __init__(self, max_concurrent_tasks=50):
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
    async def run(self, func, *args, **kwargs):
        async with self.semaphore:
            return await func(*args, **kwargs)

task_limiter = TaskLimiter(50)  # Limit to 50 concurrent tasks

# Pydantic models
class SMSReceiver(BaseModel):
    phone_number: str

class SMSRequest(BaseModel):
    sender: str
    receivers: List[str]
    content: str
    msg_type: str = Field(..., description="T for transactional, P for promotional")
    request_type: str = Field(..., description="S for single, B for bulk")
    content_type: Optional[int] = Field(1, description="1 for regular, 2 for unicode")
    token: str
    
    @validator('msg_type')
    def validate_msg_type(cls, v):
        if v not in ['T', 'P']:
            raise ValueError('msg_type must be either T or P')
        return v
    
    @validator('request_type')
    def validate_request_type(cls, v):
        if v not in ['S', 'B']:
            raise ValueError('request_type must be either S or B')
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v not in [1, 2]:
            raise ValueError('content_type must be either 1 or 2')
        return v

class BulkSMSUploadRequest(BaseModel):
    sender: str
    content: str
    msg_type: str = Field(..., description="T for transactional, P for promotional")
    content_type: Optional[int] = Field(1, description="1 for regular, 2 for unicode")
    token: str

class SMSResponse(BaseModel):
    message: str
    task_id: Optional[int] = None
    failed: Optional[List[str]] = None
    succeeded: Optional[int] = None

class SMSStatusResponse(BaseModel):
    id: int
    status: str
    created_at: str
    response_data: Optional[str] = None

# FastAPI app
app = FastAPI(
    title="SMS Sender API",
    description="API for sending SMS messages in bulk with high throughput",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task processing
async def send_sms_task(sms_log_id: int, token: str, sender: str, receivers: List[str], 
                        content: str, msg_type: str, request_type: str, content_type: int):
    # Get the DB session
    session = SessionLocal()
    
    try:
        # Fetch SMS log
        sms_log = session.query(SMSLog).filter(SMSLog.id == sms_log_id).first()
        if not sms_log:
            return
        
        # API endpoint
        url = 'https://api.mobireach.com.bd/sms/send'
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        # Prepare payload
        payload = {
            'sender': sender,
            'receiver': receivers,
            'content': content,
            'msgType': msg_type,
            'requestType': request_type,
            'contentType': content_type
        }
        
        # Send the request asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
        
        # Update the SMS log
        sms_log.response_data = response.text
        if response.status_code == 200:
            sms_log.status = 'SENT'
        else:
            sms_log.status = 'FAILED'
        
        session.commit()
    except Exception as e:
        # Log the error
        try:
            sms_log = session.query(SMSLog).filter(SMSLog.id == sms_log_id).first()
            if sms_log:
                sms_log.status = 'FAILED'
                sms_log.response_data = str(e)
                session.commit()
        except:
            pass
    finally:
        session.close()

# Send SMS in batches
async def process_batch(token: str, sender: str, receivers: List[str], content: str, 
                       msg_type: str, request_type: str, content_type: int, db):
    tasks = []
    batch_size = 50  # Process 50 messages at a time
    
    # Create log entries for each batch
    log_ids = []
    failed_receivers = []
    
    for i in range(0, len(receivers), batch_size):
        batch_receivers = receivers[i:i + batch_size]
        
        # Create SMS log entry
        sms_log = SMSLog(
            sender=sender,
            receiver=json.dumps(batch_receivers),
            content=content,
            msg_type=msg_type,
            request_type=request_type,
            content_type=content_type,
            status='PENDING'
        )
        db.add(sms_log)
        db.commit()
        db.refresh(sms_log)
        log_ids.append(sms_log.id)
        
        # Create task for this batch
        task = asyncio.create_task(
            task_limiter.run(
                send_sms_task,
                sms_log.id, token, sender, batch_receivers, content, msg_type, request_type, content_type
            )
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    return log_ids, failed_receivers

# API endpoints
@app.post("/send-sms", response_model=SMSResponse, 
         summary="Send SMS in background",
         description="Send SMS messages in the background. This endpoint returns immediately and processes the SMS sending asynchronously.")
async def send_sms(request: SMSRequest, background_tasks: BackgroundTasks, db=Depends(get_db)):
    # Handle promotional messages with proper content type
    if request.msg_type == 'P' and request.content_type != 2:
        request.content_type = 2  # Force Unicode for promotional messages
    
    # Create SMS log entry
    sms_log = SMSLog(
        sender=request.sender,
        receiver=json.dumps(request.receivers),
        content=request.content,
        msg_type=request.msg_type,
        request_type=request.request_type,
        content_type=request.content_type,
        status='PENDING'
    )
    db.add(sms_log)
    db.commit()
    db.refresh(sms_log)
    
    # Add task to background
    background_tasks.add_task(
        send_sms_task,
        sms_log.id, request.token, request.sender, request.receivers, 
        request.content, request.msg_type, request.request_type, request.content_type
    )
    
    return {"message": "SMS sending initiated", "task_id": sms_log.id}

@app.post("/send-sms-sync", response_model=SMSResponse,
         summary="Send SMS synchronously", 
         description="Send SMS messages synchronously. This endpoint waits for the SMS to be sent before returning.")
async def send_sms_sync(request: SMSRequest, db=Depends(get_db)):
    try:
        # Handle promotional messages with proper content type
        if request.msg_type == 'P' and request.content_type != 2:
            request.content_type = 2  # Force Unicode for promotional messages
            
        # API endpoint
        url = 'https://api.mobireach.com.bd/sms/send'
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {request.token}'
        }
        
        # Prepare payload
        payload = {
            'sender': request.sender,
            'receiver': request.receivers,
            'content': request.content,
            'msgType': request.msg_type,
            'requestType': request.request_type,
            'contentType': request.content_type
        }
        
        # Create log entry
        sms_log = SMSLog(
            sender=request.sender,
            receiver=json.dumps(request.receivers),
            content=request.content,
            msg_type=request.msg_type,
            request_type=request.request_type,
            content_type=request.content_type,
            status='PENDING'
        )
        db.add(sms_log)
        db.commit()
        db.refresh(sms_log)
        
        # Send the request
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
        
        # Update the SMS log
        sms_log.response_data = response.text
        if response.status_code == 200:
            sms_log.status = 'SENT'
            db.commit()
            return {"message": "SMS sent successfully", "task_id": sms_log.id}
        else:
            sms_log.status = 'FAILED'
            db.commit()
            return {"message": f"Failed to send SMS: {response.text}", "task_id": sms_log.id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-bulk-sms", response_model=SMSResponse,
         summary="Send bulk SMS messages",
         description="Send bulk SMS messages efficiently. This endpoint processes messages in batches of 50.")
async def send_bulk_sms(request: SMSRequest, db=Depends(get_db)):
    try:
        # Handle promotional messages with proper content type
        if request.msg_type == 'P' and request.content_type != 2:
            request.content_type = 2  # Force Unicode for promotional messages
            
        # Process in batches
        log_ids, failed_receivers = await process_batch(
            request.token, request.sender, request.receivers, request.content,
            request.msg_type, "B", request.content_type, db
        )
        
        return {
            "message": f"Processing {len(request.receivers)} messages in batches",
            "task_id": log_ids[0] if log_ids else None,
            "succeeded": len(request.receivers) - len(failed_receivers),
            "failed": failed_receivers if failed_receivers else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv", response_model=SMSResponse,
         summary="Upload CSV file and send SMS",
         description="Upload a CSV file with phone numbers and send SMS to those numbers.")
async def upload_csv_and_send(
    sender: str = Form(...),
    content: str = Form(...),
    msg_type: str = Form(...),
    content_type: int = Form(1),
    token: str = Form(...),
    file: UploadFile = File(...),
    db=Depends(get_db)
):
    try:
        # Read CSV file
        contents = await file.read()
        decoded = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded))
        
        # Extract phone numbers
        receivers = []
        for row in csv_reader:
            if 'Phone_numbers' in row:
                phone = row['Phone_numbers'].strip()
                if phone:
                    receivers.append(phone)
        
        if not receivers:
            raise HTTPException(status_code=400, detail="No valid phone numbers found in CSV")
        
        # Handle promotional messages with proper content type
        if msg_type == 'P' and content_type != 2:
            content_type = 2  # Force Unicode for promotional messages
            
        # Process in batches
        log_ids, failed_receivers = await process_batch(
            token, sender, receivers, content,
            msg_type, "B", content_type, db
        )
        
        return {
            "message": f"Processing {len(receivers)} messages from CSV in batches",
            "task_id": log_ids[0] if log_ids else None,
            "succeeded": len(receivers) - len(failed_receivers),
            "failed": failed_receivers if failed_receivers else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sms-status/{sms_id}", response_model=SMSStatusResponse,
        summary="Check SMS status",
        description="Check the status of a sent SMS by its ID.")
async def check_sms_status(sms_id: int, db=Depends(get_db)):
    sms_log = db.query(SMSLog).filter(SMSLog.id == sms_id).first()
    if not sms_log:
        raise HTTPException(status_code=404, detail="SMS not found")
    
    return {
        "id": sms_log.id,
        "status": sms_log.status,
        "created_at": sms_log.created_at.isoformat(),
        "response_data": sms_log.response_data
    }

@app.get("/sms-logs", response_model=List[SMSStatusResponse],
        summary="Get SMS logs",
        description="Get a list of SMS logs with optional limit parameter.")
async def get_sms_logs(limit: int = 100, db=Depends(get_db)):
    logs = db.query(SMSLog).order_by(SMSLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "status": log.status,
            "created_at": log.created_at.isoformat(),
            "response_data": log.response_data
        }
        for log in logs
    ]

@app.get("/health", 
        summary="Health check",
        description="Check if the API is running")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)