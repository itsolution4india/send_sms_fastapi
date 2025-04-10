from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request, APIRouter, status, Header
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime, timezone, timedelta
from typing import List
import httpx
from uuid import uuid4
import logging
import json
import os
from logging.handlers import RotatingFileHandler
import time
from utils import generate_token, generate_message_id
from typing import Dict
from sqlalchemy import select
import requests

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

class SenderID(Base):
    __tablename__ = 'sms_app_senderid'

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Auto-filled datetime
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    sender_id = Column(String(255), nullable=False)
    token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    token_updated_date = Column(DateTime, nullable=False)
    
    users = relationship("CustomUser", back_populates="sender_id")

class CustomUser(Base):
    __tablename__ = 'sms_app_customuser'

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    email = Column(String(255), unique=True, nullable=False)
    phone_number = Column(String(13), nullable=False)
    sender_id_id = Column(Integer, ForeignKey('sms_app_senderid.id'), nullable=True)  # Assuming 'sms_app_senderid' is the table for SenderID
    sender_id = relationship("SenderID", back_populates="users")
    failed_login_attempts = Column(Integer, default=0)
    last_failed_attempt = Column(DateTime, nullable=True)
    locked_until = Column(DateTime, nullable=True)
    
    api_credentials = relationship("ApiCredentials", back_populates="user")
    sms_api_responses = relationship("SendSmsApiResponse", back_populates="user")
    accounts = relationship("Account", back_populates="user")
    
    
class ReportDetails(Base):
    __tablename__ = 'sms_app_reportdetails'

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Use timezone-aware datetime
    user_id = Column(Integer, ForeignKey('sms_app_customuser.id'))  # Assuming the CustomUser table exists
    campaign_id = Column(String(12), nullable=False)
    report_id = Column(String(12), unique=True, nullable=False)
    status = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    msgCount = Column(Integer, nullable=False)
    errorCode = Column(Integer, nullable=False)
    messageId = Column(String(255), nullable=False)
    receiver = Column(JSON, nullable=False) 

class ApiCredentials(Base):
    __tablename__ = 'sms_app_apicredentials'
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey('sms_app_customuser.id'))
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    token_updated_date = Column(DateTime, nullable=False)
    
    # Relationship
    user = relationship("CustomUser", back_populates="api_credentials")
    
class SendSmsApiResponse(Base):
    __tablename__ = 'sms_app_sendsmsapiresponse'
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey('sms_app_customuser.id'))
    status = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    content_type = Column(Integer, nullable=False)
    errorCode = Column(Integer, nullable=False)
    actual_msgCount = Column(Integer, nullable=False)
    actual_messageId = Column(String(255), nullable=False)
    actual_current_balance = Column(Integer, nullable=False)
    user_msgCount = Column(Integer, nullable=False)
    user_messageId = Column(String(255), nullable=False)
    user_current_balance = Column(Integer, nullable=False)
    receiver = Column(JSON, nullable=False)
    content = Column(Text, nullable=False)
    msg_type = Column(String(20), nullable=False)
    
    # Relationship
    user = relationship("CustomUser", back_populates="sms_api_responses")
    
class Account(Base):
    __tablename__ = 'sms_app_account'
    
    id = Column(Integer, primary_key=True, index=True)
    account_number = Column(String(16), nullable=False)
    account_holder_name = Column(String(255), nullable=False)
    account_id = Column(String(255), unique=True, nullable=False)
    gui_balance = Column(Numeric(12, 4), nullable=False)
    api_balance = Column(Numeric(12, 4), nullable=False)
    user_id = Column(Integer, ForeignKey('sms_app_customuser.id'))
    
    # Relationship
    user = relationship("CustomUser", back_populates="accounts")


CustomUser.api_credentials = relationship("ApiCredentials", back_populates="user")
CustomUser.sms_api_responses = relationship("SendSmsApiResponse", back_populates="user")
CustomUser.accounts = relationship("Account", back_populates="user")

# Pydantic model for login request
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    refresh_token: str
    
class ErrorResponse(BaseModel):
    error: str
    message: str

# Pydantic model for token response
class TokenRefreshResponse(BaseModel):
    token: str
    refresh_token: str
    
# Pydantic model for SMS send request
class SmsSendRequest(BaseModel):
    sender: str = Field(..., description="Sender ID/CLI")
    receiver: List[str] = Field(..., description="List of receiver phone numbers")
    contentType: int = Field(..., description="1=Regular, 2=Unicode")
    content: str = Field(..., description="Message content")
    msgType: str = Field(..., description="T=Transactional, P=Promotional")
    requestType: str = Field(..., description="S=Single, B=Bulk")

class MessageStatusRequest(BaseModel):
    sender: str
    messageId: str
    receiver: str

# Pydantic model for SMS send response
class SmsSendResponse(BaseModel):
    status: str
    description: str
    msgCost: str
    currentBalance: str
    contentType: int
    msgCount: int
    errorCode: int
    messageId: int
    
# SMS request body schema
class SMSRequest(BaseModel):
    sender: str
    receiver: List[str]
    msgType: str
    requestType: str
    content: str
    token: str
    campaign_id: str
    user_id: int

auth_router = APIRouter(prefix="/auth", tags=["authentication"])

sms_router = APIRouter(prefix="/sms", tags=["sms"])
        
# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@auth_router.post("/token", response_model=TokenResponse)
async def generate_api_token(
    login_request: LoginRequest, 
    db: Session = Depends(get_db)
):
    try:
        # Find API credentials
        query = select(ApiCredentials).where(
            ApiCredentials.username == login_request.username
        )
        api_credential = db.execute(query).scalar_one_or_none()
        
        # Validate credentials
        if not api_credential:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid username or password"
            )
        
        # Password validation (replace with secure method)
        if api_credential.password != login_request.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid username or password"
            )
        
        # Generate new tokens
        new_token = generate_token()
        new_refresh_token = generate_token()
        
        # Update database
        api_credential.token = new_token
        api_credential.refresh_token = new_refresh_token
        api_credential.token_updated_date = datetime.now(timezone.utc)
        
        db.commit()
        
        return {
            "token": new_token,
            "refresh_token": new_refresh_token
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Token generation failed: {str(e)}"
        )

@auth_router.post("/token/refresh", 
    responses={
        401: {"model": ErrorResponse}
    }
)
async def refresh_token(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        # Validate Authorization header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing refresh token"
            )
        
        # Extract refresh token
        refresh_token = authorization.split(" ")[1]
        
        # Current timestamp
        current_time = datetime.now(timezone.utc)
        
        # Find API credentials with matching refresh token
        query = select(ApiCredentials).where(
            ApiCredentials.refresh_token == refresh_token
        )
        api_credential = db.execute(query).scalar_one_or_none()
        
        # Validate refresh token exists
        if not api_credential:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check token age
        token_age = current_time - api_credential.token_updated_date
        if token_age > timedelta(hours=1):
            # Return custom error response for token expiration
            return {
                "error": "Unauthorized",
                "message": "Token has expired"
            }
        
        # Generate new token pair
        new_token = generate_token()
        new_refresh_token = generate_token()
        
        # Update database record
        api_credential.token = new_token
        api_credential.refresh_token = new_refresh_token
        api_credential.token_updated_date = current_time
        
        # Commit changes
        db.commit()
        
        # Return new token pair
        return {
            "token": new_token,
            "refresh_token": new_refresh_token
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Rollback and handle unexpected errors
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )

@sms_router.post("/send")
async def send_sms_api(
    sms_request: SmsSendRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    # 1. Validate Authorization Header
    if not authorization or not authorization.startswith("Bearer "):
        return {
            "error": "Unauthorized",
            "message": "Invalid or missing refresh token",
            "errorCode": 401
        }
    
    refresh_token = authorization.split(" ")[1]
    
    # 2. Validate Refresh Token
    query_api_cred = select(ApiCredentials).where(
        ApiCredentials.refresh_token == refresh_token
    )
    api_credential = db.execute(query_api_cred).scalar_one_or_none()
    
    if not api_credential:
        return {
            "error": "Unauthorized",
            "message": "Invalid refresh token",
            "errorCode": 401
        }
    
    # 3. Get User and Account
    user = api_credential.user
    
    # 4. Check Account Balance
    account = db.query(Account).filter(Account.user_id == user.id).first()
    
    if not account or account.api_balance < len(sms_request.receiver):
        return {
            "error": "Balance error",
            "message": "Insufficient Balance ",
            "errorCode": 1506
        }
    
    # 5. Validate Sender
    sender_query = select(SenderID).where(
        SenderID.id == user.sender_id_id    
    )
    sender = db.execute(sender_query).scalar_one_or_none()
    
    if not sender:
        return {
            "error": "Unauthorized",
            "message": "Invalid Sender ID",
            "errorCode": 401
        }
        
        
    # 5.1 Check token expiration and refresh if needed
    current_time = datetime.now(timezone.utc)
    time_difference = current_time - sender.token_updated_date
    
    # If token is older than 45 minutes, refresh it
    if time_difference.total_seconds() > 45 * 60:
        try:
            # Try refreshing token first
            refresh_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sender.refresh_token}"
            }
            
            refresh_response = requests.post(
                "https://api.mobireach.com.bd/auth/token/refresh",
                headers=refresh_headers
            )
            
            if refresh_response.status_code == 200:
                refresh_data = refresh_response.json()
                sender.token = refresh_data.get("token")
                sender.refresh_token = refresh_data.get("refresh_token")
                sender.token_updated_date = current_time
                db.commit()
            else:
                # If refresh token fails, try login with credentials
                login_payload = {
                    "username": "opway",
                    "password": "Dhaka@5599"
                }
                
                login_headers = {
                    "Content-Type": "application/json"
                }
                
                login_response = requests.post(
                    "https://api.mobireach.com.bd/auth/tokens",
                    json=login_payload,
                    headers=login_headers
                )
                
                if login_response.status_code == 200:
                    login_data = login_response.json()
                    sender.token = login_data.get("token")
                    sender.refresh_token = login_data.get("refresh_token")
                    sender.token_updated_date = current_time
                    db.commit()
                else:
                    # If both refresh and login fail
                    return {
                        "error": "Unauthorized",
                        "message": "Invalid refresh token",
                        "errorCode": 401
                    }
        except Exception as e:
            logging.error(f"Error refreshing token: {str(e)}")
            return {
                "error": "Unauthorized",
                "message": "Error refreshing token",
                "errorCode": 401
            }
            
    # Ensure sms_payload is initialized before use
    sms_payload = None
    try:
        # 6. Prepare SMS Send Request
        sms_payload = {
            "sender": sms_request.sender,
            "receiver": sms_request.receiver,
            "contentType": sms_request.contentType,
            "content": sms_request.content,
            "msgType": sms_request.msgType,
            "requestType": sms_request.requestType
        }
        
        # 7. Send SMS via External API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {sender.token}"
        }
        
        external_response = requests.post(
            "https://api.mobireach.com.bd/sms/send", 
            json=sms_payload, 
            headers=headers
        )
        
        # 8. Parse External API Response
        if external_response.status_code != 200:
            logging.error(f"SMS API Call Failed. Status Code: {external_response.status_code}, Response: {external_response.json()}")
            return {
                "error": "Unauthorized",
                "message": external_response.json(),
                "errorCode": 401
            }
        
        external_data = external_response.json()
        
        # 9. Update Account Balance
        account.api_balance -= len(sms_request.receiver)
        
        # 10. Create SMS API Response
        sms_api_response = SendSmsApiResponse(
            user_id=user.id,
            status=external_data.get('status', 'UNKNOWN'),
            description=external_data.get('description', ''),
            content_type=sms_request.contentType,
            errorCode=external_data.get('errorCode', 0),
            actual_msgCount=float(external_data.get('msgCost', 0)),
            actual_messageId=str(external_data.get('messageId', '')),
            actual_current_balance=float(external_data.get('currentBalance', 0)),
            user_msgCount=len(sms_request.receiver),
            user_messageId=str(generate_message_id()),
            user_current_balance=float(account.api_balance),
            receiver=sms_request.receiver,
            content=sms_request.content,
            msg_type=sms_request.msgType,
        )
        
        # 11. Commit Database Changes
        db.add(sms_api_response)
        db.commit()
        
        # 12. Return Response
        return {
            "status": "SUCCESS",
            "description": "Message sent",
            "msgCost": str(sms_api_response.actual_msgCount),
            "currentBalance": str(sms_api_response.user_current_balance),
            "contentType": sms_request.contentType,
            "msgCount": len(sms_request.receiver),
            "errorCode": sms_api_response.errorCode,
            "messageId": sms_api_response.user_messageId
        }
    
    except Exception as e:
        # Log the specific error
        logging.error(f"Error in send_sms_api: {str(e)}")
        
        db.rollback()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred"
        )

@app.get("/sms/status")
def get_message_status(
    sender: str, 
    messageId: str, 
    receiver: str, 
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    # Extract the Bearer token
    token = authorization.split("Bearer ")[1]

    # Step 1: Validate sender from SenderID table
    sender_obj = db.query(SenderID).filter(SenderID.sender_id == sender).first()
    if not sender_obj:
        raise HTTPException(status_code=404, detail="Invalid sender ID")

    # Step 2: Get CustomUser linked to this SenderID
    custom_user = db.query(CustomUser).filter(CustomUser.sender_id_id == sender_obj.id).first()
    if not custom_user:
        raise HTTPException(status_code=404, detail="No user linked with this sender ID")

    # Step 3: Validate token from ApiCredentials table
    api_credentials = db.query(ApiCredentials).filter(ApiCredentials.user_id == custom_user.id).first()
    if not api_credentials:
        raise HTTPException(status_code=403, detail="Failed to fetch API credentials")

    # Validate request token with the token in ApiCredentials
    if api_credentials.token != token:
        raise HTTPException(status_code=403, detail="Invalid token")

    actual_token = sender_obj.token
    
    # Step 4: Check if messageId exists in SendSmsApiResponse table
    sms_response = db.query(SendSmsApiResponse).filter(SendSmsApiResponse.user_messageId == messageId).first()
    if not sms_response:
        raise HTTPException(status_code=404, detail="MessageId not found")

    # Step 5: Call external API
    api_url = "https://api.mobireach.com.bd/sms/status"
    headers = {
        "Authorization": f"Bearer {actual_token}"
    }
    params = {
        "sender": sender,
        "messageId": sms_response.actual_messageId,
        "receiver": receiver
    }

    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to get message status from external API")

    # Step 6: Modify response
    api_response = response.json()

    # Remove 'duringMsgBalance' and set 'msgCost' to 1
    if "duringMsgBalance" in api_response:
        del api_response["duringMsgBalance"]
    api_response["msgCost"] = "1"
    api_response["messageId"] = messageId

    return api_response

@app.get("/account/balance")
def check_account_balance(
    username: str, 
    authorization: str = Header(...),  # Extract token from Authorization header
    db: Session = Depends(get_db)
):
    # Extract the Bearer token
    try:
        token = authorization.split("Bearer ")[1]
    except IndexError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    # Step 1: Validate API Credentials
    api_credentials = db.query(ApiCredentials).filter(ApiCredentials.username == username).first()
    if not api_credentials:
        raise HTTPException(status_code=404, detail="Username not found")

    # Step 2: Validate Token
    if api_credentials.token != token:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Step 3: Find associated Account
    account = db.query(Account).filter(Account.user_id == api_credentials.user_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="No account found for this user")

    # Step 4: Prepare Response
    response = {
        "status": "SUCCESS",
        "guiBalance": str(account.gui_balance),
        "apiBalance": str(account.api_balance)
    }

    return response

# Function to send SMS and save response in the database
async def send_sms(receivers: list, sender: str, msgType: str, requestType: str, content: str, token: str, campaign_id: str, user_id: int,total_receivers:int, db: Session):
    url = 'https://api.mobireach.com.bd/sms/send'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Create the request payload
    data = {
        "sender": sender,
        "receiver": receivers,
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
                    msgCount=total_receivers,
                    errorCode=response_data.get('errorCode', 0),
                    messageId=response_data.get('messageId', ''),
                    receiver=receivers  # Store the receiver's number
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
    total_receivers = len(sms_request.receiver)
    logger.info(f"Received SMS request for campaign {sms_request.campaign_id} with {total_receivers} {sms_request.receiver} recipients")
    
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
        response = await send_sms(receiver_list, sender, msgType, requestType, content, token, campaign_id,user_id,total_receivers, db)
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

app.include_router(auth_router)
app.include_router(sms_router)

if __name__ == "__main__":
    import uvicorn
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info("Starting SMS API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)