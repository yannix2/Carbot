import re
import os
from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Query, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import ollama
import requests
from langchain.vectorstores import Chroma
from langdetect import detect
import logging
from pymongo import MongoClient
from bson import ObjectId
from passlib.context import CryptContext
import jwt
from io import BytesIO
from datetime import datetime, timedelta
from fastapi import UploadFile, File
from langchain.document_loaders import TextLoader, PyPDFLoader
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import BackgroundTasks
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import smtplib
from email.message import EmailMessage
import uuid
import httpx
from mailjet_rest import Client
from langchain.embeddings import OllamaEmbeddings
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import mailjet_rest
import openai
import json
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI2")

print(f"MONGO_URI: {MONGO_URI}")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
current_llm = os.getenv("CURRENT_LLM", "mistral:latest")
JWT_EXPIRATION_DELTA = timedelta(hours=1)  # Token expiration time is now 1 hour
client = MongoClient(MONGO_URI, maxPoolSize=10)
db = client["Carbot"]
conversation_collection = db["conversation_history"]
feedback_collection = db["feedback_history"]
user_collection = db["users"]  # Collection for storing user data
api_key = 'bc3364428a653020a27e3981859f653b'
api_secret = 'c97fad45a9c9bddca4b0fcfce2edbcb5'
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1" 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" 
persist_directory = "MyVectorDB2.0"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
# FastAPI app and router setup
app = FastAPI()
router = APIRouter()
# Set up the absolute path to the profile_pics folder

logging.basicConfig(level=logging.INFO)


# Password encryption context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Helper functions
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def get_db():
    return db

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + JWT_EXPIRATION_DELTA
    to_encode.update({"exp": expire})
    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logging.error(f"Error encoding JWT: {e}")
        raise HTTPException(status_code=500, detail="Error creating access token")

def get_user_from_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# Document processing (PDF)
documents = [
    {"path": "docs/406458776-8-Pieces-de-rechange-classification-pdf.pdf", "title": "Pièces de rechange - Classification"},
    {"path": "docs/CHERGUI,Affaf.pdf", "title": "Gestion des stocks de pièces de rechange"},
      {"path": "docs/ridex-amortisseur-de-direction.pdf", "title": ""},
       {"path": "docs/ridex-appareil-de-commande,-verrouillage-central.pdf", "title": ""},
        {"path": "docs/ridex-arbre-de-transmission.pdf", "title": ""},
         {"path": "docs/ridex-arbre-a-came.pdf", "title": ""},
          {"path": "docs/ridex-assortiment-de-raccords.pdf", "title": ""},
           {"path": "docs/ridex-bague.pdf", "title": ""},
            {"path": "docs/ridex-biellette-de-barre-stabilisatrice.pdf", "title": ""},
             {"path": "docs/ridex-bobines-d'allumage.pdf", "title": ""},
              {"path": "docs/ridex-carter-d'huile.pdf", "title": ""},
               {"path": "docs/ridex-courroie.pdf", "title": ""},
                {"path": "docs/ridex-courroie2.pdf", "title": ""},
                 {"path": "docs/ridex-filtre-air-de-l'habitacle.pdf", "title": ""},
]
def process_documents():
    all_docs = []
    total_docs = len(documents)

    for i, doc in enumerate(documents):
        pdf_path = Path(doc["path"])
        if not pdf_path.exists():
            print(f"⚠️ {pdf_path} not found, skipping.")
            continue

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page_num, page in enumerate(pages):
            cleaned_text = clean_text(page.page_content)
            all_docs.append(Document(page_content=cleaned_text, metadata={"source": pdf_path.name, "title": doc["title"], "page": page_num + 1}))

        print(f"Processed {i+1}/{total_docs} documents.")

    print("Document processing complete.")
    persist_directory = "MyVectorDB2.0"

    if Path(persist_directory).exists():
        vectorstore = Chroma.load(persist_directory)
    else:
        vectorstore = Chroma.from_documents(all_docs, embedding_function, persist_directory=persist_directory)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vectorstore = Chroma.from_documents(all_docs, embedding_function, persist_directory=persist_directory)

# Background task to process documents
@app.on_event("startup")
async def on_startup(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_documents)

prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=""" 
    Tu es un assistant spécialisé en pièces de rechange automobile.
    Réponds aux questions de manière précise en utilisant les documents suivants :
    
    {context}
    
    🔹 **Historique de la conversation** :  
    {chat_history}
    
    🔹 **Utilisateur** : {question}  
    🔹 **Assistant** :
    """
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")

# Language detection function
def detect_language(text: str):
    return detect(text)
def chat_with_groq(question: str, context: str = "") -> str:
    prompt = f"""
    Réponds à la question suivante en utilisant le contexte fourni si nécessaire.

    CONTEXTE:
    {context}

    QUESTION:
    {question}
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Use 'llama-3.3-70b-versatile' as the model
    payload = {
        "model": "llama-3.3-70b-versatile",  # Updated model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,  # Optional, adjust as needed
        "max_tokens": 4096  # You can adjust this according to the model's limits
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Will raise HTTPError for bad status codes

        if response.status_code == 200:
            response_json = response.json()

            # Extracting the message from the response
            if 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json['choices'][0]['message']['content']
            else:
                logging.error(f"Unexpected response format: {response_json}")
                return "An unexpected error occurred."
        else:
            logging.error(f"Unexpected status code: {response.status_code}")
            logging.error(f"Response body: {response.text}")
            return "An error occurred while processing your request."

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
        logging.error(f"Response body: {response.text}")
        return "An error occurred while processing your request."
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return "An error occurred while processing your request."
class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    email: str

class QueryRequest(BaseModel):
    question: str

PROFILE_PICS_DIR = "app/static/profile_pics"  
os.makedirs(PROFILE_PICS_DIR, exist_ok=True)
# Initialize Mailjet client with your API keys


def send_verification_email(email: str, verification_token: str):
    mailjet = Client(auth=(api_key, api_secret), version='v3.1')

    # Modify the verification link to point to the front-end page
    verification_link = f"https://carbot-7xh1.onrender.com/verify-account?token={verification_token}"

    # HTML content for the email
    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f4;">
            <div style="text-align: center; padding-bottom: 20px;">
                <!-- Logo -->
                <img src="../logoh.png" alt="" style="max-width: 200px;"/>
            </div>
            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); width: 60%; margin: 0 auto;">
                <h3 style="color: #333;">Click this link to verify your email:</h3>
                <p style="font-size: 16px; color: #555;">
                    <a href="{verification_link}" style="color: #007bff; text-decoration: none; font-weight: bold;">
                        {verification_link}
                    </a>
                </p>
                <p style="font-size: 14px; color: #777;">If you did not request this, please ignore this email.</p>
            </div>
            <footer style="text-align: center; padding-top: 20px; font-size: 14px; color: #888;">
                <p>Thank you for using Carbot!</p>
            </footer>
        </body>
    </html>
    """

    # Email data
    email_data = {
        'Messages': [
            {
                'From': {
                    'Email': 'club.tunivisonstekup@gmail.com',
                    'Name': 'Carbot'
                },
                'To': [
                    {
                        'Email': email,
                        'Name': 'Recipient'
                    }
                ],
                'Subject': 'Please verify your email',
                'TextPart': f'Click this link to verify your email: {verification_link}',
                'HTMLPart': html_content,
            }
        ]
    }

    # Send the email
    try:
        result = mailjet.send.create(data=email_data)
        if result.status_code == 200:
            print(f"Verification email sent to {email}")
        else:
            raise HTTPException(status_code=500, detail="Failed to send verification email")
    except Exception as e:
        print(f"Failed to send verification email: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send verification email")
@router.post("/register")
async def register(
    background_tasks: BackgroundTasks,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    # Check if username already exists
    if user_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already exists")

    # Check if email already exists
    if user_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Create the user document without profile picture
    user_doc = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "verified": False,                # Email not verified yet
        "verification_token": str(uuid.uuid4())  # Generate a unique verification token
    }
    
    # Insert the new user into the database
    result = user_collection.insert_one(user_doc)
    
    # Add background task to send verification email
    background_tasks.add_task(send_verification_email, email, user_doc["verification_token"])
    
    return {
        "message": "User registered successfully. Please check your email to verify your account.",
        "user_id": str(result.inserted_id)
    }

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        # Try to find the user by username OR email
        user_in_db = user_collection.find_one({
            "$or": [
                {"username": form_data.username},
                {"email": form_data.username}
            ]
        })

        # If user doesn't exist or password doesn't match
        if not user_in_db or not verify_password(form_data.password, user_in_db["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Generate the access token
        access_token = create_access_token(data={"sub": user_in_db["username"]})

        # Since no profile picture is being handled, just return a default URL or None
        profile_pic_url = "/static/default-profile-pic.png"  # Default image URL or None if no image is used
        
        # Return the response with the access token and profile picture URL
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user_in_db["username"],
            "email": user_in_db["email"],
            "profile_pic": profile_pic_url  # Default profile picture URL or None
        }

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chatPsy")
async def chat(request: QueryRequest, token: str = Depends(oauth2_scheme), db: MongoClient = Depends(get_db)):
    logging.info("Received /chatPsy request")

    try:
        user_id = get_user_from_token(token)
        logging.info(f"User ID from token: {user_id}")

        question = request.question
        logging.info(f"User question: {question}")

        # Context retrieval
        context_docs = vectorstore.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in context_docs])
        logging.info("Retrieved context from vectorstore.")

        # Call Groq LLM
        response = chat_with_groq(question=question, context=context)
        logging.info(f"LLM Response: {response}")

        # Save to MongoDB
        conversation_data = {
            "user_message": question,
            "bot_message": response,
            "user_id": user_id
        }
        conversation_collection.insert_one(conversation_data)

        return jsonable_encoder({"response": response})

    except Exception as e:
        logging.exception("Erreur dans /chatPsy")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {type(e).__name__} - {str(e)}")
# Endpoint for switching LLM based on user preferences
user_llm_preferences = {}

# Switch LLM model endpoint with user preferences (flexibility)
@router.post("/switch_llm")
async def switch_llm(user_id: str, model: str = Query("mistral:latest", enum=["mistral:latest", "llama3:latest"])):
    user_llm_preferences[user_id] = model
    return {"message": f"Switched to {model} for user {user_id}"}



@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily to a buffer
        file_contents = await file.read()
        file_extension = file.filename.split('.')[-1].lower()

        # Ensure file is a PDF
        if file_extension != 'pdf':
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create a temporary file and write the contents of the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
            tmp_pdf_file.write(file_contents)
            tmp_pdf_file_path = tmp_pdf_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_pdf_file_path)
        pages = loader.load()

        # Text Splitter to break the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Process and extract content from PDF
        documents = []
        for page_num, page in enumerate(pages):
            cleaned_text = clean_text(page.page_content)  # Ensure text is cleaned
            page_documents = text_splitter.split_text(cleaned_text)
            for doc in page_documents:
                documents.append({
                    "text": doc,
                    "metadata": {"source": file.filename, "page": page_num + 1}
                })

 
        doc_objs = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in documents]
        vectorstore.add_documents(doc_objs)

        vectorstore.persist()

        return {"message": "File processed and added to vector store successfully", "documents": documents}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

def get_user_from_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")  # assuming 'sub' is where username is stored
        if not username:
            raise HTTPException(status_code=403, detail="Could not validate credentials")
        return username
    except jwt.jwtError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
@router.put("/update-password")
async def update_password(
    change_password_request: ChangePasswordRequest,     
    token: str = Depends(oauth2_scheme)
):
    username = get_user_from_token(token)
    user_in_db = user_collection.find_one({"username": username})
    if not user_in_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(change_password_request.old_password, user_in_db["password"]):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    
    new_hashed_password = hash_password(change_password_request.new_password)
    result = user_collection.update_one(
        {"username": username},
        {"$set": {"password": new_hashed_password}}
    )
    
    if result.modified_count == 1:
        return {"detail": "Password updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update password")
    


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

def create_reset_token(email: str):
    expiration = datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
    payload = {
        "sub": email,
        "exp": expiration
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def send_email(to_email: str, reset_link: str):
    # Initialize the Mailjet client
    mailjet = mailjet_rest.Client(auth=(api_key, api_secret), version='v3.1')
    
    # Create the email data
    data = {
        'Messages': [
            {
                'From': {
                     'Email': 'club.tunivisonstekup@gmail.com',
                    'Name': 'Carbot'
                },
                'To': [
                    {
                        'Email': to_email
                    }
                ],
                'Subject': 'Password Reset Request',
                'TextPart': f'Click the link to reset your password: {reset_link}',
                'HTMLPart': f'<h3>Password Reset Request</h3><p>Click <a href="{reset_link}">here</a> to reset your password.</p>'
            }
        ]
    }
    
    try:
        # Send the email via Mailjet
        result = mailjet.send.create(data=data)
        if result.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to send email via Mailjet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    # Check if the email exists in the database
    user = user_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")
    
    # Create the reset token
    reset_token = create_reset_token(request.email)
    reset_link = f"https://carbot-7xh1.onrender.com/reset-password/{reset_token}"  
    send_email(request.email, reset_link)
    
    return {"message": "If the email exists, a reset link has been sent to your email address."}

class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str

# Verify reset token function (you could also add expiration date check here)
def verify_reset_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Reset token has expired.")
    except jwt.JWTError:
        raise HTTPException(status_code=400, detail="Invalid reset token.")

# Reset password endpoint
@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    payload = verify_reset_token(request.reset_token)

    user_in_db = user_collection.find_one({"email": payload["sub"]})

    
    if not user_in_db:
        raise HTTPException(status_code=404, detail="User not found")

    # Hash the new password
    hashed_password = pwd_context.hash(request.new_password)

    # Update password in the database
    result = user_collection.update_one(
        {"email": user_in_db["email"]},
        {"$set": {"password": hashed_password}}
    )

    if result.modified_count == 1:
        return {"detail": "Password reset successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reset password")
mailjet = Client(auth=(api_key, api_secret), version="v3.1")
@router.post("/mailing")
async def mailing(
    email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
):
    # Prepare the email data
    email_data = {
        'Messages': [
            {
               'From': {
                     'Email':email,
                    'Name': 'CLIENTS CARBOT'
                },
                "To": [
                    {
                        "Email": 'club.tunivisonstekup@gmail.com',
                    }
                ],
                "Subject": subject,
                "TextPart": message,
                "HTMLPart": f"<p>{message}</p>",
            }
        ]
    }

    try:
        # Send the email via Mailjet API
        result = mailjet.send.create(data=email_data)
        if result.status_code == 200:
            return {"detail": "Email sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")
    
def decode_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return payload  # Decoded information, which could contain the user's details
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid token.")
logger = logging.getLogger("uvicorn.error")

@router.post("/verify-account")
async def verify_account(token: str = Query(...)):
    try:
        # Log the token to see what we're receiving
        # Find the user by token
        user = user_collection.find_one({"verification_token": token})
        if not user:
            raise HTTPException(status_code=400, detail="Invalid token")
        
        # Mark the user as verified
        user_collection.update_one({"_id": user["_id"]}, {"$set": {"verified": True}})
        return {"detail": "Account verified successfully."}
    
    except Exception as e:
        # Log the exception to the backend logs for easier debugging
        logger.error(f"Error verifying account: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify account")