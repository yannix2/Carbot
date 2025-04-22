from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://carbot-7xh1.onrender.com"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


templates = Jinja2Templates(directory="templates")

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Carbot Application!"}