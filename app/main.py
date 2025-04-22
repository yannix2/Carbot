from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api import router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT dynamically
    uvicorn.run("main:app", host="0.0.0.0", port=port)