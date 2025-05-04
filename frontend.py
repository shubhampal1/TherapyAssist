from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app for serving the React frontend
frontend_app = FastAPI()

# Mount the React frontend build directory
frontend_app.mount("/", StaticFiles(directory="posture-analysis-frontend/build", html=True), name="frontend")