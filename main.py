from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frontend import frontend_app
from posture_analysis_live import live_analysis_route

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes from other modules
app.include_router(live_analysis_route, prefix="/live-analysis")


# Mount the frontend app
app.mount("/", frontend_app)