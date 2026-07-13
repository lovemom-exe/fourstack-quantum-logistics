# ==========================================================================
# Author: Hoang Anh Quan
# Version: 0.1.0
# Purpose: create system framework, configure the setting, register the routes
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
# Routes
from api.router import router

# API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==========================================================================
# PARAMETERS
# ==========================================================================

# My main app
app = FastAPI(
    title="quamtum logistics optimization",
    version="0.1.0",
    description="This is a backend for quantum logistics optimization",
)

# Middleware: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routes
app.include_router(router)
