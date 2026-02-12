"""
FastAPI Main Application for Sentiment Analysis
This application provides REST API endpoints for sentiment analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from sentiment_analyzer import SentimentAnalyzer

# Initialize FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment in text using NLP techniques",
    version="1.0.0"
)

# Configure CORS to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

# Get the directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")


class TextInput(BaseModel):
    """
    Pydantic model for text input validation
    """
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's absolutely amazing and exceeded my expectations."
            }
        }


class SentimentResponse(BaseModel):
    """
    Pydantic model for sentiment analysis response
    """
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def serve_frontend():
    """
    Serve the main frontend HTML page

    Returns:
        FileResponse: The index.html file
    """
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify API is running

    Returns:
        dict: Status message
    """
    return {
        "status": "healthy",
        "message": "Sentiment Analysis API is running"
    }


@app.post("/api/analyze", response_model=SentimentResponse)
async def analyze_text(input_data: TextInput):
    """
    Analyze sentiment of provided text

    Args:
        input_data (TextInput): Text input from user

    Returns:
        SentimentResponse: Analysis results including sentiment and scores
    """
    try:
        # Validate input
        if not input_data.text or len(input_data.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Perform sentiment analysis
        result = analyzer.analyze_combined(input_data.text)

        return SentimentResponse(
            success=True,
            data=result
        )

    except Exception as e:
        return SentimentResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze/file", response_model=SentimentResponse)
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze sentiment from uploaded text file

    Args:
        file (UploadFile): Uploaded text file

    Returns:
        SentimentResponse: Analysis results including sentiment and scores
    """
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(
                status_code=400,
                detail="Only .txt files are supported"
            )

        # Read file content
        content = await file.read()
        text = content.decode('utf-8')

        # Validate content
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # Check file size (limit to 1MB)
        if len(text) > 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 1MB limit"
            )

        # Perform sentiment analysis
        result = analyzer.analyze_combined(text)

        # Add filename to result
        result['filename'] = file.filename

        return SentimentResponse(
            success=True,
            data=result
        )

    except UnicodeDecodeError:
        return SentimentResponse(
            success=False,
            error="Unable to decode file. Please ensure it's a valid UTF-8 text file"
        )
    except Exception as e:
        return SentimentResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze/batch", response_model=SentimentResponse)
async def analyze_batch(texts: list[str]):
    """
    Analyze sentiment for multiple texts in batch

    Args:
        texts (list): List of text strings

    Returns:
        SentimentResponse: Analysis results for all texts
    """
    try:
        # Validate input
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="Text list cannot be empty")

        # Limit batch size
        if len(texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch size limited to 100 texts"
            )

        # Perform batch analysis
        results = analyzer.analyze_batch(texts)

        return SentimentResponse(
            success=True,
            data={
                'count': len(results),
                'results': results
            }
        )

    except Exception as e:
        return SentimentResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/docs")
async def api_documentation():
    """
    Provide API documentation and usage examples

    Returns:
        dict: API documentation
    """
    return {
        "api_name": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/api/health": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/api/analyze": {
                "method": "POST",
                "description": "Analyze sentiment of text",
                "input": {
                    "text": "string"
                }
            },
            "/api/analyze/file": {
                "method": "POST",
                "description": "Analyze sentiment from uploaded .txt file",
                "input": "multipart/form-data file"
            },
            "/api/analyze/batch": {
                "method": "POST",
                "description": "Analyze multiple texts in batch",
                "input": {
                    "texts": ["string1", "string2"]
                }
            }
        }
    }


if __name__ == "__main__":
    # Run the application
    # Host on 0.0.0.0 to allow external access (useful for OSHA Cloud Lab)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )
