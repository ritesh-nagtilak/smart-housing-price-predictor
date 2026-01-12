import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import logging

# Import schemas (validation models)
from schemas import PredictionInput, PredictionOutput, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.model_version = "1.0.0"
    
    def load_models(self) -> bool:
        try:

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, 'models', 'housing_model.pkl')
            scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
            
            print(f"DEBUG: Looking for model at: {model_path}")
            print(f"DEBUG: Looking for scaler at: {scaler_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.model_loaded = True
            logger.info(f"✓ Models loaded successfully. Model version: {self.model_version}")
            return True
        
        except Exception as e:
            logger.error(f"✗ Failed to load models: {str(e)}")
            self.model_loaded = False
            return False
    
    def predict(self, input_data: PredictionInput) -> dict:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Check startup logs.")
        
        try:

            features = np.array([[
                input_data.area,
                input_data.bedrooms,
                input_data.bathrooms,
                input_data.location_score
            ]])
            
            features_scaled = self.scaler.transform(features)
            
            predicted_price = self.model.predict(features_scaled)[0]
            
            confidence_interval = abs(predicted_price * 0.10)
            
            return {
                'predicted_price': float(predicted_price),
                'estimated_range_low': float(predicted_price - confidence_interval),
                'estimated_range_high': float(predicted_price + confidence_interval),
                'confidence_score': 0.92 
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting API server...")
    if not model_manager.load_models():
        logger.error("⚠️  Failed to load models at startup!")
    
    yield
    
    logger.info("🛑 Shutting down API server...")

app = FastAPI(
    title="Housing Price Predictor API",
    description="Production-ready ML API for real-time house price predictions",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Verify API is running and model is loaded"
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "degraded",
        model_loaded=model_manager.model_loaded,
        version=model_manager.model_version,
        timestamp=datetime.utcnow()
    )

@app.post(
    "/predict",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict House Price",
    description="Get real-time price prediction for a property"
)
async def predict_price(input_data: PredictionInput) -> PredictionOutput:
    if not model_manager.model_loaded:
        logger.error("Model not loaded when prediction requested")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable. Try again later."
        )
    
    try:
        prediction = model_manager.predict(input_data)
        logger.info(
            f"Prediction: {input_data.area}sqft, "
            f"{input_data.bedrooms}bd -> ${prediction['predicted_price']:,.0f}"
        )
        
        return PredictionOutput(
            predicted_price=prediction['predicted_price'],
            estimated_range_low=prediction['estimated_range_low'],
            estimated_range_high=prediction['estimated_range_high'],
            confidence_score=prediction['confidence_score'],
            model_version=model_manager.model_version,
            timestamp=datetime.utcnow(),
            input_features=input_data
        )
    
    except RuntimeError as e:
        logger.error(f"Runtime error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service error. Contact support."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred."
        )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Check server logs.",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        workers=4,     
        log_level="info"
    )
