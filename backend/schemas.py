from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class PredictionInput(BaseModel):

    area: float = Field(
        ...,
        gt=0,  # Greater than 0
        le=10000,  # Less than or equal to 10000 sq ft
        description="Property area in square feet"
    )
    bedrooms: int = Field(
        ...,
        ge=1,  # Greater than or equal to 1
        le=10,  # Less than or equal to 10
        description="Number of bedrooms"
    )
    bathrooms: float = Field(
        ...,
        ge=1,  # At least 1
        le=6,
        description="Number of bathrooms"
    )
    location_score: float = Field(
        ...,
        ge=1,  # Scale from 1-10
        le=10,
        description="Location score (1=poor, 10=excellent)"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "area": 2500,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "location_score": 7.5
            }
        }
    
    @validator('area')
    def validate_area(cls, v):
        """Validate area is practical for residential properties."""
        if v < 300:
            raise ValueError('Area too small - minimum 300 sq ft for residential')
        return round(v, 2)
    
    @validator('bathrooms')
    def validate_bathrooms(cls, v):
        """Validate bathrooms makes sense."""
        if v < 1:
            raise ValueError('At least 1 bathroom required')
        return round(v, 1)


class PredictionOutput(BaseModel):
    """
    Output schema for prediction response.
    
    Always includes confidence indicators and metadata for transparency.
    """
    
    predicted_price: float = Field(
        ...,
        description="Predicted house price in USD"
    )
    estimated_range_low: float = Field(
        ...,
        description="95% confidence interval lower bound"
    )
    estimated_range_high: float = Field(
        ...,
        description="95% confidence interval upper bound"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model confidence (0-1)"
    )
    model_version: str = Field(
        ...,
        description="Version of model used for prediction"
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of prediction"
    )
    input_features: PredictionInput = Field(
        ...,
        description="Echo of input features for audit trail"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "predicted_price": 625000.00,
                "estimated_range_low": 575000.00,
                "estimated_range_high": 675000.00,
                "confidence_score": 0.92,
                "model_version": "1.0.0",
                "timestamp": "2026-01-01T17:13:00Z",
                "input_features": {
                    "area": 2500,
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "location_score": 7.5
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Is model loaded and ready")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Server timestamp")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "timestamp": "2026-01-01T17:13:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    timestamp: datetime = Field(..., description="Error timestamp")
