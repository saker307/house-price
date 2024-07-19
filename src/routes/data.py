from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import pandas as pd
import pickle
import motor.motor_asyncio
from bson import ObjectId
from helpers.config import get_settings, Settings

# Initialize the API Router
data_router = APIRouter()

# Load the model and feature names
with open('house_price2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Define the input model
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft_living: int
    sqft_lot: int
    floors: int
    view: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    zipcode: int
    lat: float
    long: float
    waterfront_N: int
    waterfront_Y: int
    condition_Average: int
    condition_Fair: int
    condition_Good: int
    condition_Poor: int
    condition_VeryGood: int

def preprocess_input(features: HouseFeatures) -> pd.DataFrame:
    # Convert the input features to a DataFrame
    input_dict = features.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Convert categorical variables to dummy variables
    input_df = pd.get_dummies(input_df)
    
    # Reindex to match the model's feature set
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    return input_df

@data_router.get("/welcome")
async def welcome():
    return {"message": "Welcome to the API service"}

@data_router.post("/predict")
async def predict(features: HouseFeatures):
    try:
        input_data = preprocess_input(features)
        prediction = model.predict(input_data)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

