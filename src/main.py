from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from helpers.config import get_settings ,Settings
from routes import data
app = FastAPI()

#model = pickle.load(open("house_price2","rb"))

@app.on_event("startup")
async def startup_db_client():
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client= app.mongo_conn[settings.MONGODB_DATABASE]

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongo_conn.close()

app.include_router(data.data_router)
