import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the API!"}

# Define endpoint for making predictions
@app.post('/predict')
def predict(data:dict):
  # Load model from .pkl file
  with open('./model.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Apply the same feature engineering steps used in training the model
    df = df.astype(np.number)
    df['const'] = 1
    # Reorder the columns to match the order used in training the model
    df = df[['age', 'bmi', 'children', 'const']]
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}