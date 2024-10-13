from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI()

model = joblib.load('my_trained_mode.pkl')  # Ensure the correct model file is used
scaler = joblib.load('scaler.pkl')  # Ensure the correct scaler file is used

# Define the input schema using Pydantic
class CarFeatures(BaseModel):
    Mileage: float
    EngineV: float
    Brand: str
    Body: str
    Engine_Type: str
    Registration: str

model_columns = ['Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault', 
                 'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 
                 'Body_van', 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        input_data = pd.DataFrame([[features.Mileage, features.EngineV, features.Brand, features.Body, 
                                    features.Engine_Type, features.Registration]],
                                  columns=['Mileage', 'EngineV', 'Brand', 'Body', 'Engine Type', 'Registration'])
        
        input_data = pd.get_dummies(input_data, drop_first=True)
        
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        
        input_data_scaled = scaler.transform(input_data)
        
        log_price_prediction = model.predict(input_data_scaled)

        price_prediction = np.exp(log_price_prediction)[0]
        
        return {"predicted_price": round(price_prediction, 2)}
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}

