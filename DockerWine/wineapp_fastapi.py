from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model
with open(f"/Users/mj_peace/Desktop/MyMLOPS/DockerWine/wine_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

app = FastAPI()

# Input schema
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine FastAPI app!"}

@app.post("/predict")
def predict(data: WineInput):
    X = np.array([[data.fixed_acidity, data.volatile_acidity, data.citric_acid, data.residual_sugar,
                   data.chlorides, data.free_sulfur_dioxide, data.total_sulfur_dioxide, data.density,
                   data.pH, data.sulphates, data.alcohol]])
    prediction = model.predict(X)[0]
    return {"predicted_class": int(prediction)}