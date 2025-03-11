House Price Prediction
Overview
This project is a House Price Prediction system that utilizes a machine learning model to estimate house prices based on the number of rooms and the area size. It consists of a Flask API backend, a Gradio-based frontend, and an additional web-based frontend using HTML and JavaScript.

Gradio_frontend
<img width="414" alt="image" src="https://github.com/user-attachments/assets/d884e7e4-72db-4826-8a35-205d5c869617" />


Flask API backend “ backendapi.py”
<img width="415" alt="image" src="https://github.com/user-attachments/assets/b9cc335b-f440-42c3-9001-98a1df7dac0e" />

backendapi.py
Code:
from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
model = load("house_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "House Price Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict_price():
    data = request.json  
    try:
        num_rooms = int(data["num_rooms"])
        area = float(data["area"])
        features = np.array([[area, num_rooms]])
        prediction = model.predict(features)[0][0] * 1_000_000
        return jsonify({"predicted_price": f"{prediction:,.0f} Rwf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


Features
Flask API: to handle predictions
Gradio interface: for interactive predictions
HTML & JavaScript frontend: for a user-friendly experience
Machine Learning Model: loaded via Joblib for predictions
CORS enabled: for frontend-backend communication

 Technologies Used
 Backend:
- Flask (API development)
- Flask-CORS (Cross-Origin Resource Sharing)
- Joblib (Model loading)
- NumPy (Data processing)

 Frontend:
- Gradio (Interactive UI)
- HTML & JavaScript (Web-based UI)
- CSS (Styling)

 Project Structure

 home.html: Web-based frontend
 backendapi.py: Flask API backend
 connect.py: Gradio-based frontend fetching from API
 index.py: Another Gradio-based interface
 house_model.pkl: Pre-trained ML model


Installation & Setup
Prerequisites
Python 3.x installed
Flask and required dependencies installed

 Install Dependencies
pip install flask flask-cors joblib numpy gradio requests

 Running the Flask Backend
python backendapi.py

The API will start at `http://127.0.0.1:5000/`.


 Running the Web Frontend
- Open “home.html” in a browser
- Enter the number of rooms and area size.
- Click "Predict" to fetch the house price from the API.
  
<img width="416" alt="image" src="https://github.com/user-attachments/assets/7b8d2220-47ec-4a75-9f6a-afd6706fe055" />


 Running the Gradio Interface
python backendapi.py
OR
python index.py

 API Endpoint
POST: url = "http://127.0.0.1:5000/predict"

   Request Body (JSON):
   json
    {
      "num_rooms": 3,
      "area": 120
    }
    
   Response (JSON):
   json
    {
      "predicted_price": "12,000,000 Rwf"
    }
  
