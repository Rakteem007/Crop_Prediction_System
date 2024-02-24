from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the models
with open('model_crop.pkl', 'rb') as file:
    model_crop = pickle.load(file)

with open('model_yield.pkl', 'rb') as file:
    model_yield = pickle.load(file)

def predict_with_missing_data(input_data, top_n=2):
    sample_input_crop = pd.DataFrame([input_data])
    # Predict the probability of each crop
    crop_probabilities = model_crop.predict_proba(sample_input_crop)[0]
    top_crop_indices = np.argsort(crop_probabilities)[::-1][:top_n]
    top_crops = np.array(model_crop.classes_)[top_crop_indices]
    
    top_crops_with_yield = []
    for crop in top_crops:
        # Adjust input data to include the predicted 'CropName' for yield prediction
        input_data_with_crop = input_data.copy()
        input_data_with_crop['CropName'] = crop
        sample_input_yield = pd.DataFrame([input_data_with_crop])
        
        yield_prediction = model_yield.predict(sample_input_yield)[0]
        top_crops_with_yield.append((crop, yield_prediction))
    
    return top_crops_with_yield

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()

    # List of fields that should be treated as numerical
    numerical_fields = ['N', 'P', 'K', 'pH', 'Humidity', 'Temperature', 'Rainfall']
    
    # Convert numerical fields to float, leave categorical fields as is
    for key in input_data.keys():
        if key in numerical_fields and input_data[key]:
            input_data[key] = float(input_data[key])

    top_crops_with_yield = predict_with_missing_data(input_data, top_n=2)
    return render_template('index.html', prediction_text=top_crops_with_yield)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
