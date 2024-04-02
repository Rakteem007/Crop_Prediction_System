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

def predict_with_adjusted_yield(input_data, top_n=2):
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
    
    # Ensure the yield of the second best fit is not more than the best fit
    if top_n > 1 and top_crops_with_yield[1][1] > top_crops_with_yield[0][1]:
#         print("Adjusting the yield of the second best fit crop to not exceed the best fit crop's yield.")
        # Apply a scaling factor to adjust the yield of the second best fit crop
        # This could be based on the difference in probabilities, or simply a fixed percentage less than the best fit
        adjustment_factor = 0.9  # For demonstration, reduce by 10%
        adjusted_yield = top_crops_with_yield[0][1] * adjustment_factor
        top_crops_with_yield[1] = (top_crops_with_yield[1][0], adjusted_yield)
    
    return top_crops_with_yield

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    
    # Edit the range here bhai -->
    # ? CHECK OUT limit-finder.additional.py TO GENERATE THIS :)
    numerical_field_ranges = {  
        'N' : (15.0,  160.0),
        'P' : (6.0,  100.0),
        'K' : (12.0,  130.0),
        'pH' : (3.8,  9.6),
        'Humidity' : (32.0,  135.0),
        'Temperature' : (4.0,  100.0),
        'Rainfall' : (450.0,  4150.0)
    }

    # List of fields that should be treated as numerical
    numerical_fields = ['N', 'P', 'K', 'pH', 'Humidity', 'Temperature', 'Rainfall']
    
    # Convert numerical fields to float, leave categorical fields as is
    for key in input_data.keys():
        if key in numerical_fields and input_data[key]:
            input_data[key] = float(input_data[key])
            allowed_range = numerical_field_ranges.get(key)
            if not allowed_range or not allowed_range[0] <= input_data[key] <= allowed_range[1]:
                errorMsg = f"There is a abnormality in {key} data field. Please Check it. Our registered range is {allowed_range[0]} to {allowed_range[1]}"
                return render_template('index.html', prediction_text=errorMsg, valid=False)
            

    top_crops_with_yield = predict_with_adjusted_yield(input_data, top_n=2)
    return render_template('index.html', prediction_text=top_crops_with_yield, valid=True)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
