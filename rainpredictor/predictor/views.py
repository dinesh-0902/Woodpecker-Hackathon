from rest_framework.decorators import api_view
from django.http import JsonResponse
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rainfall_predictor.h5')

# Load the trained model
model = load_model(MODEL_PATH)

# Initialize MinMaxScaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))

@api_view(['POST'])
def predict_rainfall(request):
    if request.method == 'POST':
        try:
            # Parse input data from the request body
            data = json.loads(request.body)
            logging.debug(f"Received data: {data}")

            # Ensure the 'rainfall' key exists in the request data
            if 'rainfall' not in data:
                return JsonResponse({'error': 'Missing rainfall data'}, status=400)

            rainfall_input = np.array(data['rainfall']).reshape(-1, 1)

            # Validate input shape
            if rainfall_input.shape[0] != 4:
                return JsonResponse({'error': 'Expected 4 days of rainfall data'}, status=400)

            # Normalize input data
            rainfall_input_scaled = scaler.fit_transform(rainfall_input)

            # Reshape data for LSTM model
            X_input = rainfall_input_scaled.reshape((1, 4, 1))

            # Make a prediction
            prediction_scaled = model.predict(X_input)

            # Inverse transform the prediction to get actual value
            prediction = scaler.inverse_transform(prediction_scaled)

            predicted_value = float(prediction[0][0])

            return JsonResponse({'predicted_rainfall': predicted_value})

            # return JsonResponse({'predicted_rainfall': prediction[0][0]})
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'detail': 'Method not allowed'}, status=405)
