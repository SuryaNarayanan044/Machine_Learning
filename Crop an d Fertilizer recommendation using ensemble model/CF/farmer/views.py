from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.http import HttpResponseRedirect

# Render the index page
def index(request):
    return render(request, "index.html")

# Signup functionality
def signup(request):
    if request.method == "POST":
        un = request.POST.get('username')
        em = request.POST.get('email')
        pw = request.POST.get('password')
        cpw = request.POST.get('confirm_password')

        # Password confirmation check
        if pw == cpw:
            # Check if email or username already exists
            if User.objects.filter(email=em).exists():
                messages.info(request, "Email already exists")
                return render(request, "signup.html")
            elif User.objects.filter(username=un).exists():
                messages.info(request, "Username already exists")
                return render(request, "signup.html")
            else:
                # Create a new user
                user = User.objects.create_user(username=un, email=em, password=pw)
                user.save()
                messages.success(request, "Signup successful!")
                return redirect('login')  # Redirect to the login page
        else:
            # Passwords do not match
            messages.info(request, "Password and Confirm Password do not match")
            return render(request, "signup.html")
    
    return render(request, "signup.html")

# Login functionality
def login(request):
    if request.method == "POST":
        un = request.POST.get('username')
        pw = request.POST.get('password')

        # Authenticate user
        user = auth.authenticate(username=un, password=pw)

        if user is not None:
            # Login the user and redirect to the index page
            auth.login(request, user)
            return redirect('index')  # Ensure 'index' is defined in your urls.py
        else:
            # Invalid credentials
            messages.info(request, "Invalid username or password")
            return render(request, "login.html")
    
    return render(request, "login.html")

# Logout functionality
def logout(request):
    auth.logout(request)
    return redirect('index')  # Ensure 'index' is defined in your urls.py


def recommendation(request):

    return render(request, "recommendation.html")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render



def crop_recommendation(request):
        # Sample dataset (replace with your actual dataset)
    data = pd.read_csv('D:\CF\CF\static\dataset\Crop_recommendation.csv')  # Replace with actual data path
    X = data.drop('crop', axis=1)  # Features
    y = data['crop']  # Target variable (Crop prediction)

    # Encode the target variable (if it's categorical)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the models
    rf_model.fit(X, y_encoded)
    xgb_model.fit(X, y_encoded)
    if request.method == 'POST':
        # Extract form data
        nitrogen = float(request.POST['nitrogen'])
        phosphorus = float(request.POST['phosphorus'])
        potassium = float(request.POST['potassium'])
        temperature = float(request.POST['temperature'])
        humidity = float(request.POST['humidity'])
        ph = float(request.POST['ph'])
        rainfall = float(request.POST['rainfall'])
        

        # Combine the input values into a DataFrame with the same column names as during training
        user_input = pd.DataFrame({
            'N': [nitrogen],            # Match the name used in training
            'P': [phosphorus],         # Match the name used in training
            'K': [potassium],          # Match the name used in training
            'temperature': [temperature], # Match the name used in training
            'humidity': [humidity],     # Match the name used in training
            'ph': [ph],                 # Match the name used in training
            'rainfall': [rainfall]     # Match the name used in training
            
        })

        # Make predictions using both models
        rf_pred = rf_model.predict(user_input)
        xgb_pred = xgb_model.predict(user_input)

        # Combine predictions using simple voting or majority vote
        stacked_pred = np.round((rf_pred + xgb_pred) / 2).astype(int)

        # Decode the predicted crop using the label encoder
        predicted_crop = label_encoder.inverse_transform(stacked_pred)

        # Return the prediction as a response to the user in the same page (recommendation.html)
        return render(request, 'crop_recommend.html', {'predicted_crop': predicted_crop[0]})

    # If GET request, render the form page
    return render(request, 'recommendation.html')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render

# Load the dataset and prepare the models when the server starts
data = pd.read_csv('D:\\CF\\CF\\static\\dataset\\Fertilizer Prediction.csv')  # Replace with actual data path
X = data.drop('Fertilizer', axis=1)  # Features (excluding target variable)
y = data['Fertilizer']  # Target variable (Fertilizer recommendation)

# Initialize LabelEncoders for categorical variables
label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()
label_encoder_fertilizer = LabelEncoder()  # New encoder for the target variable

# Encode categorical variables (like Soil and Crop)
X['Soil'] = label_encoder_soil.fit_transform(X['Soil'])  # Transform soil type to numerical
X['Crop'] = label_encoder_crop.fit_transform(X['Crop'])  # Transform crop type to numerical
y = label_encoder_fertilizer.fit_transform(y)  # Encode the target variable

# Initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the models
rf_model.fit(X, y)
xgb_model.fit(X, y)

def fertilizer_recommendation(request):
    if request.method == 'POST':
        # Extract form data from the POST request
        temperature = float(request.POST['temp'])
        humidity = float(request.POST['hum'])
        moisture = float(request.POST['moisture'])
        soil = request.POST['soiltype']  # Keep soil type as a string for encoding
        crop = request.POST['croptype']  # Keep crop type as a string for encoding
        nitrogen = float(request.POST['nitro'])
        phosphorus = float(request.POST['phos'])
        potassium = float(request.POST['pot'])

        # Check if soil and crop types are valid
        if soil not in label_encoder_soil.classes_ or crop not in label_encoder_crop.classes_:
            return render(request, 'fertilizer-recommend.html', {'error': 'Invalid soil or crop type entered.'})

        # Encode the soil type
        encoded_soil_type = label_encoder_soil.transform([soil])[0]  # Transform soil type to numerical
        encoded_crop_type = label_encoder_crop.transform([crop])[0]  # Transform crop type to numerical

        # Combine the input values into a DataFrame
        user_input = pd.DataFrame({
            'Temparature': [temperature],  # Match the name used in training
            'Humidity ': [humidity],        # Match the name used in training
            'Moisture': [moisture],        # Match the name used in training
            'Soil': [encoded_soil_type],   # Encoded soil type
            'Crop': [encoded_crop_type],   # Encoded crop type
            'N': [nitrogen],               # Match the name used in training
            'P': [phosphorus],             # Match the name used in training
            'K': [potassium]               # Match the name used in training
        })

        # Make predictions using both models
        rf_pred = rf_model.predict(user_input)
        xgb_pred = xgb_model.predict(user_input)

        # Combine predictions (here you can use averaging, or any other method)
        final_prediction_encoded = np.round((rf_pred + xgb_pred) / 2).astype(int)

        # Decode the final prediction to get the fertilizer name
        final_prediction = label_encoder_fertilizer.inverse_transform(final_prediction_encoded)

        # Return the prediction as a response to the user in the same page (fertilizer_recommend.html)
        return render(request, 'fertilizer-recommend.html', {'predicted_fertilizer': final_prediction[0]})

    # If GET request, render the form page
    return render(request, 'recommendation.html')

