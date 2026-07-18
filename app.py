import os
print("RUNNING FROM:", os.getcwd())
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import certifi
from flask import Flask, render_template, request, session, redirect, url_for, flash
from functools import wraps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
from datetime import datetime
from recommendations import RECOMMENDATIONS

PREDICTION_TO_RECOMMENDATION = {
    "Apple___Apple_scab": "Apple__Apple_scab",
    "Apple___Black_rot": "Apple__Black_rot",
    "Apple___Cedar_apple_rust": "Apple__Cedar_apple_rust",
    "Apple___healthy": "Apple__healthy",
    
    "Corn___Cercospora_leaf_spot": "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust": "Corn_(maize)__Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)__Northern_Leaf_Blight",
    "Corn___healthy": "Corn_(maize)__healthy",
    
    "Potato___Early_blight": "Potato__Early_blight",
    "Potato___Late_blight": "Potato__Late_blight",
    "Potato___healthy": "Potato__healthy",
    
    "Strawberry___Leaf_scorch": "Strawberry__Leaf_scorch",
    "Strawberry___healthy": "Strawberry__healthy",
}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "agrolens_secret_key_123") # Required for sessions

# Initialize MongoDB (Using MongoDB Atlas)
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://rishiporwal2004_db_user:Lakshita2004@crop-history.vp7zog6.mongodb.net/?retryWrites=true&w=majority&appName=crop-history")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = client["crop_db"]
    history_collection = db["history"]
    users_collection = db["users"]
    client.server_info() # test connection
except Exception as e:
    print("Could not connect to MongoDB Atlas:", e)

# Load models
apple_model = load_model(os.path.join("models", "apple_model.h5"))
corn_model = load_model(os.path.join("models", "corn_model.h5"))
potato_model = load_model(os.path.join("models", "potato_model.h5"))
strawberry_model = load_model(os.path.join("models", "strawberry_model.h5"))

# Class labels
apple_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

corn_classes = [
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy"
]

potato_classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

strawberry_classes = [
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy"
]

img_size = 224

# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "phone" not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    prediction = None
    user_name = session.get("name", "User")

    if request.method == "POST":

        crop = request.form["crop"]
        file = request.files["file"]

        # Save uploaded image
        upload_folder = os.path.join("static", "uploads")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Select model
        if crop == "apple":
            model = apple_model
            classes = apple_classes

        elif crop == "corn":
            model = corn_model
            classes = corn_classes

        elif crop == "potato":
            model = potato_model
            classes = potato_classes

        elif crop == "strawberry":
            model = strawberry_model
            classes = strawberry_classes

        else:
            return "Invalid crop selected"

        # Prediction
        pred = model.predict(img_array)
        
        # Handle models that have more output nodes than defined classes (e.g. apple_model.h5)
        if pred.shape[1] > len(classes):
            pred = pred[:, :len(classes)]
            
        predicted_class_idx = np.argmax(pred)
        confidence_val = float(np.max(pred)) * 100
        confidence = f"{confidence_val:.1f}%"

        # Safety check (fix IndexError)
        if predicted_class_idx >= len(classes):
            flash("Please upload a clearer leaf image.")
            return redirect(url_for('index'))
        else:
            prediction = classes[predicted_class_idx]

        # Prepare status
        d_lower = prediction.lower()
        if "healthy" in d_lower:
            disease_status = "Healthy"
            severity = "None"
        else:
            disease_status = "Pathogen Detected"
            if "early" in d_lower:
                severity = "Mild"
            elif "late" in d_lower or "rot" in d_lower or "scorch" in d_lower:
                severity = "Severe"
            elif "rust" in d_lower or "spot" in d_lower or "scab" in d_lower:
                severity = "Moderate"
            else:
                severity = "High"
        
        # Save to database
        try:
            history_collection.insert_one({
                "user_phone": session.get("phone"),
                "crop": crop.capitalize(),
                "disease": prediction,
                "confidence": confidence,
                "severity": severity,
                "disease_status": disease_status,
                "filename": file.filename,
                "date": datetime.now()
            })
        except Exception as e:
            print("Failed to save history to database:", e)

        # Look up treatment recommendation
        rec_key = PREDICTION_TO_RECOMMENDATION.get(prediction)
        recommendation = RECOMMENDATIONS.get(rec_key)

        return render_template(
            "result.html", 
            prediction=prediction, 
            recommendation=recommendation,
            filename=file.filename, 
            confidence=confidence, 
            severity=severity, 
            disease_status=disease_status, 
            user_name=user_name
        )

    return render_template("index.html", prediction=prediction, user_name=user_name)

@app.route("/history")
@login_required
def history():
    user_phone = session.get("phone")
    try:
        history_records = list(history_collection.find({"user_phone": user_phone}).sort("date", -1))
    except Exception as e:
        print("Error fetching history:", e)
        history_records = []
    return render_template("history.html", history=history_records, user_name=session.get("name"))

@app.route("/about")
@login_required
def about():
    return render_template("about.html", user_name=session.get("name"))

@app.route("/profile")
@login_required
def profile():
    user_phone = session.get("phone")
    user = users_collection.find_one({"phone": user_phone})
    
    # Calculate some stats for the user
    total_scans = history_collection.count_documents({"user_phone": user_phone})
    healthy_scans = history_collection.count_documents({"user_phone": user_phone, "disease_status": "Healthy"})
    
    return render_template(
        "profile.html", 
        user=user, 
        total_scans=total_scans, 
        healthy_scans=healthy_scans,
        user_name=session.get("name")
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        phone = request.form.get("phone")
        password = request.form.get("password")
        
        if not phone or not password or not name:
            flash("All fields are required")
            return redirect(url_for("register"))
            
        existing_user = users_collection.find_one({"phone": phone})
        if existing_user:
            flash("Phone number already registered. Please login.")
            return redirect(url_for("login"))
            
        users_collection.insert_one({
            "name": name,
            "phone": phone,
            "password": password # Storing as plain text per user constraint
        })
        flash("Registration successful. Please log in.")
        return redirect(url_for("login"))
        
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phone = request.form.get("phone")
        password = request.form.get("password")
        
        user = users_collection.find_one({"phone": phone, "password": password})
        if user:
            session["phone"] = user["phone"]
            session["name"] = user.get("name", "User")
            return redirect(url_for("index"))
        else:
            flash("Invalid phone number or password")
            return redirect(url_for("login"))
            
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)