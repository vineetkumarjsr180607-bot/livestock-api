import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Initialize the API
app = FastAPI(title="Livestock Disease API")

# Allow full-stack frontend applications to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "best_livestock_model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["Healthy", "Lumpy_Skin", "Other_Infections"]

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        processed_image = prepare_image(contents)
        predictions = model.predict(processed_image)[0]
        
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_index])
        
        return {
            "status": "success",
            "prediction": CLASSES[predicted_class_index],
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "active", "message": "Livestock API is online."}