import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Multimodal Livestock Disease Detection API - SIH")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_livestock_model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["Healthy", "Lumpy_Skin", "Other_Infections"]

try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("CNN Model loaded successfully.")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_symptoms_from_text(transcript_text):
    text = transcript_text.lower()
    symptoms = []

    appetite_keywords = ["खाना नहीं", "खारे हैं", "चारा नहीं", "भूख नहीं", "खात नाही", "khana", "eat", "feed", "appetite", "hunger", "not eating"]
    skin_keywords = ["निशान", "तोड़े", "शरीर पे", "गाँठ", "चकत्ते", "फुंसी", "छाले", "दाने", "गाठी", "पुरळ", "skin", "nodules", "lumps", "bumps"]
    fever_keywords = ["बुखार", "तेज गरम", "तप रहा", "शरीर गर्म", "ताप", "ताप आलाय", "fever", "hot body", "temperature"]
    fmd_keywords = ["लार", "मुंह से झाग", "खुर", "लंगड़ा", "पैर में घाव", "खुरपका", "लाळ गळणे", "तोंडाला लाळ", "drooling", "saliva", "blisters", "limping", "fmd"]
    mastitis_keywords = ["थैली", "थन", "सूजन", "दूध में खून", "दूध फटना", "दूध कम", "कास", "सूज", "दुधात रक्त", "udder", "swollen teat", "blood in milk", "mastitis"]

    if any(kw in text for kw in appetite_keywords): symptoms.append("loss_of_appetite")
    if any(kw in text for kw in skin_keywords): symptoms.append("skin_lesions_or_nodules")
    if any(kw in text for kw in fever_keywords): symptoms.append("fever")
    if any(kw in text for kw in fmd_keywords): symptoms.append("fmd_indicators")
    if any(kw in text for kw in mastitis_keywords): symptoms.append("mastitis_indicators")

    return symptoms

@app.post("/predict")
async def multimodal_predict(
    image_file: UploadFile = File(...), 
    symptom_text: str = ""
):
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN Model is not loaded on server.")

    try:
        img_bytes = await image_file.read()
        processed_image = prepare_image(img_bytes)
        predictions = cnn_model.predict(processed_image)[0]
        
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_index])

        detected_symptoms = extract_symptoms_from_text(symptom_text)

        return {
            "status": "success",
            "image_analysis": {
                "prediction": CLASSES[predicted_class_index],
                "confidence": round(confidence * 100, 2),
                "all_probabilities": {
                    CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))
                }
            },
            "audio_or_text_analysis": {
                "input_text": symptom_text,
                "detected_symptoms": detected_symptoms
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "active", "message": "Lightweight Multimodal Livestock API is online."}