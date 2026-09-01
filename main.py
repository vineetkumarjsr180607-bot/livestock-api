import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import whisper

# Initialize the Multi-modal API
app = FastAPI(title="Multimodal Livestock Disease Detection API - SIH")

# Allow frontend applications to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained CNN Image Model safely
MODEL_PATH = "best_livestock_model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["Healthy", "Lumpy_Skin", "Other_Infections"]

try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("CNN Model loaded successfully.")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

# Load Whisper model for voice note transcription
try:
    print("Loading Whisper model (small)...")
    whisper_model = whisper.load_model("small")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None


def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def extract_symptoms(transcript_text):
    text = transcript_text.lower()
    symptoms = []

    appetite_keywords = ["खाना नहीं", "खारे हैं", "चारा नहीं", "भूख नहीं", "दाणा नहीं", "खात नाही", "चारा खात नाही", "भूख नाही", "काही खात नाही", "khana", "eat", "feed", "appetite", "hunger", "not eating"]
    skin_keywords = ["निशान", "तोड़े", "शरीर पे", "गाँठ", "चकत्ते", "फुंसी", "छाले शरीर", "दाने", "गाठी", "पुरळ", "अंगावर डाग", "चट्टे", "त्वचा", "फोड", "skin", "nodules", "lumps", "bumps", "pocks"]
    fever_keywords = ["बुखार", "तेज गरम", "तप रहा", "शरीर गर्म", "ताप", "अंग गरम आहे", "ताप आलाय", "fever", "hot body", "temperature"]
    fmd_keywords = ["लार", "लाार", "मुंह से झाग", "खुर", "लंगड़ा", "पैर में घाव", "खुरपका", "लाळ गळणे", "तोंडाला लाळ", "खूर", "लंगडणे", "तोंडात फोड", "पायात घाव", "drooling", "saliva", "blisters in mouth", "limping", "foot lesion", "fmd"]
    mastitis_keywords = ["थैली", "थन", "सूजन", "दूध में खून", "दूध फटना", "दूध कम", "दूध खराब", "कास", "सूज", "दुधात रक्त", "दूध कमी", "सड", "सडांची सूज", "udder", "swollen teat", "blood in milk", "mastitis", "milk issue"]

    if any(kw in text for kw in appetite_keywords):
        symptoms.append("loss_of_appetite")
    if any(kw in text for kw in skin_keywords):
        symptoms.append("skin_lesions_or_nodules")
    if any(kw in text for kw in fever_keywords):
        symptoms.append("fever")
    if any(kw in text for kw in fmd_keywords):
        symptoms.append("fmd_indicators")
    if any(kw in text for kw in mastitis_keywords):
        symptoms.append("mastitis_indicators")

    return symptoms


# The Multi-Modal Prediction Endpoint
@app.post("/predict")
async def multimodal_predict(
    image_file: UploadFile = File(...), 
    audio_file: UploadFile = File(None)
):
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN Model is not loaded on server.")

    temp_audio_path = "temp_audio.wav"
    try:
        # 1. Process Image through EfficientNet
        img_bytes = await image_file.read()
        processed_image = prepare_image(img_bytes)
        predictions = cnn_model.predict(processed_image)[0]
        
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_index])

        # 2. Process Audio through Whisper & Symptom Extractor (if provided)
        transcript = ""
        detected_symptoms = []
        
        if audio_file and whisper_model:
            audio_bytes = await audio_file.read()
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)

            result = whisper_model.transcribe(temp_audio_path)
            transcript = result["text"].strip()
            detected_symptoms = extract_symptoms(transcript)
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        # 3. Return Combined Multimodal JSON Response
        return {
            "status": "success",
            "image_analysis": {
                "prediction": CLASSES[predicted_class_index],
                "confidence": round(confidence * 100, 2),
                "all_probabilities": {
                    CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))
                }
            },
            "audio_analysis": {
                "transcript": transcript,
                "detected_symptoms": detected_symptoms
            }
        }
        
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {"status": "active", "message": "Multimodal Livestock API is online."}