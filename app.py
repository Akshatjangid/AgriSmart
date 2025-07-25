import os
import joblib
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

# --- 1. Initialize App and Define Paths ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_MODEL_PATH = os.path.join(BASE_DIR, 'crop_model.pkl')
CROP_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.h5')
DISEASE_ENCODER_PATH = os.path.join(BASE_DIR, 'disease_label_encoder.pkl')

# --- 2. Load Models ---
try:
    print("Loading crop model...")
    crop_model = joblib.load(CROP_MODEL_PATH)
    print("Loading crop label encoder...")
    crop_label_encoder = joblib.load(CROP_ENCODER_PATH)
    print("Loading disease model...")
    disease_model = load_model(DISEASE_MODEL_PATH)
    print("Loading disease label encoder...")
    try:
        disease_label_encoder = joblib.load(DISEASE_ENCODER_PATH)
    except:
        import pickle
        disease_label_encoder = pickle.load(open(DISEASE_ENCODER_PATH, 'rb'))
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    crop_model, crop_label_encoder, disease_model, disease_label_encoder = None, None, None, None

# --- 3. Treatment Dictionary ---
# Comprehensive list of treatments for all diseases
treatments = {
    "Pepper_bell__Bacterial_spot": {
        "en": "Use copper-based fungicides. Remove and destroy infected plants. Avoid overhead watering.",
        "hi": "तांबे पर आधारित फफूंदनाशकों का प्रयोग करें। संक्रमित पौधों को हटा दें और नष्ट कर दें। ऊपर से पानी देने से बचें।"
    },
    "Potato___Early_blight": {
        "en": "Spray with fungicides containing mancozeb or chlorothalonil. Practice crop rotation and remove infected plant debris.",
        "hi": "मैनकोज़ेब या क्लोरोथालोनिल युक्त फफूंदनाशकों का छिड़काव करें। फसल चक्र अपनाएं और संक्रमित पौधों के मलबे को हटा दें।"
    },
    "Potato___Late_blight": {
        "en": "Apply fungicides like metalaxyl or chlorothalonil. Ensure good air circulation and destroy infected plants.",
        "hi": "मेटलैक्सिल या क्लोरोथालोनिल जैसे फफूंदनाशकों का प्रयोग करें। अच्छी हवा का संचार सुनिश्चित करें और संक्रमित पौधों को नष्ट कर दें।"
    },
    "Tomato__Target_Spot": {
        "en": "Use fungicides such as chlorothalonil or mancozeb. Improve air circulation and mulch around plants.",
        "hi": "क्लोरोथालोनिल या मैनकोजेब जैसे फफूंदनाशकों का प्रयोग करें। हवा के संचार में सुधार करें और पौधों के चारों ओर गीली घास बिछाएं।"
    },
    "Tomato__Tomato_mosaic_virus": {
        "en": "No cure. Remove and destroy infected plants. Control insects that spread the virus. Wash hands and tools.",
        "hi": "कोई इलाज नहीं। संक्रमित पौधों को हटा दें और नष्ट कर दें। वायरस फैलाने वाले कीड़ों को नियंत्रित करें। हाथ और औजार धो लें।"
    },
    "Tomato__Tomato_YellowLeaf_Curl_Virus": {
        "en": "No cure. Control whiteflies, which spread the virus. Use reflective mulches and remove infected plants immediately.",
        "hi": "कोई इलाज नहीं। वायरस फैलाने वाली सफेद मक्खियों को नियंत्रित करें। परावर्तक मल्च का प्रयोग करें और संक्रमित पौधों को तुरंत हटा दें।"
    },
    "Tomato_Bacterial_spot": {
        "en": "Apply copper-based bactericides. Avoid working with plants when they are wet. Use disease-free seeds.",
        "hi": "तांबे पर आधारित जीवाणुनाशकों का प्रयोग करें। जब पौधे गीले हों तो उनके साथ काम करने से बचें। रोग मुक्त बीजों का प्रयोग करें।"
    },
    "Tomato_Early_blight": {
        "en": "Apply fungicides like chlorothalonil or copper. Prune lower leaves and stake plants for better air circulation.",
        "hi": "क्लोरोथालोनिल या तांबे जैसे फफूंदनाशकों का प्रयोग करें। बेहतर वायु संचार के लिए निचली पत्तियों की छँटाई करें और पौधों को सहारा दें।"
    },
    "Tomato_Late_blight": {
        "en": "Use fungicides like chlorothalonil, mancozeb, or copper-based sprays. Ensure proper plant spacing for air circulation.",
        "hi": "क्लोरोथालोनिल, मैंकोजेब, या तांबे पर आधारित स्प्रे जैसे फफूंदनाशकों का प्रयोग करें। हवा के संचार के लिए पौधों के बीच उचित दूरी सुनिश्चित करें।"
    },
    "Tomato_Leaf_Mold": {
        "en": "Ensure good ventilation to lower humidity. Use fungicides like mancozeb or copper oxychloride. Stake plants.",
        "hi": "नमी कम करने के लिए अच्छा वेंटिलेशन सुनिश्चित करें। मैंकोजेब या कॉपर ऑक्सीक्लोराइड जैसे फफूंदनाशकों का प्रयोग करें। पौधों को सहारा दें।"
    },
    "Tomato_Septoria_leaf_spot": {
        "en": "Use fungicides containing chlorothalonil or mancozeb. Remove infected leaves and mulch the soil.",
        "hi": "क्लोरोथालोनिल या मैनकोजेब युक्त फफूंदनाशकों का प्रयोग करें। संक्रमित पत्तियों को हटा दें और मिट्टी में गीली घास डालें।"
    },
    "Tomato_Spider_mites_Two-spotted_spider_mite": {
        "en": "Use miticides like abamectin or spiromesifen. Spray plants with water to increase humidity, which mites dislike.",
        "hi": "एबामेक्टिन या स्पाइरोमेसिफेन जैसे माइटिसाइड्स का प्रयोग करें। नमी बढ़ाने के लिए पौधों पर पानी का छिड़काव करें, जो घुन को नापसंद है।"
    },
    "default": {
        "en": "This plant appears to be healthy, or treatment information is not available. Please consult a local agricultural expert for verification.",
        "hi": "यह पौधा स्वस्थ दिखाई देता है, या इलाज की जानकारी उपलब्ध नहीं है। कृपया सत्यापन के लिए किसी स्थानीय कृषि विशेषज्ञ से सलाह लें।"
    }
}


# --- 4. Page Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crop_form')
def crop_form():
    return render_template('crop_form.html')

@app.route('/disease_form')
def disease_form():
    return render_template('disease_form.html')


# --- 5. Prediction Logic ---
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if not all([crop_model, crop_label_encoder]):
        return render_template('result.html', crop="⚠️ Error: Crop model not loaded. Check server logs.")
    try:
        features = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        prediction_encoded = crop_model.predict([features])
        prediction_name = crop_label_encoder.inverse_transform(prediction_encoded)
        return render_template('result.html', crop=prediction_name[0])
    except Exception as e:
        return render_template('result.html', crop=f"⚠️ Error: {e}")


@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if not all([disease_model, disease_label_encoder]):
        return render_template('result.html', disease="⚠️ Error: Disease model not loaded. Check server logs.")

    try:
        print("📥 Received file for prediction")
        file = request.files.get('leaf_image')
        if not file:
            print("❌ No file submitted")
            return render_template('result.html', disease="⚠️ No image file submitted.")
        
        image = Image.open(io.BytesIO(file.read())).convert('RGB').resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        print("🤖 Predicting with disease model...")
        prediction_array = disease_model.predict(image_array)
        print("✅ Prediction complete")

        predicted_class_index = np.argmax(prediction_array)
        predicted_label = disease_label_encoder.classes_[predicted_class_index]
        print(f"🧠 Predicted class: {predicted_label}")

        treatment = treatments.get(predicted_label, treatments['default'])

        return render_template(
            'result.html',
            disease=predicted_label,
            treatment_en=treatment['en'],
            treatment_hi=treatment['hi']
        )
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return render_template('result.html', disease=f"⚠️ Error: {e}")


# --- 6. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
