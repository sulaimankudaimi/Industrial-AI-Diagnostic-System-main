import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import gdown
from PIL import Image

# --- إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="AI Sound Analyzer | Sulaiman Kudaimi", layout="wide")

# تصميم الهيدر باسمك
st.markdown(f"""
    <div style="background-color:#003366;padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">🏭 Industrial Machine Health AI</h1>
    <h3 style="color:#e0e0e0;text-align:center;">Designed & Developed by: <b>Sulaiman Kudaimi</b></h3>
    </div>
    """, unsafe_allow_html=True)

st.write("") # مسافة

# --- ربط الموديل من الدرايف ---
# قمت باستخراج المعرف (ID) من الرابط الذي أرسلته
MODEL_URL = 'https://drive.google.com/file/d/1xghQcu2rDtb6Jp4pvGWs0QUcMJM7NFaE/view?usp=drive_link'
MODEL_PATH = 'audio_anomaly_model.h5'

@st.cache_resource
def load_audio_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('🚀 جاري الاتصال بخوادم الدرايف لتحميل الموديل الذكي...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# --- دالة معالجة الصوت وتحويله لصورة ---
def process_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # حفظ المؤشر كصورة مؤقتة للتحليل
    fig, ax = plt.subplots(figsize=(2, 2))
    librosa.display.specshow(S_db, ax=ax)
    plt.axis('off')
    plt.savefig("temp_spec.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # تحويل الصورة لمصفوفة تدخل للموديل
    img = Image.open("temp_spec.png").convert('RGB').resize((128, 128))
    return np.array(img) / 255.0

# --- تنفيذ البرنامج ---
try:
    model = load_audio_model()
    st.sidebar.success("✅ AI Engine Connected")
except Exception as e:
    st.error(f"Error loading model: {e}")

# رفع الملف للتجربة
st.subheader("📤 Upload Machine Sound (.wav)")
uploaded_file = st.file_uploader("قم برفع ملف صوتي لماكينة (مروحة، مضخة، إلخ)", type=["wav"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🎵 Audio Signal Analysis")
        st.audio(uploaded_file)
        # معالجة وعرض الـ Spectrogram
        features = process_audio(uploaded_file)
        st.image("temp_spec.png", caption="Generated Spectrogram (AI Input)", use_container_width=True)

    with col2:
        st.info("🤖 AI Diagnostic Result")
        # التوقع
        prediction = model.predict(np.expand_dims(features, axis=0))
        
        # لنفترض أن الفئات هي [Normal, Abnormal] بناءً على تدريبك
        classes = ['Abnormal (عطل مكتشف)', 'Normal (حالة سليمة)']
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # عرض النتيجة بشكل أنيق
        if "Normal" in result:
            st.success(f"### Result: {result}")
        else:
            st.error(f"### Result: {result}")
            
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
        
        # شريط تقدم للثقة
        st.progress(int(confidence))

st.markdown("---")
st.caption("© 2026 Industrial AI Systems | Powered by Sulaiman Kudaimi Research")
