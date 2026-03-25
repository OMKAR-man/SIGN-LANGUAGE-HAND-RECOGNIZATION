import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import os
import time
from model_utils import HandSignClassifier

st.set_page_config(
    page_title="Hand Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .letter-display {
        font-size: 8rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        text-shadow: 2px 2px 8px rgba(102,126,234,0.3);
        line-height: 1;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 8px;
        height: 12px;
    }
    .prediction-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 2px solid #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-info {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize classifier
@st.cache_resource
def load_classifier():
    return HandSignClassifier()

def draw_hand_landmarks(image, hand_landmarks, mp_hands):
    """Draw hand landmarks on image"""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    annotated = image.copy()
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return annotated

def process_frame(frame, classifier):
    """Process a single frame and return prediction + annotated frame"""
    mp_hands = mp.solutions.hands
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        prediction = None
        confidence = 0.0
        annotated = draw_hand_landmarks(rgb, results.multi_hand_landmarks, mp_hands) if results.multi_hand_landmarks else rgb
        
        if results.multi_hand_landmarks:
            prediction, confidence = classifier.predict(rgb, results.multi_hand_landmarks[0])
        
        return annotated, prediction, confidence

def webcam_page(classifier):
    st.markdown("## 📷 Live Webcam Recognition")
    st.markdown("Show a hand sign to the camera and the system will recognize the letter in real time.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎥 Live Feed")
        cam_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🔍 Recognition View")
        rec_placeholder = st.empty()
    
    # Prediction display below cameras
    pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
    with pred_col2:
        letter_placeholder = st.empty()
        confidence_placeholder = st.empty()
        info_placeholder = st.empty()
    
    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    with ctrl_col2:
        start_btn = st.button("▶ Start Camera", key="start_cam")
        stop_btn = st.button("⏹ Stop Camera", key="stop_cam")
    
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    
    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False
    
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Please check your camera connection.")
            st.session_state.webcam_running = False
            return
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        word_built = []
        
        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                # Normal feed
                cam_placeholder.image(rgb, channels="RGB", use_container_width=True)
                
                # Recognition feed
                prediction = None
                confidence = 0.0
                
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                annotated = rgb.copy()
                
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated, lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    prediction, confidence = classifier.predict(rgb, results.multi_hand_landmarks[0])
                
                # Add overlay text on recognition frame
                if prediction:
                    cv2.putText(annotated, f"{prediction} ({confidence:.0%})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (102, 126, 234), 3)
                
                rec_placeholder.image(annotated, channels="RGB", use_container_width=True)
                
                # Letter display
                if prediction and confidence > 0.6:
                    letter_placeholder.markdown(f"""
                    <div class="prediction-box">
                        <div class="letter-display">{prediction}</div>
                        <p style="font-size:1.2rem; color:#666;">Detected Letter</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    conf_pct = int(confidence * 100)
                    confidence_placeholder.markdown(f"""
                    <div style="margin:0.5rem 0;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span>Confidence</span><span><b>{conf_pct}%</b></span>
                        </div>
                        <div style="background:#e9ecef;border-radius:8px;height:12px;">
                            <div class="confidence-bar" style="width:{conf_pct}%;height:12px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    letter_placeholder.markdown("""
                    <div class="prediction-box">
                        <div style="font-size:3rem;">🤚</div>
                        <p style="color:#999;">Show a hand sign...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    confidence_placeholder.empty()
                
                time.sleep(0.05)
        
        finally:
            cap.release()
            hands.close()
    else:
        # Placeholder when camera is off
        cam_placeholder.markdown("""
        <div style="background:#f0f0f0;border-radius:12px;height:300px;display:flex;align-items:center;
        justify-content:center;border:2px dashed #ccc;">
            <div style="text-align:center;color:#999;">
                <div style="font-size:4rem;">📷</div>
                <p>Camera Off</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        rec_placeholder.markdown("""
        <div style="background:#f0f0f0;border-radius:12px;height:300px;display:flex;align-items:center;
        justify-content:center;border:2px dashed #ccc;">
            <div style="text-align:center;color:#999;">
                <div style="font-size:4rem;">🔍</div>
                <p>Recognition View</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def video_upload_page(classifier):
    st.markdown("## 📁 Video Upload Recognition")
    st.markdown("Upload a video file and the system will detect hand signs frame by frame.")
    
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
    )
    
    if uploaded:
        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        st.success(f"✅ Video uploaded: **{uploaded.name}**")
        
        # Options
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            skip_frames = st.slider("Process every Nth frame", 1, 10, 3)
        with opt_col2:
            confidence_threshold = st.slider("Min confidence", 0.3, 0.95, 0.6)
        with opt_col3:
            show_all_frames = st.checkbox("Show all frames", value=False)
        
        if st.button("🚀 Start Recognition", key="process_video"):
            process_video(tmp_path, classifier, skip_frames, confidence_threshold, show_all_frames)
        
        os.unlink(tmp_path)

def process_video(video_path, classifier, skip_frames, confidence_threshold, show_all_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.markdown(f"**Video Info:** {total_frames} frames | {fps:.1f} FPS | Duration: {total_frames/fps:.1f}s")
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎬 Processing View")
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with col2:
        st.markdown("### 📊 Results")
        results_placeholder = st.empty()
    
    detections = []
    frame_idx = 0
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            progress = frame_idx / total_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_idx}/{total_frames}")
            
            if frame_idx % skip_frames != 0 and not show_all_frames:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            annotated = rgb.copy()
            prediction = None
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                prediction, confidence = classifier.predict(rgb, results.multi_hand_landmarks[0])
            
            if prediction and confidence >= confidence_threshold:
                detections.append({
                    "frame": frame_idx,
                    "time": frame_idx / max(fps, 1),
                    "letter": prediction,
                    "confidence": confidence
                })
                # Draw prediction on frame
                cv2.putText(annotated, f"  {prediction}  ({confidence:.0%})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (102, 126, 234), 4)
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]),
                              (102, 126, 234), 8)
            
            if frame_idx % (skip_frames * 2) == 0 or show_all_frames:
                frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
            
            # Live results update
            if detections:
                last = detections[-1]
                # Build letter frequency
                from collections import Counter
                letter_counts = Counter([d["letter"] for d in detections])
                top_letters = letter_counts.most_common(5)
                
                results_html = f"""
                <div style="padding:1rem;">
                    <h4>Latest Detection</h4>
                    <div class="letter-display" style="font-size:4rem;">{last['letter']}</div>
                    <p>Frame {last['frame']} | {last['time']:.1f}s | {last['confidence']:.0%}</p>
                    <hr>
                    <h4>Top Letters</h4>
                    {''.join(f'<div style="display:flex;justify-content:space-between;padding:4px 0;"><b>{l}</b><span>{c} times</span></div>' for l,c in top_letters)}
                    <hr>
                    <p>Total detections: <b>{len(detections)}</b></p>
                </div>
                """
                results_placeholder.markdown(results_html, unsafe_allow_html=True)
    
    finally:
        cap.release()
        hands.close()
    
    # Final summary
    st.markdown("---")
    st.markdown("### 🏁 Recognition Complete!")
    
    if detections:
        from collections import Counter
        
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.metric("Total Detections", len(detections))
        with sum_col2:
            letter_counts = Counter([d["letter"] for d in detections])
            most_common = letter_counts.most_common(1)[0]
            st.metric("Most Common", most_common[0], f"{most_common[1]} times")
        with sum_col3:
            avg_conf = np.mean([d["confidence"] for d in detections])
            st.metric("Avg Confidence", f"{avg_conf:.0%}")
        
        # Detection timeline
        st.markdown("#### 📋 Detection Timeline")
        timeline_data = [{"Time (s)": f"{d['time']:.1f}", "Frame": d["frame"],
                           "Letter": d["letter"], "Confidence": f"{d['confidence']:.0%}"}
                         for d in detections[-20:]]
        st.dataframe(timeline_data, use_container_width=True)
        
        # Unique letters
        unique_letters = sorted(set(d["letter"] for d in detections))
        st.markdown(f"**Detected letters:** {' · '.join(unique_letters)}")
        
        # Try to form word
        if len(unique_letters) > 1:
            st.markdown(f"**Word attempt:** `{''.join(unique_letters)}`")
    else:
        st.warning("No hand signs detected. Try lowering the confidence threshold or checking your video.")

def dataset_info_page():
    st.markdown("## 📂 Dataset Information")
    
    st.markdown("""
    ### Expected Dataset Structure
    
    Place your dataset folder in the project root. The classifier will automatically 
    scan for letter folders and images.
    """)
    
    st.code("""
    dataset/
    ├── A/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    ├── B/
    │   ├── image001.jpg
    │   └── ...
    ├── C/
    │   └── ...
    └── Z/
        └── ...
    """)
    
    dataset_path = st.text_input("Dataset path", value="dataset", 
                                   help="Path to your dataset folder")
    
    if st.button("🔄 Train/Retrain Model"):
        with st.spinner("Training model... This may take a few minutes."):
            from model_utils import HandSignClassifier
            classifier = HandSignClassifier()
            result = classifier.train(dataset_path)
            if result:
                st.success("✅ Model trained successfully!")
                st.cache_resource.clear()
            else:
                st.error("❌ Training failed. Check the dataset path and structure.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Supported image formats:**
        - JPG / JPEG
        - PNG
        - BMP
        - WebP
        """)
    with col2:
        st.info("""
        **Recommendations:**
        - 100+ images per letter
        - Varied backgrounds
        - Different lighting
        - Multiple hand sizes
        """)

# ─── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="text-align:center;padding:1rem 0;">
    <div style="font-size:3rem;">🤟</div>
    <h2 style="margin:0;">Hand Sign<br>Recognition</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📷 Webcam", "📁 Upload Video", "📂 Dataset & Training"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ How it works")
st.sidebar.markdown("""
<div class="sidebar-info">
1. 🖐️ MediaPipe detects hand landmarks
2. 🧠 ML model classifies the gesture
3. 🔤 The matched letter is displayed
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Model Status")

# Check model
classifier = load_classifier()
if classifier.is_trained():
    classes = classifier.get_classes()
    st.sidebar.success(f"✅ Model loaded ({len(classes)} classes)")
    st.sidebar.markdown(f"**Letters:** `{'  '.join(sorted(classes))}`")
else:
    st.sidebar.warning("⚠️ No trained model found.\nGo to **Dataset & Training** to train.")

# ─── Main Content ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🤟 Hand Sign Recognition System</h1>
    <p>ASL / Custom Hand Gesture Recognition powered by MediaPipe & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if page == "📷 Webcam":
    webcam_page(classifier)
elif page == "📁 Upload Video":
    video_upload_page(classifier)
elif page == "📂 Dataset & Training":
    dataset_info_page()