import streamlit as st
import numpy as np
import time
import tensorflow as tf
from collections import deque
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import opencv with better error handling
try:
    import cv2
    logger.info("OpenCV imported successfully")
except ImportError as e:
    st.error(f"Error importing OpenCV: {e}")
    st.error("Please make sure opencv-python-headless is installed.")

# Import mediapipe with better error handling
try:
    import mediapipe as mp
    logger.info("MediaPipe imported successfully")
except ImportError as e:
    st.error(f"Error importing MediaPipe: {e}")
    st.error("Please make sure mediapipe is installed.")

# Display app information and environment details
st.sidebar.markdown("## Environment Info")
st.sidebar.write(f"TensorFlow version: {tf.__version__}")
try:
    st.sidebar.write(f"OpenCV version: {cv2.__version__}")
except:
    st.sidebar.write("OpenCV not available")
try:
    st.sidebar.write(f"MediaPipe version: {mp.__version__}")
except:
    st.sidebar.write("MediaPipe not available")

# Set page configuration
st.set_page_config(
    page_title="ASL Recognition",
    page_icon="👋",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .prediction-box {
        background-color: #2E2E2E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .instruction-box {
        background-color: #333333;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .error-box {
        background-color: #ffcccc;
        color: #990000;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("ASL Sign Language Recognition")
st.markdown("""
<div class="instruction-box">
    Show hand signs in the webcam feed to recognize ASL letters.
    Hold your hand steady for the system to recognize the sign.
</div>
""", unsafe_allow_html=True)

# Cloud deployment detection
is_cloud = os.environ.get('STREAMLIT_CLOUD', '') or 'HOSTED' in os.environ
if is_cloud:
    st.info("Running in cloud environment. Camera functionality may be limited.")

# Initialize MediaPipe Hands with proper error handling
@st.cache_resource
def load_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return mp_hands, mp_drawing, mp_drawing_styles, hands, True
    except Exception as e:
        logger.error(f"Error initializing MediaPipe: {e}")
        return None, None, None, None, False

# Load MediaPipe
mp_hands, mp_drawing, mp_drawing_styles, hands, mediapipe_loaded = load_mediapipe()
if not mediapipe_loaded:
    st.warning("MediaPipe could not be initialized. Hand detection will not be available.")

# Load the model with better error handling
@st.cache_resource
def load_asl_model():
    try:
        with st.spinner("Loading ASL recognition model..."):
            model_path = "asl_model.h5"
            if not os.path.exists(model_path):
                st.warning(f"Model file {model_path} not found. Looking for alternative models...")
                # Try alternative model name
                alt_model_path = "asl_model_advanced.h5"
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                    st.info(f"Using alternative model: {alt_model_path}")
                else:
                    return None, False, "Model file not found"
            
            logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
            return model, True, None
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        return None, False, error_msg

# ASL Labels
labels = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R',
    'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# App state initialization
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=10)
if 'current_letter' not in st.session_state:
    st.session_state.current_letter = "?"
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0
if 'detection_start_time' not in st.session_state:
    st.session_state.detection_start_time = None
if 'letter_confirmed' not in st.session_state:
    st.session_state.letter_confirmed = False
if 'display_text' not in st.session_state:
    st.session_state.display_text = ""
if 'is_camera_on' not in st.session_state:
    st.session_state.is_camera_on = False
if 'fps_history' not in st.session_state:
    st.session_state.fps_history = deque(maxlen=30)
if 'camera_error' not in st.session_state:
    st.session_state.camera_error = None

# Constants
img_size = 96  # Match model input size from training
confidence_threshold = 0.6  # Minimum confidence to accept a prediction

# Helper functions
def crop_hand_region(frame, landmarks):
    """Extract hand region from the frame based on MediaPipe landmarks"""
    try:
        # Get coordinates for hand bounding box
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        # Add padding around the hand
        padding = 0.15
        
        # Calculate the bounding box with padding
        x_min = max(0, int(min(x_coords) * frame.shape[1] - padding * frame.shape[1]))
        y_min = max(0, int(min(y_coords) * frame.shape[0] - padding * frame.shape[0]))
        x_max = min(frame.shape[1], int(max(x_coords) * frame.shape[1] + padding * frame.shape[1]))
        y_max = min(frame.shape[0], int(max(y_coords) * frame.shape[0] + padding * frame.shape[0]))
        
        # Ensure we have a square crop (important for ASL signs)
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)
        
        # Center the crop box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Get the new coordinates for a square crop
        x_min = max(0, center_x - max_dim // 2)
        y_min = max(0, center_y - max_dim // 2)
        x_max = min(frame.shape[1], x_min + max_dim)
        y_max = min(frame.shape[0], y_min + max_dim)
        
        # Return the cropped hand region and the bounding box coordinates
        if x_max > x_min and y_max > y_min:
            return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
        else:
            return None, (0, 0, 0, 0)
    except Exception as e:
        logger.error(f"Error in crop_hand_region: {e}")
        return None, (0, 0, 0, 0)

def preprocess_image(img):
    """Preprocess the image for model prediction"""
    try:
        if img is None or img.size == 0:
            return None
        
        # Resize to model's expected input size
        img = cv2.resize(img, (img_size, img_size))
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        return None

def get_stable_prediction():
    """Get a stable prediction from the history"""
    try:
        if not st.session_state.prediction_history:
            return "unknown", 0.0
        
        # Count predictions
        prediction_counts = {}
        confidence_sums = {}
        
        for pred, conf in st.session_state.prediction_history:
            if pred in prediction_counts:
                prediction_counts[pred] += 1
                confidence_sums[pred] += conf
            else:
                prediction_counts[pred] = 1
                confidence_sums[pred] = conf
        
        # Get most frequent prediction with highest confidence
        most_frequent = max(prediction_counts, key=prediction_counts.get)
        average_confidence = confidence_sums[most_frequent] / prediction_counts[most_frequent]
        
        # Only return if confidence is above threshold
        if average_confidence >= confidence_threshold:
            return most_frequent, average_confidence
        else:
            return "unknown", average_confidence
    except Exception as e:
        logger.error(f"Error in get_stable_prediction: {e}")
        return "unknown", 0.0

def reset_text():
    """Reset the display text"""
    st.session_state.display_text = ""

def toggle_camera():
    """Toggle the camera on/off state"""
    st.session_state.is_camera_on = not st.session_state.is_camera_on
    st.session_state.camera_error = None  # Reset camera error state
    if not st.session_state.is_camera_on:
        # Reset states when turning off
        st.session_state.prediction_history.clear()
        st.session_state.detection_start_time = None
        st.session_state.letter_confirmed = False

# Load model when the app starts
model, model_loaded, model_error = load_asl_model()

if not model_loaded:
    st.error(f"Failed to load model: {model_error}")
    st.warning("Demo mode only: The app will run with limited functionality.")

# UI Layout
col1, col2 = st.columns([3, 1])

# Right sidebar for controls and information
with col2:
    st.markdown("<h2>Controls</h2>", unsafe_allow_html=True)
    
    # Camera toggle button
    if st.session_state.is_camera_on:
        if st.button("Stop Camera", key="camera_toggle"):
            toggle_camera()
    else:
        if st.button("Start Camera", key="camera_toggle"):
            toggle_camera()
    
    # Reset text button
    st.button("Clear Text", on_click=reset_text)
    
    # Display current recognition
    st.markdown("<h2>Recognition</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; font-size: 48px;'>{st.session_state.current_letter}</h3>", unsafe_allow_html=True)
    
    # Show confidence with progress bar
    if st.session_state.current_confidence > 0:
        st.progress(min(st.session_state.current_confidence, 1.0))
        st.text(f"Confidence: {st.session_state.current_confidence*100:.1f}%")
    else:
        st.progress(0.0)
        st.text("Confidence: N/A")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Hold timer visualization (if currently detecting)
    if st.session_state.detection_start_time is not None:
        hold_time = time.time() - st.session_state.detection_start_time
        if hold_time <= 1.0:
            st.text("Hold steady to confirm:")
            st.progress(min(hold_time, 1.0))
    
    # Display recognized text
    st.markdown("<h2>Text Output</h2>", unsafe_allow_html=True)
    st.text_area("", value=st.session_state.display_text, height=100, key="text_output")
    
    # Instructions
    st.markdown("<h2>Instructions</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. Start the camera
    2. Show ASL hand signs in frame
    3. Hold signs steady for 1 second
    4. Use 'space' sign for spaces
    5. Use 'del' sign to backspace
    """)

# Main area for webcam display
with col1:
    # Show camera error if exists
    if st.session_state.camera_error:
        st.error(f"Camera Error: {st.session_state.camera_error}")
    
    # Placeholder for webcam feed
    video_placeholder = st.empty()

    # Only run the webcam if toggled on and model is loaded
    if st.session_state.is_camera_on:
        # Create a frame placeholder
        frame_placeholder = video_placeholder.empty()
        
        # Start webcam capture with better error handling
        try:
            cap = cv2.VideoCapture(0)
            
            # Set camera properties if supported
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                raise Exception("Could not open webcam. Browser might not support camera access in this environment.")
            
            # Camera successfully opened, proceed
            logger.info("Camera initialized successfully")
            
            # FPS calculation
            prev_frame_time = time.time()
            
            try:
                while st.session_state.is_camera_on:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.camera_error = "Failed to capture image from webcam"
                        break
                    
                    # Calculate FPS
                    new_frame_time = time.time()
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time
                    st.session_state.fps_history.append(fps)
                    
                    # Flip the frame horizontally (mirror effect)
                    frame = cv2.flip(frame, 1)
                    
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe if available
                    hand_detected = False
                    if mediapipe_loaded and hands is not None:
                        results = hands.process(rgb_frame)
                        
                        # Process hand landmarks if detected
                        if results.multi_hand_landmarks:
                            hand_detected = True
                            for hand_landmarks in results.multi_hand_landmarks:
                                # Draw landmarks on the frame
                                mp_drawing.draw_landmarks(
                                    frame, 
                                    hand_landmarks, 
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                                
                                # Extract the hand region based on landmarks
                                hand_img, (x_min, y_min, x_max, y_max) = crop_hand_region(frame, hand_landmarks.landmark)
                                
                                if hand_img is not None:
                                    # Draw bounding box around hand
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                    
                                    # Only process if model is loaded
                                    if model_loaded and model is not None:
                                        # Preprocess the hand image for the model
                                        processed_img = preprocess_image(hand_img)
                                        
                                        if processed_img is not None:
                                            try:
                                                # Make prediction
                                                preds = model.predict(processed_img, verbose=0)
                                                
                                                # Get top prediction
                                                idx = np.argmax(preds[0])
                                                confidence = preds[0][idx]
                                                
                                                # Add to prediction history only if confidence is above minimum threshold
                                                if confidence > 0.3:  # Lower threshold for adding to history
                                                    st.session_state.prediction_history.append((labels[idx], confidence))
                                                
                                                # Get stable prediction
                                                pred_letter, pred_confidence = get_stable_prediction()
                                                
                                                # Update current prediction
                                                if pred_letter != "unknown":
                                                    st.session_state.current_letter = pred_letter
                                                    st.session_state.current_confidence = pred_confidence
                                                    
                                                    # Start timing for confirmation
                                                    if st.session_state.detection_start_time is None:
                                                        st.session_state.detection_start_time = time.time()
                                                    
                                                    # Confirm letter after holding for 1 second
                                                    elapsed = time.time() - st.session_state.detection_start_time
                                                    if elapsed > 1.0 and not st.session_state.letter_confirmed:
                                                        st.session_state.letter_confirmed = True
                                                        if pred_letter == "space":
                                                            st.session_state.display_text += " "
                                                        elif pred_letter == "del" and st.session_state.display_text:
                                                            st.session_state.display_text = st.session_state.display_text[:-1]
                                                        elif pred_letter != "nothing":
                                                            st.session_state.display_text += pred_letter
                                                else:
                                                    # Reset detection timer if prediction changes
                                                    st.session_state.detection_start_time = None
                                                    st.session_state.letter_confirmed = False
                                            except Exception as e:
                                                logger.error(f"Error in prediction: {e}")
                    
                    # Reset if no hand detected
                    if not hand_detected:
                        st.session_state.detection_start_time = None
                        st.session_state.letter_confirmed = False
                        st.session_state.prediction_history.clear()
                        st.session_state.current_letter = "?"
                        st.session_state.current_confidence = 0.0
                    
                    # Add FPS display
                    avg_fps = sum(st.session_state.fps_history) / len(st.session_state.fps_history) if st.session_state.fps_history else 0
                    cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Convert frame to RGB for display
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update the frame in the app
                    frame_placeholder.image(display_frame, channels="RGB", use_column_width=True)
                    
                    # Check if the button state has changed
                    if not st.session_state.is_camera_on:
                        break
                    
                    # Small sleep to reduce CPU usage
                    time.sleep(0.01)
                    
            except Exception as e:
                error_msg = f"Error during camera operation: {str(e)}"
                logger.error(error_msg)
                st.session_state.camera_error = error_msg
            finally:
                # Release resources
                cap.release()
        except Exception as e:
            error_msg = f"Camera initialization error: {str(e)}"
            logger.error(error_msg)
            st.session_state.camera_error = error_msg
            video_placeholder.error("Could not access webcam. This feature may not be available in cloud environments.")
            video_placeholder.image("https://via.placeholder.com/640x480.png?text=Camera+Not+Available", use_column_width=True)
    else:
        # Show placeholder when camera is off
        video_placeholder.image("https://via.placeholder.com/640x480.png?text=Camera+Off", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("Made with Streamlit, TensorFlow, and MediaPipe")

# Add a simple error handler for common issues
st.sidebar.markdown("## Troubleshooting")
st.sidebar.markdown("""
If you encounter issues:
1. Make sure your webcam is connected
2. Allow browser camera access
3. Try refreshing the page
4. Ensure good lighting
5. Note: Camera functionality may be limited in Streamlit Cloud
""")

# Add download button for the recognized text
if st.session_state.display_text:
    st.sidebar.download_button(
        label="Download Text",
        data=st.session_state.display_text,
        file_name="asl_recognized_text.txt",
        mime="text/plain"
    )

# Display debug panel in sidebar
with st.sidebar.expander("Debug Information"):
    st.write("Runtime Environment:")
    st.write(f"- Deployed in cloud: {is_cloud}")
    st.write(f"- MediaPipe loaded: {mediapipe_loaded}")
    st.write(f"- Model loaded: {model_loaded}")
    if not model_loaded:
        st.write(f"- Model error: {model_error}")
    st.write(f"- Camera error: {st.session_state.camera_error}")
    
    # Add package information
    st.write("Installed Packages:")
    try:
        import pkg_resources
        pkg_list = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        st.code("\n".join(pkg_list[:10] + ["..."]))
    except:
        st.write("Could not retrieve package information")
