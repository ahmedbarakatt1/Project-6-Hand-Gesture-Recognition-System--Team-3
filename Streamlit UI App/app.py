import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


st.set_page_config(page_title="Milestone 3: Gesture AI", page_icon="🖐️", layout="wide")

IMG_SIZE = 128
LABELS_MAP = {
    0: '01_palm', 1: '02_l', 2: '03_fist', 3: '04_fist_moved',
    4: '05_thumb', 5: '06_index', 6: '07_ok', 7: '08_palm_moved',
    8: '09_c', 9: '10_down'
}


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('hand_gesture_model.keras')
        return model
    except:
        return None

model = load_model()



def smart_preprocess(img_array):
    
   
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(img_YCrCb, lower_skin, upper_skin)
    
    kernel = np.ones((5,5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    
    return skin_mask



def get_action(gesture_name):
    if "fist" in gesture_name: return " DRAG ITEM", "error"      # Red
    if "palm" in gesture_name: return "STOP VIDEO", "warning"   # Orange
    if "ok" in gesture_name: return " CONFIRM", "success"       # Green
    if "index" in gesture_name: return " CLICK / POINT", "info" # Blue
    if "down" in gesture_name: return "⬇SCROLL DOWN", "primary" # Purple
    if "l" in gesture_name: return " GO BACK", "secondary"      # Gray
    if "c" in gesture_name: return " COPY TEXT", "success"
    return "WAITING...", "secondary"


st.sidebar.title("Control Panel")
if model:
    st.sidebar.success(" Model Loaded Successfully")
else:
    st.sidebar.error(" Model NOT Found! Please check the file.")
st.sidebar.info(" Tip: Use a plain background and ensure your hand is well-lit.")

st.title(" Milestone 3: Hand Gesture Recognition")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Camera Input")
    camera_image = st.camera_input("Take a snapshot")

with col2:
    st.subheader("AI Analysis")
    
    if camera_image:
        image = Image.open(camera_image)
        img_array = np.array(image)
        
        processed_mask = smart_preprocess(img_array)
        resized_mask = cv2.resize(processed_mask, (IMG_SIZE, IMG_SIZE))
        
        img_normal = resized_mask
        img_flipped = cv2.flip(resized_mask, 1)
        
    
        input_normal = img_normal.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        input_flipped = img_flipped.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        
        if model:
            pred_normal = model.predict(input_normal, verbose=0)
            pred_flipped = model.predict(input_flipped, verbose=0)
            
            if np.max(pred_flipped) > np.max(pred_normal):
                final_pred = pred_flipped
                status = "Auto-Flipped (Right Hand)"
                final_img_display = img_flipped
            else:
                final_pred = pred_normal
                status = "Normal (Left Hand)"
                final_img_display = img_normal
                
            idx = np.argmax(final_pred)
            conf = np.max(final_pred)
            gesture_name = LABELS_MAP.get(idx, "Unknown")
        
            action_text, msg_type = get_action(gesture_name)
    
            st.image(final_img_display, caption=f"AI Input Mask ({status})", width=200)
            
            st.metric("Detected Gesture", gesture_name, f"{conf*100:.1f}% Confidence")
       
            if msg_type == "error": st.error(f"SYSTEM ACTION: {action_text}")
            elif msg_type == "warning": st.warning(f"SYSTEM ACTION: {action_text}")
            elif msg_type == "success": st.success(f"SYSTEM ACTION: {action_text}")
            elif msg_type == "info": st.info(f"SYSTEM ACTION: {action_text}")
            else: st.write(f"SYSTEM ACTION: {action_text}")

    else:
        st.info("Waiting for snapshot... Click the camera button.")