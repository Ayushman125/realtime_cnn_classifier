import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from src.data_loader import preprocess_frame
from PIL import Image

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_my_model(model_path='models/trained_cnn_model.h5'):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'models/trained_cnn_model.h5' exists and is valid.")
        return None

def main():
    st.title("Real-Time Object Classification")
    st.sidebar.title("Options")

    model = load_my_model()
    if model is None:
        st.warning("Model not loaded. Please train the model first by running `python src/train.py`.")
        return

    class_names = [str(i) for i in range(10)] # For MNIST digits 0-9
    # Assuming model's input shape is (None, height, width, channels)
    input_height, input_width = model.input_shape[1:3]
    model_input_size = (input_height, input_width)


    option = st.sidebar.radio("Choose Input:", ("Upload Image", "Webcam"))

    if option == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Convert PIL image to OpenCV format (BGR for OpenCV processing)
            open_cv_image = np.array(image.convert("RGB")) # Ensure RGB
            open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR

            processed_frame = preprocess_frame(open_cv_image, image_size=model_input_size)
            predictions = model.predict(processed_frame, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100

            st.success(f"Prediction: **{predicted_class_name}** with **{confidence:.2f}%** confidence")

    elif option == "Webcam":
        st.write("Click 'Start Webcam' to begin real-time classification in the browser.")
        st.warning("Note: Webcam feed in Streamlit might have higher latency than the desktop app.")

        # This approach is less real-time than the desktop app due to Streamlit's refresh cycle.
        # For true real-time, the desktop inference.py is better.
        run_webcam = st.checkbox("Run Webcam")

        if run_webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam. Make sure it's connected and not in use.")
                return

            st_frame = st.image([]) # Placeholder for the video feed

            while run_webcam: # Loop continues as long as checkbox is checked
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam.")
                    break

                processed_frame = preprocess_frame(frame.copy(), image_size=model_input_size)
                predictions = model.predict(processed_frame, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class_name = class_names[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx] * 100

                text = f"Prediction: {predicted_class_name} ({confidence:.2f}%)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame in Streamlit (BGR to RGB for Streamlit's image function)
                st_frame.image(frame, channels="BGR", use_column_width=True)

                # Check if the checkbox is still checked to continue loop
                run_webcam = st.checkbox("Run Webcam", value=True, key="webcam_loop_check")
                # Small delay to prevent burning CPU (adjust as needed)
                # This delay combined with Streamlit's nature makes it not truly "real-time" like desktop app
                import time
                time.sleep(0.05)


            cap.release()
            st.success("Webcam stopped.")
        else:
             st.info("Webcam is off. Check the box to start.")

if __name__ == "__main__":
    main()