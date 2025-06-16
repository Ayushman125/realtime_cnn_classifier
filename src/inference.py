import cv2
import numpy as np
import tensorflow as tf
from src.data_loader import preprocess_frame

def real_time_inference(model_path='models/trained_cnn_model.h5', class_names=None):
    """
    Performs real-time object classification using webcam input.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)] # For MNIST digits 0-9

    try:
        model = tf.keras.models.load_model(model_path)
        # Assuming model's input shape is (None, height, width, channels)
        input_height, input_width = model.input_shape[1:3]
        input_shape = (input_height, input_width)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    cap = cv2.VideoCapture(0) # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure it's connected and not in use by another application.")
        return

    print("Press 'q' to quit the real-time inference.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break

        # Preprocess the frame for the model
        # Ensure frame is copy before modifying in preprocess_frame if needed elsewhere
        processed_frame = preprocess_frame(frame.copy(), image_size=input_shape)

        # Make prediction
        predictions = model.predict(processed_frame, verbose=0) # verbose=0 to suppress Keras output
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100

        # Display prediction on the frame
        text = f"Prediction: {predicted_class_name} ({confidence:.2f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-Time Object Classification', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference stopped.")

if __name__ == "__main__":
    real_time_inference()