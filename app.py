import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('CNN_model.h5')  # Replace with the path to your trained model

# Function to extract frames from the input video
def extract_frames(video_path):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    while success:
        frames.append(image)
        success, image = video_capture.read()
    video_capture.release()
    return frames

# Function to preprocess frames for model prediction
def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        processed_frames.append(normalized_frame)
    return np.array(processed_frames)

# Function to predict whether the video is real or fake
def predict_video(video_path):
    frames = extract_frames(video_path)
    processed_frames = preprocess_frames(frames)
    predictions = model.predict(processed_frames)
    mean_prediction = np.mean(predictions)
    if abs(mean_prediction) < 1e-6:  # You can adjust the threshold as needed
        return "Real"
    else:
        return "Fake"

# Main function to get input from the user and make predictions
def main():
    st.title("DeepFake Detection App")

    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    # Option to upload a video file
    st.subheader("Upload Video File")
    video_file = st.file_uploader("Choose a video file", type=['mp4', 'mov'])

    

    # Perform prediction on uploaded video file
    if video_file is not None:
        st.video(video_file)
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                video_path = os.path.join("temp", video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.getvalue())
                result = predict_video(video_path)
                st.success(f"The video is predicted to be: {result}")

   

if __name__ == "__main__":
    main()
