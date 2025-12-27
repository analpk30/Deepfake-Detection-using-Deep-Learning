import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore


@st.cache_resource
def load_trained_model(path: str = 'CNN_model.h5'):
    """Load and cache the trained Keras model."""
    return load_model(path)

# Function to extract frames from the input video
def extract_frames(video_path, max_frames: int = 64):
    """Extract up to `max_frames` frames from `video_path` using uniform sampling.

    Returns a list of BGR frames (as read by OpenCV).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # If total is unknown or small, read all frames
    if total <= 0 or total <= max_frames:
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
    else:
        # Uniform sampling of frame indices
        step = max(1, total // max_frames)
        indices = set(range(0, total, step))
        idx = 0
        success, image = cap.read()
        while success and len(frames) < max_frames:
            if idx in indices:
                frames.append(image)
            idx += 1
            success, image = cap.read()
    cap.release()
    return frames

# Function to preprocess frames for model prediction
def preprocess_frames(frames, target_size=(128, 128)):
    """Convert BGR->RGB, resize and normalize frames.

    Returns a numpy array shaped (N, H, W, C) with dtype float32.
    """
    processed_frames = []
    for frame in frames:
        # Convert BGR (OpenCV) to RGB (models usually expect RGB)
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # If conversion fails, fallback to the original frame
            frame_rgb = frame
        resized_frame = cv2.resize(frame_rgb, target_size)
        normalized_frame = resized_frame.astype('float32') / 255.0  # Normalize pixel values
        processed_frames.append(normalized_frame)
    if len(processed_frames) == 0:
        return np.empty((0, *target_size, 3), dtype='float32')
    return np.array(processed_frames, dtype='float32')


def _create_mtcnn_detector():
    """Create and return an MTCNN detector instance.

    This is in a small helper so we only import when needed and can
    provide a clear error message if `mtcnn` is not installed.
    """
    try:
        from mtcnn.mtcnn import MTCNN
    except Exception as e:
        raise ImportError(
            "MTCNN is not installed. Install it with `pip install mtcnn` to enable face cropping."
        ) from e
    return MTCNN()


def crop_faces(frames, detector_type: str = 'mtcnn', target_size=(128, 128), padding: float = 0.2):
    """Detect largest face per frame and return cropped, resized face images.

    - `detector_type` currently supports 'mtcnn'.
    - `padding` is fraction of max(face_w, face_h) added as border.
    """
    crops = []
    if detector_type != 'mtcnn':
        raise ValueError("Only 'mtcnn' detector_type is supported currently")
    detector = _create_mtcnn_detector()

    for frame in frames:
        if frame is None:
            continue
        # MTCNN expects RGB images
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame
        results = detector.detect_faces(rgb)
        if not results:
            continue
        # pick face with highest confidence
        best = max(results, key=lambda x: x.get('confidence', 0.0))
        x, y, w, h = best['box']
        # normalize coordinates and pad
        x1 = max(0, int(x - padding * max(w, h)))
        y1 = max(0, int(y - padding * max(w, h)))
        x2 = int(x + w + padding * max(w, h))
        y2 = int(y + h + padding * max(w, h))
        # clip to frame size
        h_f, w_f = rgb.shape[:2]
        x2 = min(w_f, x2)
        y2 = min(h_f, y2)
        try:
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, target_size)
            crops.append(resized.astype('float32') / 255.0)
        except Exception:
            continue
    if len(crops) == 0:
        return np.empty((0, *target_size, 3), dtype='float32')
    return np.array(crops, dtype='float32')

# Function to predict whether the video is real or fake
def predict_video(video_path, model_path: str = 'CNN_model.h5',
                  max_frames: int = 64, batch_size: int = 32,
                  threshold: float = 0.5,
                  use_face_detection: bool = False,
                  detector_type: str = 'mtcnn',
                  detector_padding: float = 0.2):
    """Predict video-level label using frame-level model predictions.

    Returns a dict:{"label":"Real"|"Fake","score":float,"frames_used":int}
    where `score` is the mean predicted probability for the positive class.
    """
    model = load_trained_model(model_path)
    frames = extract_frames(video_path, max_frames=max_frames)
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    X = preprocess_frames(frames)
    # Optionally detect and crop faces before prediction
    if use_face_detection:
        try:
            X_faces = crop_faces(frames, detector_type=detector_type, target_size=(X.shape[1], X.shape[2]), padding=detector_padding)
        except Exception as e:
            raise RuntimeError(f"Face detection failed: {e}")
        if X_faces.shape[0] == 0:
            raise ValueError("Face detector found no faces in sampled frames")
        X = X_faces
    # Batch predictions to avoid OOM
    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        try:
            p = model.predict(batch)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        # Flatten predictions to 1D list (handles shape (N,1) or (N,))
        p = np.array(p).reshape(-1)
        preds.extend(p.tolist())
    mean_pred = float(np.mean(preds))
    label = "Fake" if mean_pred >= threshold else "Real"
    return {"label": label, "score": mean_pred, "frames_used": len(frames)}

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
        st.write("Optional: enable face detection to crop faces before prediction (MTCNN)")
        use_face = st.checkbox("Use face detection (MTCNN)", value=False)
        detector_choice = st.selectbox("Face detector", options=["mtcnn"], index=0)
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                video_path = os.path.join("temp", video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.getvalue())
                try:
                    res = predict_video(video_path, use_face_detection=use_face, detector_type=detector_choice)
                    st.success(
                        f"The video is predicted to be: {res['label']} (score={res['score']:.4f}, frames={res['frames_used']})"
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                finally:
                    # Clean up uploaded temp file
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

   

if __name__ == "__main__":
    main()
