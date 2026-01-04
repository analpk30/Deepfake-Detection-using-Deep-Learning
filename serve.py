from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# reuse detection helper from train_utils
from train_utils import read_frames_from_video, read_frames_from_dir, detect_and_crop_faces


app = FastAPI(title="DeepFake Detection API")

# load model once on startup to avoid reloading per request
MODEL = None
@app.on_event("startup")
def load_model_on_startup():
    global MODEL
    try:
        MODEL = load_model('CNN_model.h5')
    except Exception as e:
        MODEL = None
        # keep startup lightweight; errors will be returned on predict
        print("Warning: failed to load model on startup:", e)


def preprocess_frames_simple(frames, target_size=(128, 128)):
    imgs = []
    for f in frames:
        if f is None:
            continue
        try:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = f
        try:
            resized = cv2.resize(rgb, target_size)
        except Exception:
            continue
        imgs.append(resized.astype('float32') / 255.0)
    if len(imgs) == 0:
        return np.empty((0, *target_size, 3), dtype='float32')
    return np.array(imgs, dtype='float32')


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), use_face: bool = Form(False), max_frames: int = Form(64)):
    # save uploaded file to temp
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # use preloaded model if available
        model = MODEL
        if model is None:
            # try to lazy-load if startup load failed
            try:
                model = load_model('CNN_model.h5')
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Model not available: {e}"})

        # read frames
        frames = read_frames_from_video(tmp_path, max_frames=max_frames)
        if len(frames) == 0:
            return JSONResponse(status_code=400, content={"error": "No frames extracted"})

        if use_face:
            X = detect_and_crop_faces(frames, target_size=(128, 128))
            if X.shape[0] == 0:
                # fallback to full frames
                X = preprocess_frames_simple(frames)
        else:
            X = preprocess_frames_simple(frames)

        preds = []
        batch_size = 32
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            p = model.predict(batch)
            p = np.array(p).reshape(-1)
            preds.extend(p.tolist())

        score = float(np.mean(preds)) if len(preds) else 0.0
        label = "Fake" if score >= 0.5 else "Real"
        return {"label": label, "score": score, "frames_used": len(frames)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
