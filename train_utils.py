import os
import cv2
import glob
import random
import numpy as np


def list_video_or_frame_dirs(root):
    """Yield tuples (path, label) where path is a video file or a folder of frames.

    Expects directory structure: root/<class_name>/* where class_name is 'Real' or 'Fake'.
    """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        # accept both videos and directories of frames
        for entry in os.listdir(cls_dir):
            path = os.path.join(cls_dir, entry)
            yield path, (1 if cls.lower() != 'real' else 0)


def read_frames_from_video(video_path, max_frames=16):
    """Read up to `max_frames` frames sampled uniformly from a video file.
    Returns list of BGR frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0 or total <= max_frames:
        success, img = cap.read()
        while success and len(frames) < max_frames:
            frames.append(img)
            success, img = cap.read()
    else:
        step = max(1, total // max_frames)
        indices = set(range(0, total, step))
        idx = 0
        success, img = cap.read()
        while success and len(frames) < max_frames:
            if idx in indices:
                frames.append(img)
            idx += 1
            success, img = cap.read()
    cap.release()
    return frames


def read_frames_from_dir(frames_dir, max_frames=16):
    files = sorted(glob.glob(os.path.join(frames_dir, '*')))
    if len(files) == 0:
        return []
    # sample uniformly
    total = len(files)
    if total <= max_frames:
        chosen = files
    else:
        step = max(1, total // max_frames)
        chosen = [files[i] for i in range(0, total, step)][:max_frames]
    frames = [cv2.imread(f) for f in chosen]
    frames = [f for f in frames if f is not None]
    return frames


def detect_and_crop_faces(frames, target_size=(128, 128), padding=0.2):
    """Detect largest face per frame and return cropped RGB images normalized to [0,1].

    Tries to use RetinaFace if available, otherwise falls back to MTCNN if installed.
    """
    crops = []
    # try RetinaFace
    try:
        from retinaface import RetinaFace
        detector = 'retinaface'
    except Exception:
        detector = None
    if detector is None:
        try:
            from mtcnn.mtcnn import MTCNN
            detector = 'mtcnn'
            mtcnn = MTCNN()
        except Exception:
            raise ImportError("No face detector available. Install `retinaface` or `mtcnn`.")

    for frame in frames:
        if frame is None:
            continue
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame
        if detector == 'retinaface':
            try:
                resp = RetinaFace.detect_faces(rgb)
            except Exception:
                resp = None
            if not resp:
                continue
            # resp can be dict keyed by faces
            if isinstance(resp, dict):
                vals = list(resp.values())
                best = max(vals, key=lambda r: r.get('score', 0.0))
                x, y, w, h = best['facial_area'] if 'facial_area' in best else best.get('box', (0, 0, 0, 0))
            else:
                continue
        else:
            res = mtcnn.detect_faces(rgb)
            if not res:
                continue
            best = max(res, key=lambda x: x.get('confidence', 0.0))
            x, y, w, h = best['box']

        x1 = max(0, int(x - padding * max(w, h)))
        y1 = max(0, int(y - padding * max(w, h)))
        x2 = int(x + w + padding * max(w, h))
        y2 = int(y + h + padding * max(w, h))
        h_f, w_f = rgb.shape[:2]
        x2 = min(w_f, x2)
        y2 = min(h_f, y2)
        crop = rgb[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue
        try:
            resized = cv2.resize(crop, target_size)
            crops.append(resized.astype('float32') / 255.0)
        except Exception:
            continue

    if len(crops) == 0:
        return np.empty((0, *target_size, 3), dtype='float32')
    return np.array(crops, dtype='float32')


def data_generator(dataset_root, max_frames=16, batch_size=16, shuffle=True, use_face_crop=True):
    """Yield batches (X, y) where X is shape (batch_size, H, W, C) by sampling frames and cropping faces.

    This generator yields per-frame samples (not per-video sequences). For per-video training a different approach
    (aggregating frames per sample) would be needed.
    """
    entries = list(list_video_or_frame_dirs(dataset_root))
    if shuffle:
        random.shuffle(entries)

    X_batch = []
    y_batch = []
    while True:
        for path, label in entries:
            # decide if path is a file (video) or directory of frames
            if os.path.isdir(path):
                frames = read_frames_from_dir(path, max_frames=max_frames)
            else:
                frames = read_frames_from_video(path, max_frames=max_frames)

            if use_face_crop:
                crops = detect_and_crop_faces(frames, target_size=(128, 128))
            else:
                # fallback: resize frames directly
                crops = []
                for f in frames:
                    try:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    except Exception:
                        rgb = f
                    try:
                        resized = cv2.resize(rgb, (128, 128)).astype('float32') / 255.0
                        crops.append(resized)
                    except Exception:
                        continue
                crops = np.array(crops, dtype='float32') if len(crops) else np.empty((0, 128, 128, 3), dtype='float32')

            # yield one training example per crop
            for c in crops:
                X_batch.append(c)
                y_batch.append(label)
                if len(X_batch) >= batch_size:
                    yield np.stack(X_batch, axis=0), np.array(y_batch, dtype='float32')
                    X_batch = []
                    y_batch = []
