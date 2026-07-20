#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import random
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from tensorflow.keras.models import load_model  # type: ignore

from constants import (
    CLASS_REAL_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_DETECTOR_PADDING,
    DEFAULT_DETECTOR_TYPE,
    DEFAULT_EVAL_AGG_METHOD,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_GRADCAM_OUT,
    DEFAULT_EVAL_MAX_FRAMES,
    DEFAULT_EVAL_OUT_PREFIX,
    DEFAULT_EVAL_THRESHOLD,
    DEFAULT_EVAL_TOP_K,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_PRED_THRESHOLD,
    DEFAULT_STREAMLIT_TEMP_DIR,
    DEFAULT_TARGET_SIZE,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_MAX_FRAMES,
    DEFAULT_TRAIN_OUTPUT,
    DEFAULT_TRAIN_STEPS_PER_EPOCH,
    DEFAULT_TRAIN_VAL_STEPS,
    FACE_DETECTOR_MTCNN,
    FACE_DETECTOR_RETINAFACE,
    LABEL_FAKE,
    LABEL_REAL,
    SERVE_APP_TITLE,
    SERVE_HOST,
    SERVE_PORT,
)


@st.cache_resource
def load_trained_model(path: str = DEFAULT_MODEL_PATH):
    return load_model(path)


def list_video_or_frame_dirs(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        for entry in os.listdir(cls_dir):
            path = os.path.join(cls_dir, entry)
            yield path, (1 if cls.lower() != CLASS_REAL_NAME else 0)


def read_frames_from_video(video_path, max_frames=DEFAULT_TRAIN_MAX_FRAMES):
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


def read_frames_from_dir(frames_dir, max_frames=DEFAULT_TRAIN_MAX_FRAMES):
    files = sorted(glob.glob(os.path.join(frames_dir, "*")))
    if len(files) == 0:
        return []
    total = len(files)
    if total <= max_frames:
        chosen = files
    else:
        step = max(1, total // max_frames)
        chosen = [files[i] for i in range(0, total, step)][:max_frames]
    frames = [cv2.imread(f) for f in chosen]
    return [f for f in frames if f is not None]


def extract_frames(video_path, max_frames=DEFAULT_EVAL_MAX_FRAMES):
    return read_frames_from_video(video_path, max_frames=max_frames)


def preprocess_frames(frames, target_size=DEFAULT_TARGET_SIZE):
    processed_frames = []
    for frame in frames:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = frame
        resized_frame = cv2.resize(frame_rgb, target_size)
        normalized_frame = resized_frame.astype("float32") / 255.0
        processed_frames.append(normalized_frame)
    if len(processed_frames) == 0:
        return np.empty((0, *target_size, 3), dtype="float32")
    return np.array(processed_frames, dtype="float32")


def preprocess_frames_simple(frames, target_size=DEFAULT_TARGET_SIZE):
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
        imgs.append(resized.astype("float32") / 255.0)
    if len(imgs) == 0:
        return np.empty((0, *target_size, 3), dtype="float32")
    return np.array(imgs, dtype="float32")


def _create_mtcnn_detector():
    try:
        from mtcnn.mtcnn import MTCNN
    except Exception as e:
        raise ImportError(
            "MTCNN is not installed. Install it with `pip install mtcnn` to enable face cropping."
        ) from e
    return MTCNN()


def detect_and_crop_faces(
    frames, target_size=DEFAULT_TARGET_SIZE, padding=DEFAULT_DETECTOR_PADDING
):
    crops = []
    try:
        from retinaface import RetinaFace

        detector = FACE_DETECTOR_RETINAFACE
    except Exception:
        detector = None
    if detector is None:
        try:
            mtcnn = _create_mtcnn_detector()
            detector = FACE_DETECTOR_MTCNN
        except Exception as e:
            raise ImportError(
                "No face detector available. Install `retinaface` or `mtcnn`."
            ) from e

    for frame in frames:
        if frame is None:
            continue
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame
        if detector == FACE_DETECTOR_RETINAFACE:
            try:
                resp = RetinaFace.detect_faces(rgb)
            except Exception:
                resp = None
            if not resp:
                continue
            if isinstance(resp, dict):
                vals = list(resp.values())
                best = max(vals, key=lambda r: r.get("score", 0.0))
                x, y, w, h = (
                    best["facial_area"]
                    if "facial_area" in best
                    else best.get("box", (0, 0, 0, 0))
                )
            else:
                continue
        else:
            res = mtcnn.detect_faces(rgb)
            if not res:
                continue
            best = max(res, key=lambda x: x.get("confidence", 0.0))
            x, y, w, h = best["box"]

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
            crops.append(resized.astype("float32") / 255.0)
        except Exception:
            continue

    if len(crops) == 0:
        return np.empty((0, *target_size, 3), dtype="float32")
    return np.array(crops, dtype="float32")


def crop_faces(
    frames,
    detector_type=DEFAULT_DETECTOR_TYPE,
    target_size=DEFAULT_TARGET_SIZE,
    padding=DEFAULT_DETECTOR_PADDING,
):
    if detector_type != FACE_DETECTOR_MTCNN:
        raise ValueError("Only 'mtcnn' detector_type is supported currently")
    detector = _create_mtcnn_detector()
    crops = []

    for frame in frames:
        if frame is None:
            continue
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame
        results = detector.detect_faces(rgb)
        if not results:
            continue
        best = max(results, key=lambda x: x.get("confidence", 0.0))
        x, y, w, h = best["box"]
        x1 = max(0, int(x - padding * max(w, h)))
        y1 = max(0, int(y - padding * max(w, h)))
        x2 = int(x + w + padding * max(w, h))
        y2 = int(y + h + padding * max(w, h))
        h_f, w_f = rgb.shape[:2]
        x2 = min(w_f, x2)
        y2 = min(h_f, y2)
        try:
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, target_size)
            crops.append(resized.astype("float32") / 255.0)
        except Exception:
            continue
    if len(crops) == 0:
        return np.empty((0, *target_size, 3), dtype="float32")
    return np.array(crops, dtype="float32")


def build_xception(
    input_shape=DEFAULT_INPUT_SHAPE, pretrained="imagenet", freeze_base=False
):
    base = tf.keras.applications.Xception(
        include_top=False, weights=pretrained, input_shape=input_shape, pooling="avg"
    )
    if freeze_base:
        base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base(x, training=not freeze_base)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def data_generator(
    dataset_root,
    max_frames=DEFAULT_TRAIN_MAX_FRAMES,
    batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    shuffle=True,
    use_face_crop=True,
):
    entries = list(list_video_or_frame_dirs(dataset_root))
    if shuffle:
        random.shuffle(entries)

    X_batch = []
    y_batch = []
    while True:
        for path, label in entries:
            if os.path.isdir(path):
                frames = read_frames_from_dir(path, max_frames=max_frames)
            else:
                frames = read_frames_from_video(path, max_frames=max_frames)

            if use_face_crop:
                crops = detect_and_crop_faces(frames, target_size=DEFAULT_TARGET_SIZE)
            else:
                crops = []
                for f in frames:
                    try:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    except Exception:
                        rgb = f
                    try:
                        resized = (
                            cv2.resize(rgb, DEFAULT_TARGET_SIZE).astype("float32")
                            / 255.0
                        )
                        crops.append(resized)
                    except Exception:
                        continue
                crops = (
                    np.array(crops, dtype="float32")
                    if len(crops)
                    else np.empty((0, *DEFAULT_TARGET_SIZE, 3), dtype="float32")
                )

            for c in crops:
                X_batch.append(c)
                y_batch.append(label)
                if len(X_batch) >= batch_size:
                    yield np.stack(X_batch, axis=0), np.array(y_batch, dtype="float32")
                    X_batch = []
                    y_batch = []


def sample_generator(
    dataset_root,
    max_frames=DEFAULT_TRAIN_MAX_FRAMES,
    shuffle=True,
    use_face_crop=True,
):
    entries = list(list_video_or_frame_dirs(dataset_root))
    if shuffle:
        random.shuffle(entries)

    while True:
        for path, label in entries:
            if os.path.isdir(path):
                frames = read_frames_from_dir(path, max_frames=max_frames)
            else:
                frames = read_frames_from_video(path, max_frames=max_frames)

            if use_face_crop:
                try:
                    crops = detect_and_crop_faces(
                        frames, target_size=DEFAULT_TARGET_SIZE
                    )
                except Exception:
                    crops = np.empty((0, *DEFAULT_TARGET_SIZE, 3), dtype="float32")
            else:
                crops_list = []
                for f in frames:
                    try:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    except Exception:
                        rgb = f
                    try:
                        resized = (
                            cv2.resize(rgb, DEFAULT_TARGET_SIZE).astype("float32")
                            / 255.0
                        )
                        crops_list.append(resized)
                    except Exception:
                        continue
                crops = (
                    np.array(crops_list, dtype="float32")
                    if len(crops_list)
                    else np.empty((0, *DEFAULT_TARGET_SIZE, 3), dtype="float32")
                )

            for c in crops:
                yield c, np.int32(label)


def build_tf_dataset(
    dataset_root,
    batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    max_frames=DEFAULT_TRAIN_MAX_FRAMES,
    shuffle=True,
    use_face_crop=True,
    buffer_size=1024,
):
    output_signature = (
        tf.TensorSpec(shape=DEFAULT_INPUT_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: sample_generator(
            dataset_root,
            max_frames=max_frames,
            shuffle=shuffle,
            use_face_crop=use_face_crop,
        ),
        output_signature=output_signature,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def predict_video(
    video_path,
    model_path=DEFAULT_MODEL_PATH,
    max_frames=DEFAULT_EVAL_MAX_FRAMES,
    batch_size=DEFAULT_EVAL_BATCH_SIZE,
    threshold=DEFAULT_PRED_THRESHOLD,
    use_face_detection=False,
    detector_type=DEFAULT_DETECTOR_TYPE,
    detector_padding=DEFAULT_DETECTOR_PADDING,
):
    model = load_trained_model(model_path)
    frames = extract_frames(video_path, max_frames=max_frames)
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    X = preprocess_frames(frames)
    if use_face_detection:
        try:
            X_faces = crop_faces(
                frames,
                detector_type=detector_type,
                target_size=(X.shape[1], X.shape[2]),
                padding=detector_padding,
            )
        except Exception as e:
            raise RuntimeError(f"Face detection failed: {e}") from e
        if X_faces.shape[0] == 0:
            raise ValueError("Face detector found no faces in sampled frames")
        X = X_faces
    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        try:
            p = model.predict(batch)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}") from e
        p = np.array(p).reshape(-1)
        preds.extend(p.tolist())
    mean_pred = float(np.mean(preds))
    label = LABEL_FAKE if mean_pred >= threshold else LABEL_REAL
    return {"label": label, "score": mean_pred, "frames_used": len(frames)}


def score_entry(
    model,
    path,
    label,
    max_frames=DEFAULT_EVAL_MAX_FRAMES,
    batch_size=DEFAULT_EVAL_BATCH_SIZE,
    use_crop=False,
):
    if os.path.isdir(path):
        frames = read_frames_from_dir(path, max_frames=max_frames)
    else:
        frames = read_frames_from_video(path, max_frames=max_frames)

    if len(frames) == 0:
        return None

    if use_crop:
        X = detect_and_crop_faces(frames, target_size=DEFAULT_TARGET_SIZE)
        if X.shape[0] == 0:
            X = preprocess_frames_simple(frames, target_size=DEFAULT_TARGET_SIZE)
    else:
        X = preprocess_frames_simple(frames, target_size=DEFAULT_TARGET_SIZE)

    if X.shape[0] == 0:
        return None

    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        p = model.predict(batch)
        p = np.array(p).reshape(-1)
        preds.extend(p.tolist())

    mean_score = float(np.mean(preds))
    return {
        "path": path,
        "label": int(label),
        "score": mean_score,
        "frame_scores": preds,
        "num_frames": len(preds),
    }


def aggregate_scores(
    frame_scores, method=DEFAULT_EVAL_AGG_METHOD, threshold=DEFAULT_EVAL_THRESHOLD
):
    if len(frame_scores) == 0:
        return None, None
    arr = np.array(frame_scores)
    if method == "mean":
        score = float(np.mean(arr))
    elif method == "median":
        score = float(np.median(arr))
    elif method == "majority":
        score = float(np.mean(arr >= threshold))
    else:
        raise ValueError("Unknown aggregation method")
    label = 1 if score >= threshold else 0
    return score, label


def write_results_csv(results, fname):
    if len(results) == 0:
        print(f"No results for {fname}")
        return None
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "num_frames"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "path": r["path"],
                    "label": r["label"],
                    "score": r["score"],
                    "num_frames": r.get("num_frames", ""),
                }
            )
    return fname


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    img_tensor = tf.expand_dims(img_array, axis=0)
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM")

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = 0
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))


def save_gradcam_overlay(orig_img, heatmap, out_path, alpha=0.4):
    img = (orig_img * 255).astype("uint8")
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    from PIL import Image

    Image.fromarray(overlay).save(out_path)


def evaluate(
    data_root,
    model_path,
    max_frames=DEFAULT_EVAL_MAX_FRAMES,
    batch_size=DEFAULT_EVAL_BATCH_SIZE,
    out_prefix=DEFAULT_EVAL_OUT_PREFIX,
    agg_method=DEFAULT_EVAL_AGG_METHOD,
    threshold=DEFAULT_EVAL_THRESHOLD,
    save_gradcam=False,
    top_k=DEFAULT_EVAL_TOP_K,
    gradcam_out=DEFAULT_EVAL_GRADCAM_OUT,
):
    model = load_model(model_path)
    entries = list(list_video_or_frame_dirs(data_root))
    results = []
    for path, label in entries:
        r = score_entry(
            model,
            path,
            label,
            max_frames=max_frames,
            batch_size=batch_size,
            use_crop=False,
        )
        r_crop = score_entry(
            model,
            path,
            label,
            max_frames=max_frames,
            batch_size=batch_size,
            use_crop=True,
        )
        if r is None and r_crop is None:
            continue
        for mode_name, entry in [("full", r), ("crop", r_crop)]:
            if entry is None:
                continue
            frame_scores = entry["frame_scores"]
            agg_score, _ = aggregate_scores(
                frame_scores, method=agg_method, threshold=threshold
            )
            record = {
                "path": entry["path"],
                "label": entry["label"],
                "score": agg_score,
                "mode": mode_name,
                "num_frames": entry.get("num_frames", 0),
            }
            results.append(record)
            if save_gradcam:
                scores_arr = np.array(frame_scores)
                distances = np.abs(scores_arr - threshold)
                idxs = np.argsort(-distances)[:top_k]
                if os.path.isdir(path):
                    frames = read_frames_from_dir(path, max_frames=max_frames)
                else:
                    frames = read_frames_from_video(path, max_frames=max_frames)
                for rank, idx in enumerate(idxs):
                    if idx < 0 or idx >= len(frames):
                        continue
                    f = frames[idx]
                    if f is None:
                        continue
                    try:
                        rgb = (
                            cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                        )
                    except Exception:
                        rgb = f.astype("float32") / 255.0
                    try:
                        heatmap = make_gradcam_heatmap(rgb, model)
                        out_dir = os.path.join(gradcam_out, mode_name)
                        os.makedirs(out_dir, exist_ok=True)
                        base = os.path.basename(path).replace(os.path.sep, "_")
                        out_path = os.path.join(
                            out_dir, f"{base}_frame{idx}_rank{rank}.png"
                        )
                        save_gradcam_overlay(rgb, heatmap, out_path)
                    except Exception as e:
                        print(f"Grad-CAM failed for {path} frame {idx}: {e}")

    def compute_metrics_for_mode(mode):
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            return None
        y_true = [r["label"] for r in mode_results]
        y_score = [r["score"] for r in mode_results]
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        except Exception:
            precision, recall, ap = None, None, None
        y_pred = [1 if s >= threshold else 0 for s in y_score]
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            "auc": auc,
            "precision_recall": (precision, recall),
            "ap": ap,
            "report": report,
            "results": mode_results,
        }

    metrics_full = compute_metrics_for_mode("full")
    metrics_crop = compute_metrics_for_mode("crop")
    write_results_csv(
        [r for r in results if r["mode"] == "full"], out_prefix + "_full.csv"
    )
    write_results_csv(
        [r for r in results if r["mode"] == "crop"], out_prefix + "_crop.csv"
    )

    print("Evaluation results:")
    if metrics_full is not None:
        print(f"Full-frame AUC: {metrics_full['auc']}, AP: {metrics_full['ap']}")
        if metrics_full["precision_recall"][0] is not None:
            plt.figure()
            plt.step(
                metrics_full["precision_recall"][1],
                metrics_full["precision_recall"][0],
                where="post",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Full-frame PR curve")
            plt.savefig(out_prefix + "_full_pr.png")
            plt.close()
        print("Full-frame classification report:")
        print(metrics_full["report"])
    if metrics_crop is not None:
        print(f"Face-cropped AUC: {metrics_crop['auc']}, AP: {metrics_crop['ap']}")
        if metrics_crop["precision_recall"][0] is not None:
            plt.figure()
            plt.step(
                metrics_crop["precision_recall"][1],
                metrics_crop["precision_recall"][0],
                where="post",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Face-cropped PR curve")
            plt.savefig(out_prefix + "_crop_pr.png")
            plt.close()
        print("Face-cropped classification report:")
        print(metrics_crop["report"])


def parse_train_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Path to dataset root"
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    p.add_argument("--steps_per_epoch", type=int, default=DEFAULT_TRAIN_STEPS_PER_EPOCH)
    p.add_argument("--val_steps", type=int, default=DEFAULT_TRAIN_VAL_STEPS)
    p.add_argument("--output", type=str, default=DEFAULT_TRAIN_OUTPUT)
    p.add_argument("--use_face_crop", action="store_true")
    p.add_argument("--max_frames", type=int, default=DEFAULT_TRAIN_MAX_FRAMES)
    return p.parse_args()


def run_training():
    args = parse_train_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = build_xception(
            input_shape=DEFAULT_INPUT_SHAPE, pretrained="imagenet", freeze_base=False
        )

    train_ds = build_tf_dataset(
        args.data_root,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        use_face_crop=args.use_face_crop,
        shuffle=True,
    )
    val_ds = build_tf_dataset(
        args.data_root,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        use_face_crop=args.use_face_crop,
        shuffle=False,
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output, save_best_only=True, monitor="loss"
        ),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]
    model.fit(
        train_ds,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_ds,
        validation_steps=args.val_steps,
        callbacks=callbacks,
    )


def parse_eval_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--max_frames", type=int, default=DEFAULT_EVAL_MAX_FRAMES)
    p.add_argument("--batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--out_prefix", type=str, default=DEFAULT_EVAL_OUT_PREFIX)
    return p.parse_args()


def run_evaluation():
    args = parse_eval_args()
    evaluate(
        args.data_root,
        args.model,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        out_prefix=args.out_prefix,
    )


api = FastAPI(title=SERVE_APP_TITLE)
MODEL = None


@api.on_event("startup")
def load_model_on_startup():
    global MODEL
    try:
        MODEL = load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        MODEL = None
        print("Warning: failed to load model on startup:", e)


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_face: bool = Form(False),
    max_frames: int = Form(DEFAULT_EVAL_MAX_FRAMES),
):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        model = MODEL
        if model is None:
            try:
                model = load_model(DEFAULT_MODEL_PATH)
            except Exception as e:
                return JSONResponse(
                    status_code=500, content={"error": f"Model not available: {e}"}
                )
        frames = read_frames_from_video(tmp_path, max_frames=max_frames)
        if len(frames) == 0:
            return JSONResponse(
                status_code=400, content={"error": "No frames extracted"}
            )
        if use_face:
            X = detect_and_crop_faces(frames, target_size=DEFAULT_TARGET_SIZE)
            if X.shape[0] == 0:
                X = preprocess_frames_simple(frames)
        else:
            X = preprocess_frames_simple(frames)
        preds = []
        batch_size = DEFAULT_EVAL_BATCH_SIZE
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            p = model.predict(batch)
            p = np.array(p).reshape(-1)
            preds.extend(p.tolist())

        score = float(np.mean(preds)) if len(preds) else 0.0
        label = LABEL_FAKE if score >= DEFAULT_PRED_THRESHOLD else LABEL_REAL
        return {"label": label, "score": score, "frames_used": len(frames)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def streamlit_main():
    st.title("DeepFake Detection App")
    os.makedirs(DEFAULT_STREAMLIT_TEMP_DIR, exist_ok=True)
    st.subheader("Upload Video File")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

    if video_file is not None:
        st.video(video_file)
        st.write(
            "Optional: enable face detection to crop faces before prediction (MTCNN)"
        )
        use_face = st.checkbox("Use face detection (MTCNN)", value=False)
        detector_choice = st.selectbox(
            "Face detector", options=[FACE_DETECTOR_MTCNN], index=0
        )
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                video_path = os.path.join(DEFAULT_STREAMLIT_TEMP_DIR, video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.getvalue())
                try:
                    res = predict_video(
                        video_path,
                        use_face_detection=use_face,
                        detector_type=detector_choice,
                    )
                    st.success(
                        f"The video is predicted to be: {res['label']} (score={res['score']:.4f}, frames={res['frames_used']})"
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                finally:
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass


def serve_main():
    uvicorn.run(api, host=SERVE_HOST, port=SERVE_PORT)


if __name__ == "__main__":
    streamlit_main()
