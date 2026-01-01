#!/usr/bin/env python3
"""Evaluate a trained model on a dataset at video level.

Computes video-level mean prediction and reports AUC. Compares two modes:
- full-frame (resize whole frame)
- face-cropped (use detect_and_crop_faces from train_utils)

Usage:
    python eval.py --data_root Dataset --model CNN_model.h5 --max_frames 64
"""
import os
import argparse
import csv
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf

from train_utils import list_video_or_frame_dirs, read_frames_from_video, read_frames_from_dir, detect_and_crop_faces
import cv2


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


def score_entry(model, path, label, max_frames=64, batch_size=32, use_crop=False):
    # read frames
    if os.path.isdir(path):
        frames = read_frames_from_dir(path, max_frames=max_frames)
    else:
        frames = read_frames_from_video(path, max_frames=max_frames)

    if len(frames) == 0:
        return None

    if use_crop:
        X = detect_and_crop_faces(frames, target_size=(128, 128))
        if X.shape[0] == 0:
            # fallback to full-frame
            X = preprocess_frames_simple(frames, target_size=(128, 128))
    else:
        X = preprocess_frames_simple(frames, target_size=(128, 128))

    if X.shape[0] == 0:
        return None

    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        p = model.predict(batch)
        p = np.array(p).reshape(-1)
        preds.extend(p.tolist())

    mean_score = float(np.mean(preds))
    return {'path': path, 'label': int(label), 'score': mean_score, 'frame_scores': preds, 'num_frames': len(preds)}


def aggregate_scores(frame_scores, method='mean', threshold=0.5):
    """Aggregate frame-level scores into a single video-level score/label.

    method: 'mean'|'median'|'majority'
    For 'majority', returns fraction of frames >= threshold as score.
    """
    if len(frame_scores) == 0:
        return None, None
    arr = np.array(frame_scores)
    if method == 'mean':
        score = float(np.mean(arr))
    elif method == 'median':
        score = float(np.median(arr))
    elif method == 'majority':
        frac = float(np.mean(arr >= threshold))
        score = frac
    else:
        raise ValueError('Unknown aggregation method')
    label = 1 if score >= threshold else 0
    return score, label


def write_results_csv(results, fname):
    if len(results) == 0:
        print(f"No results for {fname}")
        return None
    # write CSV with path,label,score,num_frames
    with open(fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'score', 'num_frames'])
        writer.writeheader()
        for r in results:
            writer.writerow({'path': r['path'], 'label': r['label'], 'score': r['score'], 'num_frames': r.get('num_frames', '')})


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    # img_array: shape (H,W,3) float32 scaled [0,1]
    img_tensor = tf.expand_dims(img_array, axis=0)
    # find last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError('No convolutional layer found for Grad-CAM')

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

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
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    return heatmap


def save_gradcam_overlay(orig_img, heatmap, out_path, alpha=0.4):
    # orig_img expected RGB [0,1] float32
    img = (orig_img * 255).astype('uint8')
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    # save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    from PIL import Image
    Image.fromarray(overlay).save(out_path)


def evaluate(data_root, model_path, max_frames=64, batch_size=32, out_prefix='scores', agg_method='mean', threshold=0.5, save_gradcam=False, top_k=3, gradcam_out='gradcam'):
    model = load_model(model_path)
    entries = list(list_video_or_frame_dirs(data_root))
    results = []
    per_video_preds = []
    per_video_labels = []
    for path, label in entries:
        r = score_entry(model, path, label, max_frames=max_frames, batch_size=batch_size, use_crop=False)
        r_crop = score_entry(model, path, label, max_frames=max_frames, batch_size=batch_size, use_crop=True)
        if r is None and r_crop is None:
            continue
        # prepare both modes
        for mode_name, entry in [('full', r), ('crop', r_crop)]:
            if entry is None:
                continue
            frame_scores = entry['frame_scores']
            agg_score, pred_label = aggregate_scores(frame_scores, method=agg_method, threshold=threshold)
            record = {'path': entry['path'], 'label': entry['label'], 'score': agg_score, 'mode': mode_name, 'num_frames': entry.get('num_frames', 0)}
            results.append(record)
            # collect for metrics (only include selected mode later)
            # Save gradcam overlays if requested (use frame-level highest confidence frames)
            if save_gradcam:
                # choose top_k frames by distance from threshold
                scores_arr = np.array(frame_scores)
                distances = np.abs(scores_arr - threshold)
                idxs = np.argsort(-distances)[:top_k]
                # read frames again to get images
                if os.path.isdir(path):
                    frames = read_frames_from_dir(path, max_frames=max_frames)
                else:
                    frames = read_frames_from_video(path, max_frames=max_frames)
                # ensure same ordering as used for frames
                for rank, idx in enumerate(idxs):
                    if idx < 0 or idx >= len(frames):
                        continue
                    f = frames[idx]
                    if f is None:
                        continue
                    try:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
                    except Exception:
                        rgb = f.astype('float32') / 255.0
                    # compute heatmap
                    try:
                        heatmap = make_gradcam_heatmap(rgb, model)
                        out_dir = os.path.join(gradcam_out, mode_name)
                        os.makedirs(out_dir, exist_ok=True)
                        base = os.path.basename(path).replace(os.path.sep, '_')
                        out_path = os.path.join(out_dir, f"{base}_frame{idx}_rank{rank}.png")
                        save_gradcam_overlay(rgb, heatmap, out_path)
                    except Exception as e:
                        # continue on errors
                        print(f"Grad-CAM failed for {path} frame {idx}: {e}")

    # Now compute metrics separately for full and crop modes
    def compute_metrics_for_mode(mode):
        mode_results = [r for r in results if r['mode'] == mode]
        if not mode_results:
            return None
        y_true = [r['label'] for r in mode_results]
        y_score = [r['score'] for r in mode_results]
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
        # PR curve and average precision
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        except Exception:
            precision, recall, ap = None, None, None
        # aggregated predicted labels
        y_pred = [1 if s >= threshold else 0 for s in y_score]
        report = classification_report(y_true, y_pred, output_dict=True)
        return {'auc': auc, 'precision_recall': (precision, recall), 'ap': ap, 'report': report, 'results': mode_results}

    metrics_full = compute_metrics_for_mode('full')
    metrics_crop = compute_metrics_for_mode('crop')

    # write CSVs
    write_results_csv([r for r in results if r['mode'] == 'full'], out_prefix + '_full.csv')
    write_results_csv([r for r in results if r['mode'] == 'crop'], out_prefix + '_crop.csv')

    print('Evaluation results:')
    if metrics_full is not None:
        print(f"Full-frame AUC: {metrics_full['auc']}, AP: {metrics_full['ap']}")
        # save PR curve
        if metrics_full['precision_recall'][0] is not None:
            plt.figure()
            plt.step(metrics_full['precision_recall'][1], metrics_full['precision_recall'][0], where='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Full-frame PR curve')
            plt.savefig(out_prefix + '_full_pr.png')
            plt.close()
        print('Full-frame classification report:')
        print(metrics_full['report'])
    if metrics_crop is not None:
        print(f"Face-cropped AUC: {metrics_crop['auc']}, AP: {metrics_crop['ap']}")
        if metrics_crop['precision_recall'][0] is not None:
            plt.figure()
            plt.step(metrics_crop['precision_recall'][1], metrics_crop['precision_recall'][0], where='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Face-cropped PR curve')
            plt.savefig(out_prefix + '_crop_pr.png')
            plt.close()
        print('Face-cropped classification report:')
        print(metrics_crop['report'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='Dataset')
    p.add_argument('--model', type=str, default='CNN_model.h5')
    p.add_argument('--max_frames', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--out_prefix', type=str, default='scores')
    args = p.parse_args()

    evaluate(args.data_root, args.model, max_frames=args.max_frames, batch_size=args.batch_size, out_prefix=args.out_prefix)


if __name__ == '__main__':
    main()
