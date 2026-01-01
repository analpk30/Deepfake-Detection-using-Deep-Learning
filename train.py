#!/usr/bin/env python3
"""Training script for Xception-based deepfake detection using RetinaFace crops.

Usage example:
    python train.py --data_root Dataset --epochs 10 --batch_size 32 --use_face_crop

This script is a scaffold to get you started; tweak augmentation, learning rate schedule,
and dataset reading for best results.
"""
import os
import argparse
import math
import tensorflow as tf
from models import build_xception
from train_utils import build_tf_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='Dataset', help='Path to dataset root')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--steps_per_epoch', type=int, default=200)
    p.add_argument('--val_steps', type=int, default=50)
    p.add_argument('--output', type=str, default='models/xception_model.h5')
    p.add_argument('--use_face_crop', action='store_true')
    p.add_argument('--max_frames', type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = build_xception(input_shape=(128, 128, 3), pretrained='imagenet', freeze_base=False)

    train_ds = build_tf_dataset(args.data_root, batch_size=args.batch_size, max_frames=args.max_frames, use_face_crop=args.use_face_crop, shuffle=True)
    val_ds = build_tf_dataset(args.data_root, batch_size=args.batch_size, max_frames=args.max_frames, use_face_crop=args.use_face_crop, shuffle=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor='loss'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    model.fit(
        train_ds,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_ds,
        validation_steps=args.val_steps,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
