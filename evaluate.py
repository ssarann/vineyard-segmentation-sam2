#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "data/images")
MASK_ROOT = os.environ.get("MASK_ROOT", "data/masks")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt")
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/SSD_h100_final_iter030000.torch")
ZERO_SHOT = os.environ.get("ZERO_SHOT", "0") == "1" 
MODEL_CFG = "sam2.1_hiera_l.yaml"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

data = []
for root, _, files in os.walk(IMAGE_ROOT):
    for fname in files:
        if fname.upper().endswith(".JPG"):
            img_path = os.path.join(root, fname)
            rel = os.path.relpath(img_path, IMAGE_ROOT)
            mask_path = os.path.join(MASK_ROOT, os.path.splitext(rel)[0] + ".png")
            if os.path.exists(mask_path):
                data.append({"image": img_path, "annotation": mask_path})

import random
random.seed(42)
random.shuffle(data)

n = len(data)
n_val = int(n * 0.012)
n_test = int(n * 0.20)
n_train = n - n_val - n_test

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Total: {len(data)}, Test: {len(test_data)}")

USE_FULL_TEST = True
test_subset = test_data if USE_FULL_TEST else test_data[:5]
print(f"Using {len(test_subset)} images for evaluation")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
if ZERO_SHOT:
    print("Evaluating original (zero-shot) SAM 2.1 weights")
else:
    print(f"Loading fine-tuned weights from {MODEL_PATH}")
    sam2_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
sam2_model.to(device)
sam2_model.eval()

mask_generator = SAM2AutomaticMaskGenerator(
    sam2_model,
    points_per_side=46,
    points_per_batch=46,
    pred_iou_thresh=0.3,
    stability_score_thresh=0.3,
    box_nms_thresh=0.3,
    min_mask_region_area=10
)

def run_automatic_mask_generation_with_tiling(image):
 
    print("Running SAM2 automatic mask generation with tiling...")
    
    H, W = image.shape[:2]
    G = 2  # Grid size
    tile_h, tile_w = H // G, W // G
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    
    for i in range(G):
        for j in range(G):
            y0, y1 = i*tile_h, (i+1)*tile_h if i < G-1 else H
            x0, x1 = j*tile_w, (j+1)*tile_w if j < G-1 else W
            tile = image[y0:y1, x0:x1].copy()
            try:
                masks = mask_generator.generate(tile)
                print(f"Tile ({i},{j}): {len(masks)} masks")
                for k, m in enumerate(masks):
                    conf = m.get("predicted_iou", m.get("stability_score", None))
                    if conf is not None:
                        print(f" mask {k}: confidence={conf:.3f}")

                    seg = m['segmentation'].astype(np.uint8)
                    resized = cv2.resize(seg, (x1-x0, y1-y0),
                                         interpolation=cv2.INTER_NEAREST)
                    binary_mask[y0:y1, x0:x1] |= resized

            except Exception as e:
                print(f"Error in tile ({i},{j}): {e}")

    return binary_mask

def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_f1_score(pred_mask, gt_mask):
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    true_positive = np.logical_and(pred_mask, gt_mask).sum()
    false_positive = np.logical_and(pred_mask, ~gt_mask).sum()
    false_negative = np.logical_and(~pred_mask, gt_mask).sum()
    
    if true_positive == 0:
        return 1.0 if (false_positive == 0 and false_negative == 0) else 0.0
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def create_overlay_visualization(image, pred_mask, gt_mask, output_path):
    overlay = image.copy()
    
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    true_positives = np.logical_and(pred_binary, gt_binary)      
    false_positives = np.logical_and(pred_binary, ~gt_binary)    
    false_negatives = np.logical_and(~pred_binary, gt_binary)
    
    alpha = 0.4
    
    overlay[true_positives] = overlay[true_positives] * (1-alpha) + np.array([0, 255, 0]) * alpha
    
    overlay[false_positives] = overlay[false_positives] * (1-alpha) + np.array([0, 0, 255]) * alpha
    
    overlay[false_negatives] = overlay[false_negatives] * (1-alpha) + np.array([0, 165, 255]) * alpha
    
    cv2.imwrite(output_path, overlay)
    
    total_pixels = pred_binary.size
    tp_count = true_positives.sum()
    fp_count = false_positives.sum() 
    fn_count = false_negatives.sum()
    
    print(f"  TP: {tp_count:6d} ({100*tp_count/total_pixels:.2f}%) - GREEN")
    print(f"  FP: {fp_count:6d} ({100*fp_count/total_pixels:.2f}%) - RED") 
    print(f"  FN: {fn_count:6d} ({100*fn_count/total_pixels:.2f}%) - ORANGE")

results = []
for idx, sample in enumerate(test_subset):
    img_path = sample["image"]
    gt_path = sample["annotation"]

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    print(f"\n[{idx+1}/{len(test_subset)}] Processing {os.path.basename(img_path)}")
    
    pred_mask = run_automatic_mask_generation_with_tiling(image_rgb)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    iou = compute_iou(pred_mask, gt_mask)
    f1 = compute_f1_score(pred_mask, gt_mask)
    
    results.append({
        "image": os.path.basename(img_path), 
        "iou": iou,
        "f1": f1
    })

    print(f"IoU={iou:.4f}, F1={f1:.4f}")
    
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    pred_mask_path = os.path.join(RESULTS_DIR, f"pred_{img_name}.png")
    cv2.imwrite(pred_mask_path, pred_mask * 255)
    
    gt_mask_path = os.path.join(RESULTS_DIR, f"gt_{img_name}.png") 
    cv2.imwrite(gt_mask_path, gt_mask * 255)
    
  
    save_viz_indices = [0, 1, 2, 3, 4, 5, 9, 14, 19, 24, 29] 
    if idx in save_viz_indices or len(test_subset) <= 10: 
        overlay_path = os.path.join(RESULTS_DIR, f"overlay_{img_name}.png")
        create_overlay_visualization(image, pred_mask * 255, gt_mask * 255, overlay_path)
        print(f"Saved TP/FP/FN visualization: {overlay_path}")

csv_path = os.path.join(RESULTS_DIR, "results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "iou", "f1"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

avg_iou = np.mean([r["iou"] for r in results])
avg_f1 = np.mean([r["f1"] for r in results])
std_iou = np.std([r["iou"] for r in results])
std_f1 = np.std([r["f1"] for r in results])

print(f"\n DETAILED RESULTS")
print(f"Dataset: {len(test_subset)} test images")
print(f"Average IoU: {avg_iou:.4f} +- {std_iou:.4f}")
print(f"Average F1:  {avg_f1:.4f} +- {std_f1:.4f}")
print(f"IoU range: [{min([r['iou'] for r in results]):.4f}, {max([r['iou'] for r in results]):.4f}]")
print(f"F1 range:  [{min([r['f1'] for r in results]):.4f}, {max([r['f1'] for r in results]):.4f}]")
print(f"Results saved to: {csv_path}")

iou_above_05 = sum(1 for r in results if r["iou"] > 0.5)
iou_above_07 = sum(1 for r in results if r["iou"] > 0.7)
print(f"Images with IoU > 0.5: {iou_above_05}/{len(results)} ({100*iou_above_05/len(results):.1f}%)")
print(f"Images with IoU > 0.7: {iou_above_07}/{len(results)} ({100*iou_above_07/len(results):.1f}%)")

print(f"\n LEGEND ")
print(f"GREEN  = True Positives (TP)")
print(f"RED    = False Positives (FP)") 
print(f"ORANGE   = False Negatives (FN)")

if len(test_subset) <= 10:
    print("\n RESULTS")
    for r in results:
        print(f"{r['image']}: IoU={r['iou']:.4f}, F1={r['f1']:.4f}")