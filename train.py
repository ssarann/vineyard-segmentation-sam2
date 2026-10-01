import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
import csv
import random
import sys
import json
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch.nn.functional as F
import importlib.resources as imp
import sam2

torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.allow_tf32 = True  
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

COMPILE_MODEL = False
torch.set_float32_matmul_precision('medium')  

BATCH_SIZE = 10
PREFETCH_FACTOR = 4  

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "data/images")
MASK_ROOT = os.environ.get("MASK_ROOT", "data/masks")
sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability()}")
    major, minor = torch.cuda.get_device_capability()
    if major >= 9:
        print("H100 detected - enabling advanced optimizations")
        H100_OPTIMIZATIONS = True
    else:
        print("Non-H100 GPU detected")
        H100_OPTIMIZATIONS = False

print(f"  Image root exists: {os.path.exists(IMAGE_ROOT)}")
print(f"  Mask root exists: {os.path.exists(MASK_ROOT)}")
print(f"  SAM2 checkpoint exists: {os.path.exists(sam2_checkpoint)}")

if not os.path.exists(sam2_checkpoint):
    print("ERROR: SAM2 checkpoint not found!")
    sys.exit(1)

print("Gathering image/annotation pairs...")
data = []
for root, _, files in os.walk(IMAGE_ROOT):
    for fname in files:
        if fname.upper().endswith(".JPG"):
            img_path = os.path.join(root, fname)
            rel = os.path.relpath(img_path, IMAGE_ROOT)
            mask_path = os.path.join(MASK_ROOT, os.path.splitext(rel)[0] + ".png")
            data.append({"image": img_path, "annotation": mask_path})

print(f"Found {len(data)} image files, filtering for existing masks...")
data = [d for d in data if os.path.exists(d["annotation"])]
print(f"Valid image-mask pairs: {len(data)}")

if len(data) == 0:
    raise ValueError("No valid image-mask pairs found!")

random.seed(42)
data_shuffled = data.copy()
random.shuffle(data_shuffled)

n = len(data_shuffled)
n_val  = int(n * 0.012)
n_test = int(n * 0.20)
n_train = n - n_val - n_test 

train_data = data_shuffled[:n_train]
val_data   = data_shuffled[n_train:n_train + n_val]
test_data  = data_shuffled[n_train + n_val:]

fixed_val_data = list(val_data) 
print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
run_id = f"sling_h100_iter_{job_id}_{random.randint(1000,9999)}"
print(f"H100-Optimized Iteration-based Run ID: {run_id}")

results_dir = f"results_{run_id}"
os.makedirs(results_dir, exist_ok=True)

aug_counter = 0

def generate_blind_prompts(h, w, num_points_range=(16, 156)):
    num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
    if np.random.random() < 0.7:
        points, labels = [], []
        for _ in range(num_points):
            x = np.random.randint(50, w - 50)
            y = np.random.randint(50, h - 50)
            points.append([x, y])
            labels.append(1 if np.random.random() < 0.6 else 0)
        return points, labels
    else:
        grid_size = int(np.sqrt(num_points))
        points, labels = [], []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= num_points:
                    break
                x = int((j + 0.5) * w / grid_size) + np.random.randint(-20, 20)
                y = int((i + 0.5) * h / grid_size) + np.random.randint(-20, 20)
                x = max(50, min(w-50, x))
                y = max(50, min(h-50, y))
                points.append([x, y])
                labels.append(1 if np.random.random() < 0.7 else 0)
        return points, labels

def get_random_batch(train_data, batch_size=BATCH_SIZE):
    global aug_counter
    
    batch_samples = random.sample(train_data, min(batch_size, len(train_data)))
    
    images, masks, points, labels = [], [], [], []
    
    for sample in batch_samples:
        if not os.path.exists(sample["image"]) or not os.path.exists(sample["annotation"]):
            continue
            
        try:
            Img = cv2.imread(sample["image"])[..., ::-1]
            ann_map = cv2.imread(sample["annotation"], cv2.IMREAD_GRAYSCALE)
            if Img is None or ann_map is None:
                continue
                
            h, w = Img.shape[:2]
            if h < 1024 or w < 1024:
                continue
            
            y0 = np.random.randint(0, h - 1024 + 1)
            x0 = np.random.randint(0, w - 1024 + 1)
            Img = Img[y0:y0 + 1024, x0:x0 + 1024]
            ann_map = ann_map[y0:y0 + 1024, x0:x0 + 1024]

            if aug_counter % 40 == 0: 
                Img = cv2.flip(Img, 1)
                ann_map = cv2.flip(ann_map, 1)
            if aug_counter % 50 == 0:  
                Img = cv2.GaussianBlur(Img, (3, 3), 0)
            aug_counter += 1

            pts, lbs = generate_blind_prompts(1024, 1024)
            
            if not pts:
                continue
                
            images.append(Img.copy())
            masks.append((ann_map > 127).astype(np.float32))
            points.append(pts)
            labels.append(lbs)
            
        except Exception as e:
            print(f"Warning: Error loading {sample['image']}: {e}")
            continue
            
    return images, np.array(masks), points, labels

print("Initializing SAM2 model with H100")
sam2_model = build_sam2("sam2.1_hiera_l.yaml", sam2_checkpoint, device="cuda")

if H100_OPTIMIZATIONS:
    sam2_model = sam2_model.to(memory_format=torch.channels_last)
    print("Model converted to channels_last memory format")

print("Applying H100 optimizations to SAM2 components...")
try:
    if hasattr(sam2_model, 'image_encoder') and COMPILE_MODEL:
        print("Optimizing image encoder for H100...")
        sam2_model.image_encoder = torch.compile(sam2_model.image_encoder, mode='default')
    
    if hasattr(sam2_model, 'sam_mask_decoder') and COMPILE_MODEL:
        print("Optimizing mask decoder for H100...")
        sam2_model.sam_mask_decoder = torch.compile(sam2_model.sam_mask_decoder, mode='default')
        
    print("H100 component optimizations applied!")
except Exception as e:
    print(f"Component optimization failed: {e}, continuing without compilation")

predictor = SAM2ImagePredictor(sam2_model)

predictor.model.train()
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)


try:
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), 
        lr=3e-4,      
        weight_decay=4e-5,
        fused=True 
    )
    print("Using FusedAdamW optimizer for H100")
except:
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), 
        lr=3e-4, 
        weight_decay=4e-5
    )
    print("Using standard AdamW optimizer")

scaler = torch.amp.GradScaler(
    'cuda',
    init_scale=65536.0,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.7,     
    patience=1,     
    min_lr=1e-6
)

total_iterations = 30000
val_interval = 1500  
save_interval = 5000
early_stop_iou = 0.90
best_val_iou = 0.0

print(f" Optimized Iteration Setup:")
print(f"  Total Iterations: {total_iterations}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
print(f"  Validation Interval: {val_interval}")
print(f"  Save Interval: {save_interval}")
print(f"  Model Compilation: {COMPILE_MODEL}")

auto_gen = SAM2AutomaticMaskGenerator(
    sam2_model,
    points_per_side=46,     
    points_per_batch=46,    
    pred_iou_thresh=0.3,   
    stability_score_thresh=0.3,
    box_nms_thresh=0.3,
    min_mask_region_area=10
)

def run_validation_auto(iteration, downsize_width=800):
    print(f"Running validation for iteration {iteration}...")
    predictor.model.eval()
    ious = []

    with torch.no_grad():
        samples = fixed_val_data
        for idx, sample in enumerate(samples):
            gt_img = cv2.imread(sample["annotation"], cv2.IMREAD_GRAYSCALE)
            if gt_img is None:
                continue
            gt = (gt_img > 127).astype(np.uint8)

            img = cv2.imread(sample["image"])
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if H100_OPTIMIZATIONS:
                img_rgb = np.ascontiguousarray(img_rgb)
            
            H, W = img_rgb.shape[:2]

            pred_mask = np.zeros((H, W), dtype=np.uint8)
            G = 2
            tile_h, tile_w = H // G, W // G
            
            for i in range(G):
                for j in range(G):
                    y0, y1 = i*tile_h, (i+1)*tile_h if i < G-1 else H
                    x0, x1 = j*tile_w, (j+1)*tile_w if j < G-1 else W
                    tile = img_rgb[y0:y1, x0:x1]
                    
                    try:
                        with torch.amp.autocast('cuda', enabled=True):
                            masks = auto_gen.generate(tile)
                        
                        for m in masks:
                            seg = m["segmentation"].astype(np.uint8)
                            pred_mask[y0:y1, x0:x1] |= cv2.resize(
                                seg, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        print(f"Warning: Error in H100 tile processing: {e}")
                        continue

            inter = np.logical_and(pred_mask, gt).sum()
            union = np.logical_or(pred_mask, gt).sum()
            this_iou = float(inter) / (union + 1e-6)
            ious.append(this_iou)

            if idx == 0:
                vis = img_rgb.copy()
                mask_idx = gt.astype(bool)
                vis[mask_idx] = (vis[mask_idx] * 0.3 + np.array([255,255,255]) * 0.7).astype(np.uint8)
                pred_idx = pred_mask.astype(bool)
                vis[pred_idx] = (vis[pred_idx] * 0.5 + np.array([255,0,0]) * 0.5).astype(np.uint8)

                text = f"H100 Iter {iteration} IoU: {this_iou:.4f}"
                cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,255,255), 2)

                new_h = int(H * downsize_width / W)
                vis_small = cv2.resize(vis, (downsize_width, new_h), interpolation=cv2.INTER_AREA)
                out_name = os.path.join(results_dir, f"h100_val_iter{iteration:06d}_sample.png")
                cv2.imwrite(out_name, cv2.cvtColor(vis_small, cv2.COLOR_RGB2BGR))

    predictor.model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if not ious:
        print("H100 Val: No valid samples processed")
        return 0.0
    
    mean_iou = float(np.mean(ious))
    std_iou = float(np.std(ious))
    print(f"H100 Val: Mean IoU={mean_iou:.4f}+-{std_iou:.4f} from {len(ious)} samples")
    return mean_iou

train_iterations = []
train_iou_history = []
train_loss_history = []
val_iterations = []
val_iou_history = []
learning_rates = []

print(f"\n STARTING TRAINING")
print(f"Total Iterations: {total_iterations}")
print(f"Batch Size: {BATCH_SIZE} (10 random images per iteration)")

for iteration in range(1, total_iterations + 1):
    if iteration % 1000 == 0:
        print(f"\n=== Iteration {iteration}/{total_iterations} ===")
    
    images, masks, input_point, input_label = get_random_batch(train_data, batch_size=BATCH_SIZE)
    
    if len(images) == 0:
        print(f"Warning: No valid images in batch for iteration {iteration}")
        continue
    
    optimizer.zero_grad(set_to_none=True)
    
    if H100_OPTIMIZATIONS:
        for i in range(len(images)):
            images[i] = np.ascontiguousarray(images[i])
    
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        predictor.set_image_batch(images)
        input_point = [np.array(p, dtype=np.float32) for p in input_point]
        input_label = [np.array(l, dtype=np.int64) for l in input_label]
        
        batched_coords = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(p) for p in input_point], batch_first=True).cuda()
        batched_labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(l) for l in input_label], batch_first=True).cuda()
        
        mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
            batched_coords, batched_labels, box=None, mask_logits=None, normalize_coords=True)
        
        sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None)
        
        high_res = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
        
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"],
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res)
        
        logits = low_res_masks[:, 0]
        gt_mask = torch.tensor(masks).cuda().unsqueeze(1)
        
        if H100_OPTIMIZATIONS:
            gt_mask = F.interpolate(
                gt_mask, size=logits.shape[-2:], 
                mode="nearest", antialias=False
            ).squeeze(1)
        else:
            gt_mask = F.interpolate(gt_mask, size=logits.shape[-2:], mode="nearest").squeeze(1)
        
        seg_loss = F.binary_cross_entropy_with_logits(logits, gt_mask)
        
        with torch.no_grad():
            prd_mask = torch.sigmoid(logits)
            pred_bin = (prd_mask > 0.5).float()
            inter = (gt_mask * pred_bin).view(pred_bin.size(0), -1).sum(1)
            union = gt_mask.view(pred_bin.size(0), -1).sum(1) + pred_bin.view(pred_bin.size(0), -1).sum(1) - inter
            iou = inter / (union + 1e-6)
        
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + 0.1 * score_loss
        
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  
    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    batch_iou = float(iou.mean().cpu())
    current_lr = optimizer.param_groups[0]['lr']
    
    train_iterations.append(iteration)
    train_iou_history.append(batch_iou)
    train_loss_history.append(loss.item())
    learning_rates.append(current_lr)
    
    if iteration % 1000 == 0:
        print(f"  Iter {iteration}: IoU={batch_iou:.4f}, Loss={loss.item():.4f}, LR={current_lr:.2e}")
    
    if iteration % val_interval == 0:
        val_iou = run_validation_auto(iteration)  
        val_iterations.append(iteration)
        val_iou_history.append(val_iou)
        
        scheduler.step(val_iou)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"LR Update: {current_lr:.2e} -> {new_lr:.2e}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print(f"New Best Val IoU: {best_val_iou:.4f}")
            best_model_path = os.path.join(results_dir, f"SSD_h100_best_model_iter.torch")
            torch.save(predictor.model.state_dict(), best_model_path)
        
        history_path = os.path.join(results_dir, "h100_iteration_training_history.csv")
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "train_iou", "train_loss", "learning_rate", "val_iou"])
            for i, iter_num in enumerate(train_iterations):
                val_iou_for_iter = None
                if iter_num in val_iterations:
                    val_idx = val_iterations.index(iter_num)
                    val_iou_for_iter = val_iou_history[val_idx]
                writer.writerow([iter_num, train_iou_history[i], train_loss_history[i], 
                               learning_rates[i], val_iou_for_iter])
        
        if val_iou >= early_stop_iou:
            print(f"Early Stopping at iteration {iteration}, val IoU={val_iou:.4f}")
            break
    
    if iteration % save_interval == 0:
        checkpoint_path = os.path.join(results_dir, f"SSD_h100_iter{iteration:06d}.torch")
        torch.save(predictor.model.state_dict(), checkpoint_path)
        print(f"Model saved at iteration {iteration}")

final_model_name = os.path.join(results_dir, f"SSD_h100_final_iter{iteration:06d}.torch")
torch.save(predictor.model.state_dict(), final_model_name)
print(f"Final model saved: {final_model_name}")

print(f"\n RESULTS ")
print(f"Best Validation IoU: {best_val_iou:.4f}")
print(f"Final Training IoU: {train_iou_history[-1]:.4f}")
print(f"Iterations Completed: {len(train_iterations)}")
print(f"Total Training Samples Processed: {len(train_iterations) * BATCH_SIZE}")