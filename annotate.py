#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMAGE_ROOT      = os.environ.get("IMAGE_ROOT", "data/images")
MASK_ROOT       = os.environ.get("MASK_ROOT", "data/masks")
device          = "cuda" if torch.cuda.is_available() else "cpu"
sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt")
model_cfg       = r"configs/sam2.1/sam2.1_hiera_l.yaml"

image_paths=[]
for r,_,fs in os.walk(IMAGE_ROOT):
    for f in fs:
        if os.path.splitext(f)[1]==".JPG":
            image_paths.append(os.path.join(r,f))
image_paths.sort()
def mask_exists(p):
    rel=os.path.relpath(p,IMAGE_ROOT)
    out=os.path.splitext(rel)[0]+".png"
    return os.path.exists(os.path.join(MASK_ROOT,out))
def find_unmasked(start=0):
    for i in range(start,len(image_paths)):
        if not mask_exists(image_paths[i]):
            return i
    return None

sam2_model=build_sam2(model_cfg,sam2_checkpoint,device=device)
predictor=SAM2ImagePredictor(sam2_model)
idx=find_unmasked(0)
if idx is None: raise RuntimeError
unmasked_indices = [j for j in range(len(image_paths)) if not mask_exists(image_paths[j])]
total_unmasked = len(unmasked_indices)
all_masks=[]
image=None
fig,ax=plt.subplots()
import matplotlib as mpl; mpl.rcParams['toolbar']='None'

def load_image(i):
    global image,image_path,all_masks
    image_path=image_paths[i]
    image=np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)
    all_masks=[]
    ax.clear(); ax.imshow(image)
    ax.set_title(f"[{i+1}/{len(image_paths)}] neoznaceni: {total_unmasked} klik, u, o, h, n, m, p, q")
    fig.canvas.draw_idle()

def redraw():
    ax.clear(); ax.imshow(image)
    ax.set_title(f"[{idx+1}/{len(image_paths)}] H-shrani, N-naslednji, P-prejsnji, U-razveljavi")
    if all_masks:
        cm=np.zeros(image.shape[:2],np.uint8)
        for m in all_masks: cm|=m
        ax.imshow(np.ma.masked_where(cm==0,cm),alpha=0.4,cmap='jet',interpolation='none')
    fig.canvas.draw_idle()

def on_click(ev):
    if ev.button!=1 or ev.inaxes!=ax: return
    x,y=int(ev.xdata),int(ev.ydata)
    h,w=image.shape[:2]
    cw,ch=w//2,h//2
    x0,x1=max(0,x-cw//2),min(w,x+cw//2)
    y0,y1=max(0,y-ch//2),min(h,y+ch//2)
    crop=image[y0:y1,x0:x1]
    predictor.set_image(crop)
    inp=np.array([[[x-x0,y-y0]]],dtype=np.float32)
    lab=np.array([[1]],dtype=np.int32)
    masks,_,_=predictor.predict(point_coords=inp,point_labels=lab,multimask_output=False)
    mc=(masks[0]>0.5).astype(np.uint8)
    
    # filtriramo
    mask_area = mc.sum()
    total_area = mc.size
    if mask_area / total_area > 0.5:
        print(f"[Opozorilo] Maska prevelika ({mask_area/total_area:.2%}), preskakuje...")
        predictor.set_image(image)
        return
    
    full=np.zeros(image.shape[:2],np.uint8)
    full[y0:y1,x0:x1]=mc
    all_masks.append(full)
    #redraw()
    predictor.set_image(image)

def on_key(ev):
    global idx
    k=(ev.key or "").lower()
    if k=='u' and all_masks:
        all_masks.pop(); redraw()
    elif k=='o':
        redraw()
    elif k=='h':
        cm=np.zeros(image.shape[:2],np.uint8)
        for m in all_masks: cm|=m
        rel=os.path.relpath(image_path,IMAGE_ROOT)
        out=os.path.splitext(rel)[0]+".png"
        full=os.path.join(MASK_ROOT,out)
        os.makedirs(os.path.dirname(full),exist_ok=True)
        cv2.imwrite(full,cm*255)
        nxt=find_unmasked(idx+1)
        if nxt is None: plt.close(fig)
        else: idx=nxt; load_image(idx)
    elif k=='n':
        nxt=find_unmasked(idx+1)
        if nxt is None: plt.close(fig)
        else: idx=nxt; load_image(idx)
    elif k=='m':
        nxt=find_unmasked(idx+10)
        if nxt is None: plt.close(fig)
        else: idx=nxt; load_image(idx)
    elif k=='w':
        nxt=find_unmasked(idx+4400)
        if nxt is None: plt.close(fig)
        else: idx=nxt; load_image(idx)
    elif k=='p':
        prv=None
        for j in range(idx-1,-1,-1):
            if not mask_exists(image_paths[j]):
                prv=j; break
        if prv is not None: idx=prv; load_image(idx)
    elif k=='q':
        plt.close(fig)

fig.canvas.mpl_connect('button_press_event',on_click)
fig.canvas.mpl_connect('key_press_event',on_key)
load_image(idx)
plt.show()