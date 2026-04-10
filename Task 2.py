import os
import math
import random
import argparse
import warnings
from collections import defaultdict

from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

## 기본값 설정 ##
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_XLSX = "OSATS_MICCAI_trainset_avg.xlsx"
VIDEO_DIR = "videos"
RUNS_DIR  = "runs/osats_exp"
SUBMIT_CSV = "MViTv2-S_submission_OSATS.csv"  # 출력 csv 파일 이름
CHECKPOINT_BEST_TPL = "MViTv2-S_fold{fold}_best_model.pth"  # Early stopping 발생 시 저장
CHECKPOINT_REFIT_TPL = "MViTv2-S_fold{fold}_refit_model.pth"  # 전체 데이터 재학습 모델 저장

## 하이퍼파라미터 설정 ##
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-5
PATIENCE = 10           # Early stopping patience
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Clip / Sampling 관련 ##
CLIP_LEN = 16  # 한 클립의 프레임 수
CROP_SIZE = 224
# 4:28~4:30 구간에서 4프레임 간격으로 총 16프레임 추출
WINDOW_START_SEC = 268.0  # 4:28
WINDOW_END_SEC = 270.0  # 4:30
FRAME_INTERVAL = 4  # 4 frames interval
FOLDS = 5

TARGET_COLUMNS = [
    'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT', 'OSATS_SUTURE',
    'OSATS_FLOW', 'OSATS_KNOWLEDGE', 'OSATS_PERFORMANCE', 'OSATS_FINAL_QUALITY'
]
CSV_COLUMNS = [
    'VIDEO', 'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT', 'OSATS_SUTURE',
    'OSATS_FLOW', 'OSATS_KNOWLEDGE', 'OSATS_PERFORMANCE', 'OSATSFINALQUALITY'
]
NUM_CLASSES = 5  # 모델에는 0~4로 입력 --> 실제로는 1~5로 출력

warnings.filterwarnings("ignore", category=UserWarning)

## Utils ##
def set_seed(worker_id):
    base_seed = SEED
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)

def center_crop_224(img):
    h, w = img.shape[:2]
    if min(h, w) < CROP_SIZE:
        if h < w:
            new_h = CROP_SIZE
            new_w = int(w * (CROP_SIZE / h))
        else:
            new_w = CROP_SIZE
            new_h = int(h *(CROP_SIZE / h))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - CROP_SIZE // 2)
    x1 = max(0, cx - CROP_SIZE // 2)
    y2 = y1 + CROP_SIZE
    x2 = x1 + CROP_SIZE
    y2 = min(h, y2); x2 = min(w, x2)
    y1 = y2 - CROP_SIZE; x1 = x2 - CROP_SIZE
    return img[y1:y2, x1:x2]

def compute_grs_class(scores_1to5):
    score_sum = int(np.sum(scores_1to5))
    if 8 <= score_sum <= 15: return 0
    if 16 <= score_sum <= 23: return 1
    if 24 <= score_sum <= 31: return 2
    if 32 <= score_sum <= 40: return 3
    return 0

def to_submission_df(video_preds):
    rows = []
    for vid, pred_vec in video_preds.items():  # pred_vec: (8,) ints 0..4
        rows.append({
            "VIDEO": f"{vid}",
            "OSATS_RESPECT": int(pred_vec[0]),
            "OSATS_MOTION": int(pred_vec[1]),
            "OSATS_INSTRUMENT": int(pred_vec[2]),
            "OSATS_SUTURE": int(pred_vec[3]),
            "OSATS_FLOW": int(pred_vec[4]),
            "OSATS_KNOWLEDGE": int(pred_vec[5]),
            "OSATS_PERFORMANCE": int(pred_vec[6]),
            "OSATSFINALQUALITY": int(pred_vec[7]),
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

## Dataset ##
class VideoClipDataset(Dataset):
    def __init__(self, df, mode="train", return_raw=False):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.return_raw = return_raw
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3,1,1,1)
        self.std  = torch.tensor([0.225, 0.225, 0.225]).view(3,1,1,1)

    def __len__(self):
        return len(self.df)

    def _frame_indices(self, total_frames, fps):
        if fps is None or fps <= 1e-3: fps = 30.0
        start_idx = int(WINDOW_START_SEC * fps)
        # 16개 프레임, 간격 4 --> 마지막 인덱스 = start_idx + 4*(16-1)
        idxs = [start_idx + FRAME_INTERVAL * i for i in range(CLIP_LEN)]
        # 비디오 길이 초과 시 마지막 프레임으로 패딩
        idxs = [min(i, total_frames - 1) for i in idxs]
        return idxs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        labels_1to5 = row[TARGET_COLUMNS].values.astype(np.int64)
        labels_0to4 = np.clip(labels_1to5 - 1, 0, 4).astype(np.int64)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        idxs = self._frame_indices(total_frames, fps)

        frames_rgb_224 = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret or frame is None:
                if len(frames_rgb_224) == 0:
                    frame = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                else:
                    frame = frames_rgb_224[-1][:,:,::-1]  # revert to BGR to reuse pipeline
                    frame = frame[:, :, ::-1]
            frame = center_crop_224(frame)
            frame = frame[:, :, ::-1]  # BGR to RGB
            frames_rgb_224.append(frame)
        cap.release()

        #(T,H,W,C) --> (C,T,H,W), normalize
        clip = np.stack(frames_rgb_224, axis=0)  # (T,224,224,3)
        clip_t = torch.from_numpy(clip).permute(3,0,1,2).float() / 255.0  # (3,T,224,224)
        clip_t = (clip_t - self.mean) / self.std

        labels = torch.from_numpy(labels_0to4)  # (8,)
        vid_id = os.path.splitext(os.path.basename(video_path))[0]

        if self.return_raw:
            raw = torch.from_numpy(np.ascontiguousarray(clip)).permute(0,3,1,2)  # (T,3,224,224)
            return clip_t, labels, vid_id, raw
        else:
            return clip_t, labels, vid_id

## Model ##
class MViTv2Classifier(nn.Module):
    def __init__(self, num_targets=8, num_classes=5):
        super().__init__()
        weights = models.video.MViT_V2_S_Weights.DEFAULT
        self.backbone = models.video.mvit_v2_s(weights=weights)
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(768, num_targets * num_classes)
        self.num_targets = num_targets
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.backbone(x)  # (B,768)
        logits = self.classifier(feat)  # (B, num_targets*num_classes)
        return logits.view(-1, self.num_targets, self.num_classes)  # (B,8,5)

## Training & Eval 헬퍼 ##
def ce_multitask_loss(logits, targets):
    # logits: (B,8,5), targets: (B,8) in 0..4
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )

def preds_from_logits(logits):
    # (B,8,5) -> (B,8) ints 0..4
    return torch.argmax(logits, dim=-1)

def build_eval_df(vids, preds_np, labels_np):
    # preds_np, labels_np: (N,8) ints 0..4
    rows = []
    for vid, pvec, lvec in zip(vids, preds_np, labels_np):
        for i, col in enumerate(TARGET_COLUMNS):
            rows.append({"item_id": f"{vid}_{col}",
                         "ground_truth": int(lvec[i]),
                         "prediction": int(pvec[i])})
    return pd.DataFrame(rows, columns=["item_id","ground_truth","prediction"])

## 폴드 하나 Train ##
def train_one_fold(fold_id, train_df, val_df):
    log_dir = os.path.join(RUNS_DIR, f"MViTv2-S_fold{fold_id}")
    writer = SummaryWriter(log_dir=log_dir)
    from dice_score import get_f1  # expects df with item_id/ground_truth/prediction, num_classes=5

    model = MViTv2Classifier(num_targets=len(TARGET_COLUMNS), num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_set = VideoClipDataset(train_df, mode="train", return_raw=False)
    val_set   = VideoClipDataset(val_df, mode="val", return_raw=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=set_seed)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=set_seed)

    best_f1 = -1.0
    patience = 0
    best_epoch = 0
    best_path = CHECKPOINT_BEST_TPL.format(fold=fold_id)

    for epoch in range(EPOCHS):
        # -------- Train --------
        model.train()
        running_loss = 0.0
        n_batches = 0
        for clips, targets, _ in train_loader:
            clips = clips.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(clips)          # (B,8,5)
            loss = ce_multitask_loss(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        # -------- Val --------
        model.eval()
        val_loss = 0.0
        vb = 0
        all_vids = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for clips, targets, vids in val_loader:
                clips = clips.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                logits = model(clips)
                loss = ce_multitask_loss(logits, targets)
                val_loss += loss.item(); vb += 1

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labs  = targets.cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labs)
                all_vids.extend(vids)
        val_loss = val_loss / max(vb, 1)

        if len(all_preds):
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            df_eval = build_eval_df(all_vids, all_preds, all_labels)
            val_f1 = float(get_f1(df_eval, num_classes=NUM_CLASSES))
        else:
            val_f1 = 0.0

        print(f"[Fold {fold_id}] Epoch {epoch+1}/{EPOCHS} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValF1 {val_f1:.4f}")

        # Early stopping on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[Fold {fold_id}] Early stopping at epoch {epoch+1}. Best F1={best_f1:.4f} (epoch {best_epoch})")
                break

    # load best for return
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, best_f1, best_epoch

## EarlyStopping 후 전체 데이터 재학습 ##
def refit_on_all_data(fold_id, full_df, epochs_for_refit):
    model = MViTv2Classifier(num_targets=len(TARGET_COLUMNS), num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    full_set = VideoClipDataset(full_df, mode="train", return_raw=False)
    full_loader = DataLoader(full_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=set_seed)

    model.train()
    for epoch in range(max(1, epochs_for_refit)):
        running = 0.0; nb = 0
        for clips, targets, _ in full_loader:
            clips = clips.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(clips)
            loss = ce_multitask_loss(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
        print(f"[Fold {fold_id} Refit] Epoch {epoch+1}/{epochs_for_refit} | Loss {running/max(nb,1):.4f}")

    refit_path = CHECKPOINT_REFIT_TPL.format(fold=fold_id)
    torch.save(model.state_dict(), refit_path)
    print(f"[Fold {fold_id}] Saved refit model to {refit_path}")
    return model

## Grad-CAM 없이 앙상블 추론 ##
def infer_ensemble(models_list, infer_df):
    ds = VideoClipDataset(infer_df[['video_path'] + TARGET_COLUMNS], mode="val", return_raw=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=set_seed)
    video_preds = {}
    for m in models_list:
        m.eval()

    with torch.no_grad():
        for clip, _, vid in tqdm(loader, desc="Inference"):
            clip = clip.to(DEVICE, non_blocking=True)
            logits_sum = None
            for m in models_list:
                out = m(clip)
                logits_sum = out if logits_sum is None else logits_sum + out
            logits_avg = logits_sum / float(len(models_list))
            preds = torch.argmax(logits_avg, dim=-1).cpu().numpy()[0]
            video_preds[vid[0]] = preds.astype(int)
    return video_preds

## Main ##
def main():
    # Load metadata
    df = pd.read_excel(DATA_XLSX)
    df['video_name'] = df['VIDEO'].astype(str) + ".mp4"
    df['video_path'] = df['video_name'].apply(lambda x: os.path.join(VIDEO_DIR, x))
    assert all([c in df.columns for c in TARGET_COLUMNS]), f"Missing target columns in {DATA_XLSX}"

    # Build GRS class for stratification
    df['GRS_CLASS'] = df[TARGET_COLUMNS].apply(lambda r: compute_grs_class(r.values), axis=1)

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_models = []
    fold_scores = []
    best_epochs = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(df, df['GRS_CLASS']), start=1):
        train_df = df.iloc[train_idx][['video_path'] + TARGET_COLUMNS].copy().reset_index(drop=True)
        val_df   = df.iloc[val_idx][['video_path'] + TARGET_COLUMNS].copy().reset_index(drop=True)

        # EarlyStopping
        _, best_f1, best_epoch = train_one_fold(fold_id, train_df, val_df)
        fold_scores.append(best_f1)
        best_epochs.append(best_epoch)
        print(f"[Fold {fold_id}] Best Val F1: {best_f1:.4f} @ epoch {best_epoch}")

        # 전체 데이터로 refit
        full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        refit_model = refit_on_all_data(fold_id, full_df, epochs_for_refit=best_epoch)
        fold_models.append(refit_model)

    print("Fold F1 scores:", fold_scores, " | Mean:", float(np.mean(fold_scores)))
    print("Refit epochs per fold:", best_epochs)

    # Save submission CSV (0..4 as requested)
    video_preds = infer_ensemble(fold_models, df[['video_path'] + TARGET_COLUMNS])
    sub_df = to_submission_df(video_preds)
    sub_df.to_csv(SUBMIT_CSV, index=False)
    print(f"Saved submission to {SUBMIT_CSV}")

if __name__ == "__main__":
    main()