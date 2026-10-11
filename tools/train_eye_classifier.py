import argparse
import glob
import os
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.core.eye_classifier import EyeStateNet, EyeStateClassifier


class MRLEyeDataset(Dataset):
    """Dataset nạp ảnh mắt từ MRL Eye Dataset kết hợp CLAHE + Z-score và Augmentation."""

    def __init__(self, samples: list[tuple[str, int]], is_train: bool = True):
        self.samples = samples
        self.is_train = is_train
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.size = EyeStateClassifier.INPUT_SIZE

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)

        if self.is_train:
            # 1. Random Crop / Scale (giả lập sai số cắt vùng mắt từ keypoints thực tế)
            h, w = img.shape[:2]
            if h > 24 and w > 24:
                scale = random.uniform(0.82, 1.0)
                nh, nw = int(h * scale), int(w * scale)
                y0 = random.randint(0, h - nh)
                x0 = random.randint(0, w - nw)
                img = img[y0 : y0 + nh, x0 : x0 + nw]

            # 2. Lật ngang ngẫu nhiên
            if random.random() < 0.5:
                img = cv2.flip(img, 1)

        resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        eq = self.clahe.apply(resized)
        arr = eq.astype(np.float32)

        if self.is_train and random.random() < 0.3:
            # Nhiễu Gaussian nhẹ giả lập webcam/DroidCam thiếu sáng
            arr += np.random.normal(0, 4.0, arr.shape).astype(np.float32)

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        norm = (arr - mean) / max(std, 10.0)

        tensor = torch.from_numpy(norm).unsqueeze(0)  # (1, 48, 48)
        return tensor, torch.tensor(label, dtype=torch.long)


def collect_stratified_samples(split_dir: str, max_per_class: int, seed: int = 42) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    all_samples = []
    for label_idx, cls_name in [(0, "awake"), (1, "sleepy")]:
        cls_dir = os.path.join(split_dir, cls_name)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        if len(files) > max_per_class:
            # Lấy mẫu phân tầng đều trên toàn bộ các subject (s0001 -> s0037)
            step = len(files) / max_per_class
            files = [files[int(i * step)] for i in range(max_per_class)]
        all_samples.extend([(f, label_idx) for f in files])
    rng.shuffle(all_samples)
    return all_samples


def train_eye_classifier(
    data_dir: str = r"D:\DMS_Dataset\mrl_eye_dataset\data",
    output_weights: str = "models/eye_classifier.pt",
    epochs: int = 8,
    batch_size: int = 64,
    train_per_class: int = 5000,
    val_per_class: int = 1500,
    lr: float = 0.002,
):
    if not os.path.exists(data_dir):
        print(f"[LOI] Khong tim thay MRL Eye Dataset tai: {data_dir}")
        return

    torch.set_num_threads(min(6, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Train EyeStateNet] Dang chuan bi du lieu phan tang tu: {data_dir}...")
    train_samples = collect_stratified_samples(os.path.join(data_dir, "train"), max_per_class=train_per_class)
    val_samples = collect_stratified_samples(os.path.join(data_dir, "val"), max_per_class=val_per_class)
    print(f"  -> Train: {len(train_samples)} anh | Val: {len(val_samples)} anh | Device: {device}")

    train_loader = DataLoader(MRLEyeDataset(train_samples, is_train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MRLEyeDataset(val_samples, is_train=False), batch_size=batch_size * 2, shuffle=False)

    model = EyeStateNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(output_weights)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            train_correct += (out.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        dt = time.time() - t0

        print(
            f"Epoch [{epoch}/{epochs}] ({dt:.1f}s) | "
            f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "classes": {0: "awake", 1: "sleepy"},
                },
                output_weights,
            )

    print(f"\n[HOAN TAT] Da luu model phan loai mat tot nhat tai: {output_weights} (Best Val Acc: {best_val_acc*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mạng CNN phân loại mắt (Awake vs Sleepy) trên MRL Eye Dataset")
    parser.add_argument("--data", type=str, default=r"D:\DMS_Dataset\mrl_eye_dataset\data")
    parser.add_argument("--output", type=str, default=os.path.join(root_dir, "models", "eye_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=5000)
    args = parser.parse_args()

    train_eye_classifier(
        data_dir=args.data,
        output_weights=args.output,
        epochs=args.epochs,
        train_per_class=args.train_per_class,
    )
