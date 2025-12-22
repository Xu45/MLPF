# -*- coding: utf-8 -*-
"""
Demo_CNN.py

Single character recognition (without OCR detection) with:
- Preprocessing aligned to EMNIST (28x28)
- Lightweight linear classifier (Logistic Regression)
- Test-Time Augmentation (TTA)
- Geometric decision rule for Z vs S disambiguation

Pipeline:
1. Binarization -> largest connected component -> centering -> resize to 28x28
2. Train EMNIST Letters classifier on first run and cache it
3. Multi-view TTA voting (flip / rotate)
4. If both Z and S appear, resolve via line detection
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torchvision
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_IMG = r" .jpg"
MODEL_FILE = " .pkl"


# =============================================================================
# Unicode-safe image I/O
# =============================================================================

def imread_unicode(path: str) -> np.ndarray:
    """Read image with Unicode path support."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取失败：{path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Write image with Unicode path support."""
    ext = os.path.splitext(path)[1] or ".jpg"
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_single_char(
    img_bgr: np.ndarray,
    out_side: int = 28
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a single character image:
    - Convert to grayscale
    - OTSU binarization
    - Extract largest connected component
    - Center and resize to 28x28
    - Normalize and invert (foreground ~1, background ~0)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Ensure foreground is dark
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    fg_mask = (thresh < 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )

    if num_labels <= 1:
        crop = thresh
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + np.argmax(areas)
        x, y, w, h, _ = stats[max_idx]

        pad = max(2, int(0.05 * max(w, h)))
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(thresh.shape[1] - x, w + 2 * pad)
        h = min(thresh.shape[0] - y, h + 2 * pad)

        crop = thresh[y:y + h, x:x + w]

    h, w = crop.shape
    side = max(h, w)

    canvas = np.full((side, side), 255, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop

    img28 = cv2.resize(
        canvas, (out_side, out_side), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    img28 = 1.0 - img28

    vis = cv2.cvtColor(
        (1.0 - img28) * 255.0, cv2.COLOR_GRAY2BGR
    ).astype(np.uint8)

    return img28, vis


# =============================================================================
# Model
# =============================================================================

def get_or_train_model(model_path: str = MODEL_FILE):
    """Load cached model or train a new EMNIST Letters classifier."""
    if Path(model_path).exists():
        return load(model_path)

    print("[提示] 首次运行，开始训练轻量模型...")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.EMNIST(
        root="./emnist_data",
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )

    n_samples = min(20000, len(train_set))
    indices = np.random.choice(len(train_set), n_samples, replace=False)

    X, y = [], []
    for idx in indices:
        img, label = train_set[idx]
        X.append(img.view(-1).numpy())
        y.append(label - 1)

    X = np.stack(X, axis=0)
    y = np.array(y)

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=1000, n_jobs=-1)),
    ])

    clf.fit(X, y)
    dump(clf, model_path)

    print(f"[完成] 模型已缓存：{model_path}")
    return clf


def predict_letter(img28: np.ndarray, clf):
    """Predict uppercase letter and confidence from a 28x28 image."""
    x = img28.reshape(1, -1)

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(x)[0]
        pred = int(np.argmax(prob))
        conf = float(np.max(prob))
    else:
        pred = int(clf.predict(x)[0])
        conf = 1.0

    letter = chr(ord("A") + pred)
    return letter, conf


# =============================================================================
# Test-Time Augmentation
# =============================================================================

def tta_variants(img28: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Generate TTA variants for voting."""
    return [
        ("orig", img28),
        ("flipH", np.flip(img28, axis=1)),
        ("flipV", np.flip(img28, axis=0)),
        ("rot90", np.rot90(img28, k=1)),
        ("rot270", np.rot90(img28, k=3)),
        ("rot90FH", np.flip(np.rot90(img28, k=1), axis=1)),
        ("rot270FH", np.flip(np.rot90(img28, k=3), axis=1)),
    ]


# =============================================================================
# Geometric decision (Z vs S)
# =============================================================================

def count_lines(img28: np.ndarray) -> int:
    """
    Count detected line segments for geometric disambiguation.
    Used to distinguish Z from S.
    """
    img = (img28 * 255).astype(np.uint8)
    img = 255 - img

    enlarged = cv2.resize(
        img, (168, 168), interpolation=cv2.INTER_NEAREST
    )

    edges = cv2.Canny(enlarged, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,
        maxLineGap=5,
    )

    return 0 if lines is None else len(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    img_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_IMG

    if not Path(img_path).exists():
        print("用法：python GetABC_v2.py <图片路径>")
        print(f"[错误] 找不到图片：{img_path}")
        return

    bgr = imread_unicode(img_path)
    img28, vis = preprocess_single_char(bgr)

    base = os.path.splitext(img_path)[0]
    imwrite_unicode(base + "_proc_vis.jpg", vis)

    clf = get_or_train_model()

    # Single view prediction
    letter, conf = predict_letter(img28, clf)
    print(f"[单视角] {letter} ({conf:.3f})")

    # TTA voting
    votes = {}
    records = []

    for name, variant in tta_variants(img28):
        ltr, cf = predict_letter(variant, clf)
        records.append((name, ltr, cf))
        votes[ltr] = votes.get(ltr, 0) + 1

    print("\n[TTA]")
    for name, ltr, cf in records:
        print(f"  {name:8s} -> {ltr} ({cf:.3f})")

    vote_winner = max(votes.items(), key=lambda x: x[1])[0]
    best = max(records, key=lambda r: r[2])

    letters_present = {ltr for _, ltr, _ in records}
    final_letter = vote_winner

    if {"Z", "S"} <= letters_present:
        line_count = count_lines(img28)
        print(f"\n[几何裁决] 检测到直线段数量：{line_count}")

        final_letter = "Z" if line_count >= 2 else "S"
        print(
            f"[裁决结果] {final_letter}（覆盖投票：{vote_winner}；"
            f"最高置信度：{best[1]}）"
        )
    else:
        print(f"\n[投票结果] {vote_winner}")
        print(
            f"[最高置信度] {best[1]}  来自视角：{best[0]}  "
            f"置信度：{best[2]:.3f}"
        )

    print(f"\n== 最终预测：{final_letter} ==")


if __name__ == "__main__":
    main()
