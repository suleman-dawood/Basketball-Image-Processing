import os
import shutil
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
# ROOT = r"D:/StudyRelated/Machine Learning Projects/NBA/dataset/sample_annotation/train"
# For FULL later:
#ROOT = r"D:/StudyRelated/Machine Learning Projects/NBA/dataset/autoAnnotation/train"
ROOT = r"D:/StudyRelated/Machine Learning Projects/NBA/dataset/autoAnnotation/valid"

IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_DIR = os.path.join(ROOT, "labels")
BACKUP_DIR = os.path.join(LABELS_DIR, "backup_labels")
VIS_OUT_DIR = os.path.join(ROOT, "vis_color_clean")

IOU_THRESHOLD = 0.5
MAX_IMAGES = None  # e.g., set to 10 to limit visualisations

CLASS_NAMES = ["USA Player", "Opponent Player", "Basketball", "Referee"]
# BGR for drawing
CLASS_COLORS = {0:(0,0,255), 1:(0,255,0), 2:(255,165,0), 3:(255,0,255)}

USA_CLASS = 0
OPP_CLASS = 1
BALL_CLASS = 2
REF_CLASS = 3

def ensure_backup(labels_dir: str, backup_dir: str) -> None:
    os.makedirs(backup_dir, exist_ok=True)
    for fn in os.listdir(labels_dir):
        if fn.endswith(".txt"):
            src = os.path.join(labels_dir, fn)
            dst = os.path.join(backup_dir, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

def load_yolo(txt_path: str) -> List[Tuple[int,float,float,float,float]]:
    if not os.path.exists(txt_path):
        return []
    out = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                out.append((cls, x, y, w, h))
    return out

def save_yolo(txt_path: str, anns: List[Tuple[int,float,float,float,float]]) -> None:
    with open(txt_path, "w") as f:
        for cls, x, y, w, h in anns:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def iou_yolo(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax - aw/2, ay - ah/2
    ax2, ay2 = ax + aw/2, ay + ah/2
    bx1, by1 = bx - bw/2, by - bh/2
    bx2, by2 = bx + bw/2, by + bh/2
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

def crop_upper_half(img_bgr, box_xywh_norm):
    H, W = img_bgr.shape[:2]
    x, y, w, h = box_xywh_norm
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H)
    # upper half (jersey region)
    y_mid = y1 + (y2 - y1)//2
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y_mid)
    if x2c <= x1c or y2c <= y1c:
        return None
    return img_bgr[y1c:y2c, x1c:x2c]

def team_color_scores(img_bgr_roi):
    """
    Returns (blue_prop, white_prop) in ROI using HSV thresholds.
    - Blue (OpenCV HSV): H in [100,135], S>=60, V>=60
    - White: S<=30, V>=200
    """
    if img_bgr_roi is None or img_bgr_roi.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    blue_mask = (H >= 100) & (H <= 135) & (S >= 60) & (V >= 60)
    white_mask = (S <= 30) & (V >= 200)

    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 0.0, 0.0

    blue_prop = float(np.count_nonzero(blue_mask)) / total
    white_prop = float(np.count_nonzero(white_mask)) / total
    return blue_prop, white_prop

def resolve_pair_by_color(img_bgr, a, b):
    """
    a,b are (cls, x, y, w, h). They are assumed to be USA/Opponent overlapping.
    Decide which to keep via color score of jersey region.
    """
    # Ensure the two classes are USA/OPP
    if not {a[0], b[0]} <= {USA_CLASS, OPP_CLASS}:
        # If something odd, keep the bigger box
        area_a = a[3]*a[4]
        area_b = b[3]*b[4]
        return a if area_a >= area_b else b

    # Compute color proportions
    roi_a = crop_upper_half(img_bgr, a[1:5])
    roi_b = crop_upper_half(img_bgr, b[1:5])
    a_blue, a_white = team_color_scores(roi_a)
    b_blue, b_white = team_color_scores(roi_b)

    # Compute class-aligned scores
    def score(entry, blue, white):
        if entry[0] == USA_CLASS:
            return blue - white  # favor blue
        if entry[0] == OPP_CLASS:
            return white - blue  # favor white
        return 0.0

    s_a = score(a, a_blue, a_white)
    s_b = score(b, b_blue, b_white)

    if s_a > s_b + 0.01:
        return a
    if s_b > s_a + 0.01:
        return b

    # tie-breaker: larger box (in normalized space)
    area_a = a[3]*a[4]
    area_b = b[3]*b[4]
    return a if area_a >= area_b else b

def color_aware_clean_for_image(img_bgr, anns):
    """
    anns: List[(cls,x,y,w,h)], return cleaned list
    Strategy:
      - Basketball:
          * Always keep (never filtered out)
      - Referees:
          * Always keep
          * If overlapping strongly with players, Referee wins (player dropped)
      - Players (USA/Opponent):
          * If overlapping strongly (IoU > thr), keep one by color rule
          * If no strong overlap, keep both
    """
    # Separate classes
    balls = [a for a in anns if a[0] == BALL_CLASS]
    refs  = [a for a in anns if a[0] == REF_CLASS]
    players = [a for a in anns if a[0] in (USA_CLASS, OPP_CLASS)]

    kept = []

    # --- Balls (untouchable) ---
    kept.extend(balls)

    # --- Referees (priority) ---
    used_players = [False] * len(players)
    for ref in refs:
        ref_box = ref[1:5]
        # drop overlapping players if IoU > threshold
        new_players = []
        for idx, p in enumerate(players):
            if iou_yolo(ref_box, p[1:5]) > IOU_THRESHOLD:
                used_players[idx] = True  # suppressed
            else:
                new_players.append(p)
        players = new_players
        kept.append(ref)

    # --- Players (USA vs Opponent disambiguation) ---
    used = [False] * len(players)
    for i in range(len(players)):
        if used[i]:
            continue
        a = players[i]
        overlaps = [i]
        for j in range(i+1, len(players)):
            if used[j]:
                continue
            b = players[j]
            if iou_yolo(a[1:5], b[1:5]) > IOU_THRESHOLD:
                overlaps.append(j)

        if len(overlaps) == 1:
            kept.append(a)
            used[i] = True
        else:
            # resolve overlapping group
            winner = players[overlaps[0]]
            for k in overlaps[1:]:
                winner = resolve_pair_by_color(img_bgr, winner, players[k])
            for k in overlaps:
                used[k] = True
            kept.append(winner)

    return kept


def draw(img_bgr, anns):
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    for cls, x, y, w, h in anns:
        x1 = int((x - w/2) * W); y1 = int((y - h/2) * H)
        x2 = int((x + w/2) * W); y2 = int((y + h/2) * H)
        color = CLASS_COLORS.get(cls, (200,200,0))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, CLASS_NAMES[cls], (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out

def main():
    os.makedirs(VIS_OUT_DIR, exist_ok=True)
    ensure_backup(LABELS_DIR, BACKUP_DIR)

    img_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    img_files.sort()
    if MAX_IMAGES is not None:
        img_files = img_files[:MAX_IMAGES]

    # --- pick 5 random images to visualise ---
    import random
    vis_samples = set(random.sample(img_files, min(5, len(img_files))))

    for name in img_files:
        stem, _ = os.path.splitext(name)
        img_path = os.path.join(IMAGES_DIR, name)
        lbl_path = os.path.join(LABELS_DIR, f"{stem}.txt")
        backup_lbl = os.path.join(BACKUP_DIR, f"{stem}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        before = load_yolo(backup_lbl) if os.path.exists(backup_lbl) else load_yolo(lbl_path)
        after  = color_aware_clean_for_image(img, before)

        # write cleaned back so your other scripts keep working
        save_yolo(lbl_path, after)

        if name in vis_samples:
            # --- only visualise for these samples ---
            vis_before = draw(img, before)
            vis_after  = draw(img, after)

            fig, ax = plt.subplots(1,2, figsize=(12,6))
            ax[0].imshow(cv2.cvtColor(vis_before, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"{stem} — BEFORE ({len(before)})"); ax[0].axis('off')
            ax[1].imshow(cv2.cvtColor(vis_after,  cv2.COLOR_BGR2RGB)); ax[1].set_title(f"{stem} — AFTER ({len(after)})");  ax[1].axis('off')
            out_path = os.path.join(VIS_OUT_DIR, f"{stem}_compare.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[VIS] {name}: before={len(before)} → after={len(after)}  saved {out_path}")
        else:
            # no visualisation, just log
            print(f"[OK] {name}: before={len(before)} → after={len(after)}")

    print(f"\n✅ Done. Visualisations saved for {len(vis_samples)} random images in: {VIS_OUT_DIR}")
    print("   (Switch ROOT to autoAnnotation/train when you’re ready.)")


if __name__ == "__main__":
    main()
