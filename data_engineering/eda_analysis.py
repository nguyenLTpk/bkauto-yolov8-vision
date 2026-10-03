import os
import random
import cv2
import matplotlib

# Backend khong can GUI (server headless) - phai set TRUOC khi import pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# Cau hinh duong dan
# --------------------------------------------------------------------------- #
DATASET_ROOT = "/home/user/workspace/datasets/Training_Yolo_Format_UPDATE/data_set_for_training"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "train")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels", "train")
DATA_YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")

OUTPUT_DIR = "/home/user/workspace/data_engineering"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "eda_sample_output.jpg")

NUM_SAMPLES = 4
GRID_ROWS, GRID_COLS = 2, 2

BOX_COLOR = (0, 255, 0)  # BGR (xanh la) khi ve bang OpenCV
BOX_THICKNESS = 2
RANDOM_SEED = None  # dat 1 so nguyen (vd 42) neu muon ket qua lap lai duoc

def load_class_names(yaml_path):
    if not os.path.isfile(yaml_path):
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {}) if data else {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names

def pick_random_images(images_dir, num_samples):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
    if len(all_images) < num_samples:
        num_samples = len(all_images)
    return random.sample(all_images, num_samples)

def yolo_to_pixel_bbox(x_center, y_center, box_w, box_h, img_w, img_h):
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h
    box_w_px = box_w * img_w
    box_h_px = box_h * img_h

    x1 = int(round(x_center_px - box_w_px / 2))
    y1 = int(round(y_center_px - box_h_px / 2))
    x2 = int(round(x_center_px + box_w_px / 2))
    y2 = int(round(y_center_px + box_h_px / 2))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2

def read_yolo_label_file(label_path):
    columns = ["class_id", "x_center", "y_center", "width", "height"]
    if not os.path.isfile(label_path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(label_path, sep=r"\s+", header=None, names=columns)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns)
    if not df.empty:
        df["class_id"] = df["class_id"].astype(int)
    return df

def draw_bboxes_on_image(image_bgr, label_df, class_names):
    img_h, img_w = image_bgr.shape[:2]
    annotated = image_bgr.copy()
    for _, row in label_df.iterrows():
        class_id = int(row["class_id"])
        x1, y1, x2, y2 = yolo_to_pixel_bbox(row["x_center"], row["y_center"], row["width"], row["height"], img_w, img_h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label_text = class_names.get(class_id, str(class_id))
        text_y = y1 - 7 if y1 - 7 > 10 else y1 + 15
        cv2.putText(annotated, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    return annotated

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    class_names = load_class_names(DATA_YAML_PATH)
    sample_filenames = pick_random_images(IMAGES_DIR, NUM_SAMPLES)
    annotated_images = [] 

    for idx, image_filename in enumerate(sample_filenames, start=1):
        image_path = os.path.join(IMAGES_DIR, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        img_h, img_w = image_bgr.shape[:2]
        label_df = read_yolo_label_file(label_path)
        annotated_bgr = draw_bboxes_on_image(image_bgr, label_df, class_names)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        title = f"{image_filename}\n{img_w}x{img_h} | {len(label_df)} box"
        annotated_images.append((annotated_rgb, title))

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
        if ax_idx < len(annotated_images):
            img_rgb, title = annotated_images[ax_idx]
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("EDA - Mau anh ngau nhien voi Bounding Box (YOLO format)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_PATH, dpi=150, format="jpg")
    plt.close(fig)
    print(f"Da luu ket qua truc quan hoa vao: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    