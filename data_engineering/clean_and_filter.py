import os
import glob

# Cấu hình đường dẫn tới thư mục dataset
DATASET_ROOT = "/home/user/workspace/datasets/Training_Yolo_Format_UPDATE/data_set_for_training"
SPLITS = ["train", "valid"]

# TARGET_CLASSES = None nghĩa là giữ lại TOÀN BỘ các nhãn.
# Nếu bạn chỉ muốn giữ lại class 0 và 1, hãy đổi thành: TARGET_CLASSES = [0, 1]
TARGET_CLASSES = None

def process_split(split_name):
    img_dir = os.path.join(DATASET_ROOT, "images", split_name)
    lbl_dir = os.path.join(DATASET_ROOT, "labels", split_name)
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Bỏ qua {split_name} vì không tìm thấy thư mục.")
        return

    img_paths = glob.glob(os.path.join(img_dir, "*.*"))
    
    removed_orphans = 0
    filtered_labels = 0
    
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")
        
        # 1. Xóa ảnh mồ côi (có file ảnh nhưng bị thiếu mất file nhãn .txt)
        if not os.path.exists(lbl_path):
            os.remove(img_path)
            removed_orphans += 1
            continue
            
        # 2. Lọc class mục tiêu (chỉ chạy nếu TARGET_CLASSES có giá trị cụ thể)
        if TARGET_CLASSES is not None:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            
            # Lọc chỉ giữ lại các dòng tọa độ có ID thuộc TARGET_CLASSES
            valid_lines = [line for line in lines if int(line.strip().split()[0]) in TARGET_CLASSES]
            
            # Nếu ảnh không chứa class mục tiêu nào, xóa cả ảnh lẫn nhãn để tránh nhiễu
            if len(valid_lines) == 0:
                os.remove(img_path)
                os.remove(lbl_path)
                filtered_labels += 1
            else:
                # Ghi đè file nhãn, chỉ giữ lại tọa độ của class mục tiêu
                with open(lbl_path, "w") as f:
                    f.writelines(valid_lines)
                    
    print(f"[{split_name.upper()}] Đã phát hiện và xóa {removed_orphans} ảnh mồ côi (thiếu label).")
    if TARGET_CLASSES is not None:
        print(f"[{split_name.upper()}] Đã lọc và xóa {filtered_labels} cặp ảnh/nhãn không chứa class mục tiêu.")

def main():
    print("Bắt đầu dọn dẹp và lọc tập dữ liệu...")
    for split in SPLITS:
        process_split(split)
    print("Hoàn tất xử lý dữ liệu!")

if __name__ == "__main__":
    main()