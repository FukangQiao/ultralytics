import json
import os
from PIL import Image
from tqdm import tqdm


def process_annotations(json_file, images_folder, labels_folder):
    # 读取JSON文件
    with open(json_file, "r") as file:
        data = json.load(file)

    annotations = data["annotations"]

    for annotation in tqdm(annotations):
        category_id = annotation["category_id"]
        image_id = annotation["image_id"]

        # 获取图片尺寸
        file_path = os.path.join(images_folder, str(image_id) + ".jpg")
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except FileNotFoundError:
            continue

        # 计算归一化的坐标
        bbox = annotation["bbox"]
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w_norm = bbox[2] / width
        h_norm = bbox[3] / height

        # 写入标签文件，标签编号减1
        label = category_id - 1
        label_content = f"{label} {x_center} {y_center} {w_norm} {h_norm}"

        label_file_path = os.path.join(labels_folder, str(image_id) + ".txt")
        with open(label_file_path, "a") as file:
            file.write(label_content + "\n")


def create_labels_folder():
    # 创建labels文件夹及其子文件夹
    labels_folder = "labels"
    train_labels_folder = os.path.join(labels_folder, "train")
    val_labels_folder = os.path.join(labels_folder, "val")
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    return train_labels_folder, val_labels_folder


def main():
    train_labels_folder, val_labels_folder = create_labels_folder()

    # 处理训练和验证数据
    process_annotations(
        "D:/BaiduNetdiskDownload/COCO2017/annotations/instances_train2017.json", "myimages/train", train_labels_folder
    )
    process_annotations(
        "D:/BaiduNetdiskDownload/COCO2017/annotations/instances_val2017.json", "myimages/val", val_labels_folder
    )

    print("完成处理。")


if __name__ == "__main__":
    main()
