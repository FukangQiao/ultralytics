from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
from utils import make_dirs, coco91_to_coco80_class

import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np


def convert_coco_json(json_dir="../coco/annotations/", use_segments=False, cls91to80=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


def test():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)


def train():
    # Load a pretrained YOLO11n model
    # # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML 可以修改网络、类别等信息。
    # model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO("yolo11n.pt", verbose=True)

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="coco8.yaml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="cuda:0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # 验证数据集的精度Evaluate the model's performance on the validation set
    metrics = model.val()

    # 预测图片目标识别Perform object detection on an image
    # results = model("object_detction/bus.jpg")  # Predict on an image
    results = model("object_detction/zidane.jpg")  # Predict on an image
    # results = model("object_detction/demo_data/small-vehicles1.jpeg")  # Predict on an image
    # results = model("object_detction/demo_data/terrain2.png")  # Predict on an image
    results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model


def main():
    model = YOLO("yolo11n.pt")
    model.info(detailed=True, verbose=True)  # Print detailed model information


def test_train_coco():
    # model = YOLO("yolo11n.pt")  # 加载预习训练模型
    """
    加载预训练模型yolo11n.pt,可以自己构建yolo11n.yaml文件,修改网络结构、类别等信息
    训练数据参数mycoco8.yaml
    """
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 构建模型并加载预训练权重
    model.train(data="D:/Project/ultralytics/mycoco20172/mycoco.yaml", epochs=50, imgsz=640, device="cuda:0")  # 训练模型参数设置
    metrics = model.val()  # 验证数据集的精度
    results = model("object_detction/demo_data/small-vehicles1.jpeg")  # Predict on an image
    results[0].show()  # Display results


if __name__ == "__main__":
    # convert_coco("D:/BaiduNetdiskDownload/COCO2017/annotations/", "./mycoco2017")
    # convert_coco_json(
    #     "D:/BaiduNetdiskDownload/COCO2017/annotations",  # directory with *.json
    #     use_segments=True,
    #     cls91to80=True,
    # )
    test_train_coco()
    train()
