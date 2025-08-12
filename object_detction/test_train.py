from ultralytics import YOLO


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
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    # model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO("yolo11n.pt")

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
    results = model("object_detction/bus.jpg")  # Predict on an image
    results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model


def main():
    model = YOLO("yolo11n.pt")
    model.info(detailed=True, verbose=True)  # Print detailed model information


if __name__ == "__main__":

    main()
