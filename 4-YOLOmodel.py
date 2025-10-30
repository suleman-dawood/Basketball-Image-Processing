from ultralytics import YOLO

def main():
    # Load a YOLOv8 model
    model = YOLO('yolov8s.pt') #small model

    #use YAML config for custom classes and training parameters
    config_file_path = "D:/StudyRelated/Machine Learning Projects/NBA/dataset/autoAnnotation/data.yaml"

    project = "D:/StudyRelated/Machine Learning Projects/NBA/dataset/yolo_training"
    experiment = "small_model"

    batch_size = 32

    results = model.train(data=config_file_path,
                          epochs=50,
                          project=project,
                          name=experiment,
                          batch=batch_size,
                          device='0',  # Use GPU 0
                          imgsz=640,
                          verbose=True,
                          val=True)
    
       
    print("✅ Model training and export complete.")

if __name__ == "__main__":
    main()