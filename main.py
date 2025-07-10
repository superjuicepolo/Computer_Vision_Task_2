from ultralytics import YOLO
import os

for model_name in ["yolov10s", "yolov10m", "yolov10l"]:
    # --- TRAINING ---
    model_to_train = YOLO(f'{model_name}.pt')
    model_to_train.train(
        data=os.path.abspath("final_data/data_wl.yaml"),
        epochs=10,
        imgsz=320,
        batch=4,
        project="runs/yolo_exp",
        name=f"{model_name}_animals",
        save=True
    )


    # --- EVALUATION ---
    trained_model = YOLO(os.path.join("runs/yolo_exp", f"{model_name}_animals", "weights", "best.pt"))
    eval_results = trained_model.val(
        data=os.path.abspath("final_data/data_wl.yaml"),
        split='val'
    )

    print("Evaluation Results:")
    print(eval_results)


# --- SINGLE IMAGE PREDICTION ---
predictor = YOLO(os.path.join("runs/yolo_exp", "yolov10s_animals", "weights", "best.pt"))
output_list = predictor([
    "final_data/valid/images/0015.jpg",
    "final_data/valid/images/0888.jpg"
])

for output in output_list:
    output.show()

# --- VIDEO PREDICTION ---
video_detector = YOLO(os.path.join("runs/yolo_exp", "yolov10s_animals", "weights", "best.pt"))
video_detector.predict(
    source= "Cape Buffalo charge.mp4",
    conf=0.25,
    save=True,
    save_txt=False,
    project='runs/detect',
    name='predict',
    exist_ok=True
)
