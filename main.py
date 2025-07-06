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


# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 10/10 [00:09<00:00,  1.10it/s]
#                    all        150        262      0.773      0.652       0.75      0.517
#                buffalo         38         60      0.932       0.55      0.729      0.521
#               elephant         43         83      0.567      0.578      0.597      0.411
#                  rhino         42         58      0.808      0.741      0.848      0.621
#                  zebra         28         61      0.784      0.738      0.824      0.515

# # --- SINGLE IMAGE PREDICTION ---
# predictor = YOLO(os.path.join("runs/yolo_exp", "yolov10s_animals", "weights", "best.pt"))
# output_list = predictor([
#     "final_data/valid/images/0015.jpg",
#     "final_data/valid/images/0888.jpg"
# ])
#
# for output in output_list:
#     output.show()
#
# # --- VIDEO PREDICTION ---
# video_detector = YOLO(os.path.join("runs/yolo_exp", "yolov10s_animals", "weights", "best.pt"))
# video_detector.predict(
#     source=video_input_path,
#     conf=0.25,
#     save=True,
#     save_txt=False,
#     project='runs/detect',
#     name='predict',
#     exist_ok=True
# )
