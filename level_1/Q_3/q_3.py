from ultralytics import YOLO
import os
import cv2
import numpy as np

#load model
model = YOLO('yolov8n.pt')


#load image
input_image_folder = r"D:\Chandana\Technical_task\level_1\input_images"
image_path = os.path.join(input_image_folder, "input_1.jpg")


results = model(image_path)
print("\n--- YOLOv8 DETECTION RESULTS ---")
for result in results:
        xywh = result.boxes.xywh    # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy    # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        
        # Get class names and confidence scores
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  
        confs = result.boxes.conf  
        
        # Actually print the extracted data so you can see it
        print(f"Detected Objects: {names}")
        print(f"Confidence Scores: {confs}")
        print(f"Bounding Boxes (XYXY):\n{xyxy}\n")


annotated_image = results[0].plot()

scale = 0.1
h, w = annotated_image.shape[:2]
new_w= int(w*scale)
new_h = int(h*scale)
resized_image = cv2.resize(annotated_image, (new_w, new_h))

cv2.imshow("YOLOv8 Detection", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
