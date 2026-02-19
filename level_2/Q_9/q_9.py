import cv2
import numpy as np
import os
import glob 
import json
import csv

def load_config(config_path):
    with open(config_path) as f:
        CONFIG = json.load(f)
    
    return CONFIG

def get_class_names(names):
    with open(names, "r") as f: 
        classes = [line.strip() for line in f.readlines()] 
    return classes

def load_model(cfg_path,weights_path,use_gpu=True):
    model = cv2.dnn.readNetFromDarknet(cfg_path,weights_path)
    output_layers = model.getUnconnectedOutLayersNames()
    
    if use_gpu:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return model, output_layers

def forward_pass(model, output_layers, img, input_size, scale=1/255, feature_layer="conv_105"):
    blob = cv2.dnn.blobFromImage(img, scale, input_size, swapRB=True, crop=False)
    model.setInput(blob)
    
    requested = [feature_layer] + list(output_layers)
    results   = model.forward(requested)
    
    features       = results[0].flatten()
    detection_outs = results[1:]
    
    return features, detection_outs

def parse_detections(detection_outs, h, w, conf_thresh):
    output   = np.vstack(detection_outs) 
    scores   = output[:, 5:] 
    classids = np.argmax(scores, axis=1)
    confs    = scores[np.arange(len(scores)), classids]
    
    mask        = confs > conf_thresh
    confidences = confs[mask]
    classids    = classids[mask]
    
    cx = output[:, 0] 
    cy = output[:, 1] 
    bw = output[:, 2] 
    bh = output[:, 3] 
    
    cx_px = (output[:, 0] * w).astype(int) 
    cy_px = (output[:, 1] * h).astype(int)   
    
    bw_px = (bw * w).astype(int)   
    bh_px = (bh * h).astype(int)
    
    x = cx_px - bw_px // 2
    y = cy_px - bh_px // 2
    
    boxes = np.column_stack([x, y, bw_px, bh_px])[mask]
    
    return boxes, confidences, classids
    
def draw_and_save(img, boxes, confidences, class_ids, classes, colors, conf_thresh, nms_thresh, fname, output_dir):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thresh, nms_thresh)
    
    # Handle both empty detections and different return types
    if len(indices) > 0:
        # Loop through indices (handles both flat and nested returns)
        for i in indices.flatten() if hasattr(indices, 'flatten') else indices:
            # If i is still a list/tuple (happens in some CV versions), grab the first element
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            
            x, y, bw, bh = boxes[idx]
            color = [int(c) for c in colors[class_ids[idx]]]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(img, f"{classes[class_ids[idx]]}: {confidences[idx]:.2f}",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    
    cv2.imwrite(os.path.join(output_dir, fname), img)

def main():
    cfg     = load_config("config.json")
    classes = get_class_names(cfg["model"]["names"])
    colors  = np.random.default_rng(42).integers(0, 255, size=(len(classes), 3), dtype="uint8")

    # FIX 1: use_gpu instead of input_size
    model, output_layers = load_model(cfg["model"]["cfg"], cfg["model"]["weights"], cfg["inference"]["use_gpu"])

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_features, all_filenames, csv_rows = [], [], []

    for image_path in glob.glob(os.path.join(cfg["paths"]["dataset_dir"], "*.jpg")):
        img   = cv2.imread(image_path)
        fname = os.path.basename(image_path)
        h, w  = img.shape[:2]

        features, detection_outs = forward_pass(model, output_layers, img, tuple(cfg["inference"]["input_size"]))
        all_features.append(features)
        all_filenames.append(fname)  

        boxes, confidences, class_ids = parse_detections(detection_outs, h, w, cfg["inference"]["conf_thresh"])
        draw_and_save(img, boxes, confidences, class_ids, classes, colors,
                      cfg["inference"]["conf_thresh"], cfg["inference"]["nms_thresh"], fname, output_dir)

        for i in range(len(boxes)):
            x, y, bw, bh = boxes[i]
            csv_rows.append([fname, classes[class_ids[i]], f"{confidences[i]:.2f}", x, y, bw, bh])

    np.save(os.path.join(output_dir, "features.npy"), np.array(all_features))
    np.save(os.path.join(output_dir, "filenames.npy"), np.array(all_filenames))  

if __name__ == "__main__":
    main()
