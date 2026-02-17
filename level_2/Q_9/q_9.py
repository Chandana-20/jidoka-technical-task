import cv2
import numpy as np
import os
import json
import glob
import csv


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def load_model(cfg_path, weights_path, use_gpu=False):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers


def extract_features(net, img, input_size, feature_layer="conv_105"):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, tuple(input_size), swapRB=True, crop=False)
    net.setInput(blob)
    features = net.forward(feature_layer)
    return features.flatten()


def run_inference(net, output_layers, img, input_size):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, tuple(input_size), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(output_layers), h, w


def parse_detections(outputs, h, w, conf_thresh):
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for det in output:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < conf_thresh:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            boxes.append([x, y, int(bw * w), int(bh * h)])
            confidences.append(conf)
            class_ids.append(cid)
    return boxes, confidences, class_ids


def draw_and_save(img, boxes, confidences, class_ids, classes, colors, conf_thresh, nms_thresh, fname, output_dir):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, bw, bh = boxes[i]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(img, f"{classes[class_ids[i]]}: {confidences[i]:.2f}",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(os.path.join(output_dir, fname), img)
    return indices


def save_csv(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "confidence", "x", "y", "w", "h"])
        writer.writerows(rows)


def main():
    cfg = load_config("config.json")
    with open(cfg["model"]["names"]) as f:
        classes = [l.strip() for l in f.readlines()]
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    net, output_layers = load_model(cfg["model"]["cfg"], cfg["model"]["weights"], cfg["inference"]["use_gpu"])

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_filenames = []
    csv_rows = []

    for image_path in glob.glob(os.path.join(cfg["paths"]["dataset_dir"], "*.jpg")):
        img = cv2.imread(image_path)
        fname = os.path.basename(image_path)

        features = extract_features(net, img, cfg["inference"]["input_size"])
        all_features.append(features)
        all_filenames.append(fname)

        outputs, h, w = run_inference(net, output_layers, img, cfg["inference"]["input_size"])
        boxes, confidences, class_ids = parse_detections(outputs, h, w, cfg["inference"]["conf_thresh"])
        indices = draw_and_save(img, boxes, confidences, class_ids, classes, colors,
                                cfg["inference"]["conf_thresh"], cfg["inference"]["nms_thresh"], fname, output_dir)

        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, bw, bh = boxes[i]
                csv_rows.append([fname, classes[class_ids[i]], f"{confidences[i]:.2f}", x, y, bw, bh])
                print(f"{fname}  {classes[class_ids[i]]}  {confidences[i]:.2f}  [{x},{y},{bw},{bh}]")

    feature_matrix = np.array(all_features)
    np.save(os.path.join(output_dir, "features.npy"), feature_matrix)
    np.save(os.path.join(output_dir, "filenames.npy"), np.array(all_filenames))

    save_csv(os.path.join(output_dir, "detections.csv"), csv_rows)


if __name__ == "__main__":
    main()
