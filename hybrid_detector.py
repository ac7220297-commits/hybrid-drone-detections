"""
Hybrid Deep Learning Architecture for Robust Drone Detection
Core Engine: Dual-YOLO + Weighted Boxes Fusion (WBF)
"""

import os
import cv2
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def custom_weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    """
    Native implementation of Weighted Boxes Fusion (WBF)
    Combines predictions from multiple object detection models by clustering 
    overlapping bounding boxes and computing confidence-weighted average coordinates.
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    weights = np.array(weights) / sum(weights)

    all_boxes = []
    all_scores = []
    all_labels = []

    for model_idx, (b_model, s_model, l_model) in enumerate(zip(boxes_list, scores_list, labels_list)):
        w = weights[model_idx]
        for b, s, l in zip(b_model, s_model, l_model):
            if s >= skip_box_thr:
                all_boxes.append(list(b))
                all_scores.append(s * w)
                all_labels.append(int(l))

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    # Sort descending by score
    order = np.argsort(all_scores)[::-1]
    all_boxes = [all_boxes[i] for i in order]
    all_scores = [all_scores[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    clusters = [] # list of lists of box indices

    for i in range(len(all_boxes)):
        matched = False
        for c in clusters:
            # Check IoU with the first (highest score) box in cluster and matching label
            if all_labels[i] == all_labels[c[0]]:
                iou = bb_intersection_over_union(all_boxes[i], all_boxes[c[0]])
                if iou >= iou_thr:
                    c.append(i)
                    matched = True
                    break
        if not matched:
            clusters.append([i])

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for c in clusters:
        cluster_boxes = np.array([all_boxes[i] for i in c])
        cluster_scores = np.array([all_scores[i] for i in c])
        
        total_score = np.sum(cluster_scores)
        # Weighted coordinate fusion
        fused_box = np.sum(cluster_boxes * cluster_scores[:, None], axis=0) / (total_score + 1e-6)
        
        # Boost confidence when multiple models agree
        model_count = len(c)
        agreement_factor = min(1.0, 0.75 + (0.25 * model_count / len(boxes_list)))
        final_conf = min(1.0, (total_score / len(boxes_list)) * agreement_factor * 1.5)

        fused_boxes.append(fused_box)
        fused_scores.append(final_conf)
        fused_labels.append(all_labels[c[0]])

    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_labels)


class HybridDroneDetector:
    def __init__(self, model1_path, model2_path, weights=[1.0, 1.0], conf_thresh=0.25, iou_thresh=0.55):
        from ultralytics import YOLO
        print(f"Loading Model 1 from: {model1_path}")
        self.model1 = YOLO(model1_path)
        print(f"Loading Model 2 from: {model2_path}")
        self.model2 = YOLO(model2_path)
        self.weights = weights
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def predict(self, image):
        """
        Runs inference on both models and fuses results using WBF.
        image: numpy array (BGR or RGB) or file path
        """
        if isinstance(image, str):
            orig_img = cv2.imread(image)
        else:
            orig_img = image.copy()

        h, w = orig_img.shape[:2]

        # Run Model 1
        res1 = self.model1.predict(orig_img, conf=self.conf_thresh, verbose=False)[0]
        boxes1 = res1.boxes.xyxy.cpu().numpy() if len(res1.boxes) > 0 else np.empty((0, 4))
        scores1 = res1.boxes.conf.cpu().numpy() if len(res1.boxes) > 0 else np.empty(0)
        labels1 = res1.boxes.cls.cpu().numpy() if len(res1.boxes) > 0 else np.empty(0)

        # Run Model 2
        res2 = self.model2.predict(orig_img, conf=self.conf_thresh, verbose=False)[0]
        boxes2 = res2.boxes.xyxy.cpu().numpy() if len(res2.boxes) > 0 else np.empty((0, 4))
        scores2 = res2.boxes.conf.cpu().numpy() if len(res2.boxes) > 0 else np.empty(0)
        labels2 = res2.boxes.cls.cpu().numpy() if len(res2.boxes) > 0 else np.empty(0)

        # Apply Weighted Boxes Fusion (Hybrid Layer)
        fused_boxes, fused_scores, fused_labels = custom_weighted_boxes_fusion(
            [boxes1, boxes2],
            [scores1, scores2],
            [labels1, labels2],
            weights=self.weights,
            iou_thr=self.iou_thresh,
            skip_box_thr=self.conf_thresh
        )

        return {
            "model1": {"boxes": boxes1, "scores": scores1, "labels": labels1, "count": len(boxes1)},
            "model2": {"boxes": boxes2, "scores": scores2, "labels": labels2, "count": len(boxes2)},
            "hybrid": {"boxes": fused_boxes, "scores": fused_scores, "labels": fused_labels, "count": len(fused_boxes)},
            "orig_shape": (h, w)
        }

    def draw_detections(self, image, detections, mode="hybrid"):
        """
        Visualizes detections on image with high-contrast glowing bounding boxes.
        mode: 'model1', 'model2', or 'hybrid'
        """
        canvas = image.copy()
        data = detections[mode]
        boxes = data["boxes"]
        scores = data["scores"]

        # Color schemes:
        # Model 1: Orange-Yellow (0, 165, 255)
        # Model 2: Cyan (255, 255, 0)
        # Hybrid: Electric Neon Green (0, 255, 128)
        color = (0, 255, 128) if mode == "hybrid" else ((0, 165, 255) if mode == "model1" else (255, 220, 0))
        label_text = "DRONE (Hybrid WBF)" if mode == "hybrid" else f"DRONE ({mode.upper()})"

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            # Draw box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            
            # Draw header tag
            tag = f"{label_text} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(canvas, tag, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas
