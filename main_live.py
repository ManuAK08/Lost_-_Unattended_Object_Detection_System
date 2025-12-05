# import cv2
# from ultralytics import YOLO


# def main():
#     # 1. Load YOLO model (COCO pretrained)
#     #    yolov8n is small + faster; you can switch to yolov8s/ m / l later
#     model = YOLO("yolov8n.pt")

#     # 2. Configure camera: Continuity Camera on macOS
#     #    You already confirmed: index = 0, backend = 1200 (AVFoundation)
#     CAM_INDEX = 0
#     BACKEND = cv2.CAP_AVFOUNDATION  # 1200

#     cap = cv2.VideoCapture(CAM_INDEX, BACKEND)

#     if not cap.isOpened():
#         raise RuntimeError(
#             "ERROR: Could not access Continuity Camera. "
#             "Check macOS camera permissions and that the iPhone is connected/unlocked."
#         )

#     # Optional: set a smaller resolution if performance is slow
#     # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#     # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     print("Live YOLOv8 Object Detection (Continuity Camera)")
#     print("Press 'q' in the video window to quit.\n")

#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("ERROR: Failed to grab frame from Continuity Camera.")
#             break

#         # 3. Run YOLO inference on the frame
#         results = model(frame, verbose=False)
#         res = results[0]

#         # 4. Draw bounding boxes and labels
#         if res.boxes is not None:
#             for box in res.boxes:
#                 # xyxy = [x1, y1, x2, y2]
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 cls_id = int(box.cls[0].item())
#                 conf = float(box.conf[0].item())

#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 cls_name = model.names.get(cls_id, str(cls_id))

#                 # Draw rectangle
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                 # Draw label
#                 label = f"{cls_name} {conf:.2f}"
#                 cv2.putText(
#                     frame,
#                     label,
#                     (x1, max(0, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 0),
#                     2,
#                     cv2.LINE_AA,
#                 )

#         # 5. Show live window
#         cv2.imshow("Live YOLOv8 (Continuity Camera)", frame)

#         # 6. Quit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     # 7. Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Webcam released. Window closed.")


# if __name__ == "__main__":
#     main()


import time
import cv2
from ultralytics import YOLO


# -------------------------
# CONFIG
# -------------------------

CAM_INDEX = 0
CAM_BACKEND = cv2.CAP_AVFOUNDATION  # for macOS + Continuity Camera

# Classes we consider as "objects of interest" for unattended detection
OBJECT_CLASSES = {"backpack", "handbag", "suitcase", "laptop", "book", "cell phone"}

# Radius in pixels for "person near object"
NEAR_RADIUS_PIXELS = 150

# How long (seconds) an object must be alone after being attended to be flagged as unattended
UNATTENDED_THRESHOLD_SECONDS = 5.0

# How long to keep a lost object track before dropping it (seconds)
OBJECT_MAX_AGE_SECONDS = 5.0

# IoU threshold to match detections across frames
IOU_MATCH_THRESHOLD = 0.3


# -------------------------
# UTILITIES
# -------------------------

def iou_xyxy(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def center_distance(b1, b2):
    x1, y1 = box_center(b1)
    x2, y2 = box_center(b2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# -------------------------
# TRACKED OBJECT CLASS
# -------------------------

class TrackedObject:
    def __init__(self, obj_id, cls_name, bbox, now_ts):
        self.id = obj_id
        self.cls_name = cls_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.state = "unknown"  # unknown, attended, unattended_candidate, unattended

        self.last_seen_time = now_ts
        self.last_time_with_person_nearby = None
        self.alone_since = None

    def update_bbox(self, bbox, now_ts):
        self.bbox = bbox
        self.last_seen_time = now_ts

    def update_state(self, person_nearby, now_ts):
        if person_nearby:
            self.last_time_with_person_nearby = now_ts
            self.alone_since = None
            if self.state in ["unknown", "unattended_candidate"]:
                self.state = "attended"
        else:
            # No person nearby
            if self.state == "attended":
                # Start "alone" timer
                if self.alone_since is None:
                    self.alone_since = now_ts
                self.state = "unattended_candidate"
            elif self.state == "unattended_candidate":
                if self.alone_since is not None and (now_ts - self.alone_since) > UNATTENDED_THRESHOLD_SECONDS:
                    # Only mark unattended if it was attended at some point
                    if self.last_time_with_person_nearby is not None:
                        self.state = "unattended"


# -------------------------
# MAIN
# -------------------------

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Open camera
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    if not cap.isOpened():
        raise RuntimeError("Could not open Continuity Camera. Check permissions and device index.")

    tracked_objects = []
    next_obj_id = 1

    print("Live unattended-object detection running.")
    print("Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        now_ts = time.time()

        # -------------------------
        # Step 1: YOLO detection
        # -------------------------
        results = model(frame, verbose=False)
        res = results[0]

        persons = []
        object_detections = []  # list of (bbox, cls_name)

        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = model.names.get(cls_id, str(cls_id))

                bbox = [int(x1), int(y1), int(x2), int(y2)]

                if cls_name == "person":
                    persons.append(bbox)
                elif cls_name in OBJECT_CLASSES:
                    object_detections.append((bbox, cls_name))

        # -------------------------
        # Step 2: Match detections to tracked objects (IoU-based)
        # -------------------------
        seen_ids = set()

        for det_bbox, det_cls in object_detections:
            best_iou = 0.0
            best_obj = None

            for obj in tracked_objects:
                if obj.cls_name != det_cls or obj.id in seen_ids:
                    continue
                iou = iou_xyxy(obj.bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj

            if best_obj is not None and best_iou >= IOU_MATCH_THRESHOLD:
                # Update existing track
                best_obj.update_bbox(det_bbox, now_ts)
                seen_ids.add(best_obj.id)
            else:
                # Create new track
                new_obj = TrackedObject(next_obj_id, det_cls, det_bbox, now_ts)
                tracked_objects.append(new_obj)
                seen_ids.add(next_obj_id)
                next_obj_id += 1

        # -------------------------
        # Step 3: Update state for objects seen this frame
        # -------------------------
        for obj in tracked_objects:
            if obj.id in seen_ids:
                # Determine if any person is near this object
                person_nearby = False
                for p_bbox in persons:
                    dist = center_distance(obj.bbox, p_bbox)
                    if dist < NEAR_RADIUS_PIXELS:
                        person_nearby = True
                        break

                obj.update_state(person_nearby, now_ts)

        # -------------------------
        # Step 4: Remove stale tracks
        # -------------------------
        tracked_objects = [
            obj for obj in tracked_objects
            if (now_ts - obj.last_seen_time) <= OBJECT_MAX_AGE_SECONDS
        ]

        # -------------------------
        # Step 5: Draw results
        # -------------------------
        for obj in tracked_objects:
            if obj.id not in seen_ids:
                # Only draw if seen this frame
                continue

            x1, y1, x2, y2 = obj.bbox

            if obj.state == "attended":
                color = (0, 255, 0)        # Green
            elif obj.state == "unattended_candidate":
                color = (0, 255, 255)      # Yellow
            elif obj.state == "unattended":
                color = (0, 0, 255)        # Red
            else:
                color = (255, 255, 255)    # White for unknown

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{obj.cls_name}:{obj.state}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        # Optional: draw persons too (just for visualization)
        for p_bbox in persons:
            px1, py1, px2, py2 = p_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)
            cv2.putText(
                frame,
                "person",
                (px1, max(0, py1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Unattended Object Detection (Continuity Camera)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
