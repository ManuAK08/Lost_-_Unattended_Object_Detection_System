import cv2
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------

# Path to your input video file
INPUT_VIDEO_PATH = "input.mp4"  # change this to your video filename
OUTPUT_VIDEO_PATH = "output_unattended.mp4"

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
            if self.state == "attended":
                if self.alone_since is None:
                    self.alone_since = now_ts
                self.state = "unattended_candidate"
            elif self.state == "unattended_candidate":
                if (
                    self.alone_since is not None
                    and (now_ts - self.alone_since) > UNATTENDED_THRESHOLD_SECONDS
                ):
                    if self.last_time_with_person_nearby is not None:
                        self.state = "unattended"


# -------------------------
# MAIN
# -------------------------

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    tracked_objects = []
    next_obj_id = 1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        now_ts = frame_idx / fps  # use video time instead of wall-clock time

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
                best_obj.update_bbox(det_bbox, now_ts)
                seen_ids.add(best_obj.id)
            else:
                new_obj = TrackedObject(next_obj_id, det_cls, det_bbox, now_ts)
                tracked_objects.append(new_obj)
                seen_ids.add(next_obj_id)
                next_obj_id += 1

        # -------------------------
        # Step 3: Update state for objects seen this frame
        # -------------------------
        for obj in tracked_objects:
            if obj.id in seen_ids:
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
            obj
            for obj in tracked_objects
            if (now_ts - obj.last_seen_time) <= OBJECT_MAX_AGE_SECONDS
        ]

        # -------------------------
        # Step 5: Draw results
        # -------------------------
        for obj in tracked_objects:
            if obj.id not in seen_ids:
                continue  # draw only objects present in this frame

            x1, y1, x2, y2 = obj.bbox

            if obj.state == "attended":
                color = (0, 255, 0)        # Green
            elif obj.state == "unattended_candidate":
                color = (0, 255, 255)      # Yellow
            elif obj.state == "unattended":
                color = (0, 0, 255)        # Red
            else:
                color = (255, 255, 255)    # White

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

        # Optional: visualize persons (blue boxes)
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

        # Write to output video
        out.write(frame)

        # If you want to preview while processing, uncomment:
        # cv2.imshow("Unattended Object Detection (Video)", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
