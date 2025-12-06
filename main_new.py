import cv2
import os
import csv
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------

INPUT_VIDEO_PATH = "sania_test2.mp4"
OUTPUT_VIDEO_PATH = "output_sania_test2.mp4"

# Folder for all annotated frames
ALL_FRAMES_DIR = "frames_all"
# Folder for unattended-event frames
DEBUG_FRAME_DIR = "debug_frames"
# CSV log file
LOG_CSV_PATH = "unattended_log.csv"
# Timeline plot
TIMELINE_PNG_PATH = "unattended_timeline.png"

# Create directories
os.makedirs(ALL_FRAMES_DIR, exist_ok=True)
os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)

# Object classes to monitor
OBJECT_CLASSES = {"backpack", "handbag", "suitcase", "laptop", "book", "cell phone"}

# Person–object distance logic
NEAR_RADIUS_PIXELS = 150
UNATTENDED_THRESHOLD_SECONDS = 5.0
OBJECT_MAX_AGE_SECONDS = 5.0
IOU_MATCH_THRESHOLD = 0.3

# Save all frames as images?
SAVE_ALL_FRAMES = True


# -------------------------
# UTILITIES
# -------------------------

def iou_xyxy(box1, box2):
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
# TRACKED OBJECT
# -------------------------

class TrackedObject:
    def __init__(self, obj_id, cls_name, bbox, now_ts):
        self.id = obj_id
        self.cls_name = cls_name
        self.bbox = bbox

        self.state = "unknown"  # unknown, attended, unattended_candidate, unattended
        self.last_seen_time = now_ts
        self.last_time_with_person_nearby = None
        self.alone_since = None

        self.unattended_saved = False  # to avoid saving multiple frames for same object

    def update_bbox(self, bbox, now_ts):
        self.bbox = bbox
        self.last_seen_time = now_ts

    def update_state(self, person_nearby, now_ts):
        prev_state = self.state

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

        return prev_state, self.state


# -------------------------
# MAIN
# -------------------------

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracked_objects = []
    next_obj_id = 1
    frame_idx = 0

    # CSV log rows
    log_rows = []

    # Timeline data
    timeline_times = []
    timeline_unattended_counts = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        now_ts = frame_idx / fps

        # 1) YOLO detections
        results = model(frame, verbose=False)
        res = results[0]

        persons = []
        objs = []

        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                if cls_name == "person":
                    persons.append(bbox)
                elif cls_name in OBJECT_CLASSES:
                    objs.append((bbox, cls_name))

        # 2) IoU-based object tracking
        seen_ids = set()

        for det_bbox, det_cls in objs:
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

        # 3) Update states + detect transitions
        for obj in tracked_objects:
            if obj.id not in seen_ids:
                continue

            person_near = any(
                center_distance(obj.bbox, p_bbox) < NEAR_RADIUS_PIXELS
                for p_bbox in persons
            )

            prev_state, curr_state = obj.update_state(person_near, now_ts)

            # Save frame when object becomes unattended
            if curr_state == "unattended" and not obj.unattended_saved:
                fname = (
                    f"{DEBUG_FRAME_DIR}/frame_{frame_idx:05d}_"
                    f"obj{obj.id}_{obj.cls_name}_t{now_ts:.2f}.jpg"
                )
                cv2.imwrite(fname, frame)
                obj.unattended_saved = True

        # 4) Remove stale tracks
        tracked_objects = [
            o for o in tracked_objects
            if (now_ts - o.last_seen_time) <= OBJECT_MAX_AGE_SECONDS
        ]

        # 5) Draw and count unattended
        unattended_count = 0

        for obj in tracked_objects:
            if obj.id not in seen_ids:
                continue

            x1, y1, x2, y2 = obj.bbox

            if obj.state == "attended":
                color = (0, 255, 0)
            elif obj.state == "unattended_candidate":
                color = (0, 255, 255)
            elif obj.state == "unattended":
                color = (0, 0, 255)
                unattended_count += 1
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{obj.cls_name}:{obj.state}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            # Log per-object state for this frame
            log_rows.append([
                frame_idx,
                f"{now_ts:.3f}",
                obj.id,
                obj.cls_name,
                obj.state,
            ])

        # Optionally draw persons (blue)
        for p_bbox in persons:
            px1, py1, px2, py2 = p_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)
            cv2.putText(
                frame, "person",
                (px1, max(0, py1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Write annotated frame to video
        out.write(frame)

        # Save all frames as images (if enabled)
        if SAVE_ALL_FRAMES:
            all_frame_name = f"{ALL_FRAMES_DIR}/frame_{frame_idx:05d}.jpg"
            cv2.imwrite(all_frame_name, frame)

        # Update timeline arrays
        timeline_times.append(now_ts)
        timeline_unattended_counts.append(unattended_count)

        # If you want preview:
        # cv2.imshow("Video", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 6) Write CSV log
    with open(LOG_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_sec", "object_id", "class_name", "state"])
        writer.writerows(log_rows)

    # 7) Plot unattended timeline
    if timeline_times:
        plt.figure(figsize=(8, 4))
        plt.plot(timeline_times, timeline_unattended_counts, linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Number of unattended objects")
        plt.title("Unattended objects over time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(TIMELINE_PNG_PATH, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
