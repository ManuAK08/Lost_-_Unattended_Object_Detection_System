import os
import csv
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


# -------------------------
# CONFIG
# -------------------------

class Config:
    # I/O paths (same pattern as your code)
    INPUT_VIDEO_PATH = "Nair_test4.mp4"
    OUTPUT_VIDEO_PATH = "output_Nair_test44.mp4"

    ALL_FRAMES_DIR = "frames_all"
    DEBUG_FRAME_DIR = "debug_frames"
    LOG_CSV_PATH = "unattended_log.csv"
    TIMELINE_PNG_PATH = "unattended_timeline.png"

    # YOLO + tracking
    YOLO_MODEL = "yolov8m.pt"      # use m model (like your teammate); change to yolov8n.pt if needed
    CONF_THRESHOLD = 0.35
    # COCO class IDs: 0=person, 24=backpack, 26=handbag, 28=suitcase, 63=laptop
    TARGET_CLASS_IDS = [0, 24, 26, 28, 63]
    ITEM_CLASS_IDS = [24, 26, 28, 63]

    # Ownership / unattended logic
    OWNERSHIP_DISTANCE_RATIO = 0.15      # 20% of max(width, height)
    UNATTENDED_SECONDS = 5.0            # how long owner must be away before alert

    # Misc
    SAVE_ALL_FRAMES = True              # save every annotated frame as image


# -------------------------
# TRACKER
# -------------------------

class EnhancedItemTracker:
    """
    Tracks persons + items using YOLO+ByteTrack IDs.
    Assigns item owners based on proximity.
    Flags items as UNATTENDED once owner is away for UNATTENDED_SECONDS.
    Logs alerts to CSV and saves debug frames.
    """

    def __init__(self, fps, width, height, class_names):
        self.fps = fps
        self.width = width
        self.height = height
        self.class_names = class_names  # dict: id -> name

        self.unattended_frames = int(Config.UNATTENDED_SECONDS * fps)
        self.max_owner_distance = Config.OWNERSHIP_DISTANCE_RATIO * max(width, height)
        self.max_owner_distance_sq = self.max_owner_distance ** 2

        # Tracking state
        self.person_tracks = {}  # person_id -> {'pos': (cx, cy), 'last_seen': frame_idx}
        self.item_tracks = {}    # item_id   -> {'pos': (cx, cy), 'last_seen': frame_idx, 'cls_id': int}

        # Ownership
        self.owner_of = {}                # item_id -> person_id
        self.last_seen_owner_frame = {}   # item_id -> frame_idx
        self.away_since = {}              # item_id -> frame_idx when owner first disappeared

        # Item states
        self.item_state = {}              # item_id -> "NO_OWNER"/"ATTENDED"/"AWAY"/"UNATTENDED"
        self.unattended_triggered = set() # item_ids for which alert already logged

        # Alert logging
        self.alert_id = 0
        os.makedirs(Config.DEBUG_FRAME_DIR, exist_ok=True)
        self.csv_file = open(Config.LOG_CSV_PATH, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "alert_id",
            "frame_idx",
            "time_sec",
            "item_id",
            "class_name",
            "owner_id",
            "unattended_duration_sec",
            "center_x",
            "center_y",
        ])

    def update(self, frame_idx, now_ts, xyxy, cls, ids, frame):
        """
        Update tracker with current detections and frame.
        xyxy: (N,4) array
        cls:  (N,) array of class IDs
        ids:  (N,) array of track IDs (ByteTrack)
        """
        if xyxy is None or len(xyxy) == 0 or ids is None:
            # nothing to update
            return

        current_person_ids = set()
        current_item_ids = set()

        # 1) Update person + item tracks
        for i in range(len(xyxy)):
            track_id = int(ids[i])
            cid = int(cls[i])
            x1, y1, x2, y2 = xyxy[i]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            if cid == 0:
                # person
                self.person_tracks[track_id] = {
                    "pos": (cx, cy),
                    "last_seen": frame_idx,
                }
                current_person_ids.add(track_id)

            elif cid in Config.ITEM_CLASS_IDS:
                # item of interest
                self.item_tracks[track_id] = {
                    "pos": (cx, cy),
                    "last_seen": frame_idx,
                    "cls_id": cid,
                }
                current_item_ids.add(track_id)

        # 2) Assign / update ownership based on nearest person in this frame
        for item_id in current_item_ids:
            ipos = self.item_tracks[item_id]["pos"]
            best_pid = None
            best_dist_sq = self.max_owner_distance_sq

            for pid in current_person_ids:
                ppos = self.person_tracks[pid]["pos"]
                dx = ipos[0] - ppos[0]
                dy = ipos[1] - ppos[1]
                d2 = dx * dx + dy * dy
                if d2 <= best_dist_sq:
                    best_dist_sq = d2
                    best_pid = pid

            if best_pid is not None:
                self.owner_of[item_id] = best_pid
                self.last_seen_owner_frame[item_id] = frame_idx
                self.away_since[item_id] = None  # reset "away" timer

        # 3) Update item states + trigger alerts
        for item_id, item in self.item_tracks.items():
            owner_id = self.owner_of.get(item_id)

            if owner_id is None:
                self.item_state[item_id] = "NO_OWNER"
                continue

            if owner_id in current_person_ids:
                # owner currently visible in this frame
                self.item_state[item_id] = "ATTENDED"
                self.away_since[item_id] = None
            else:
                # owner not visible in this frame
                if self.away_since.get(item_id) is None:
                    self.away_since[item_id] = frame_idx

                frames_away = frame_idx - self.away_since[item_id]
                if frames_away >= self.unattended_frames:
                    self.item_state[item_id] = "UNATTENDED"
                    if item_id not in self.unattended_triggered:
                        self._log_alert(frame_idx, now_ts, item_id, frame)
                else:
                    self.item_state[item_id] = "AWAY"

    def _log_alert(self, frame_idx, now_ts, item_id, frame):
        """Log first time an item becomes UNATTENDED + save debug frame."""
        self.unattended_triggered.add(item_id)
        self.alert_id += 1

        item = self.item_tracks[item_id]
        cls_id = item["cls_id"]
        class_name = self.class_names.get(cls_id, str(cls_id))
        cx, cy = item["pos"]
        owner_id = self.owner_of.get(item_id, -1)
        away_start = self.away_since.get(item_id, frame_idx)
        unattended_duration_sec = (frame_idx - away_start) / self.fps

        # CSV row
        self.csv_writer.writerow([
            self.alert_id,
            frame_idx,
            f"{now_ts:.3f}",
            item_id,
            class_name,
            owner_id,
            f"{unattended_duration_sec:.3f}",
            f"{cx:.1f}",
            f"{cy:.1f}",
        ])

        # Save debug frame
        if frame is not None:
            fname = os.path.join(
                Config.DEBUG_FRAME_DIR,
                f"alert_{self.alert_id:03d}_frame_{frame_idx:05d}_"
                f"item{item_id}_{class_name}.jpg"
            )
            cv2.imwrite(fname, frame)

    def get_item_state(self, item_id):
        return self.item_state.get(item_id, "NO_OWNER")

    def get_item_owner(self, item_id):
        return self.owner_of.get(item_id)

    def get_unattended_count(self):
        return sum(1 for s in self.item_state.values() if s == "UNATTENDED")

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.close()


# -------------------------
# DRAWING
# -------------------------

def draw_frame(frame, tracker, xyxy, cls, ids):
    """
    Draw persons, items, ownership lines, and item state on the frame.
    """
    if xyxy is None or len(xyxy) == 0 or ids is None:
        return frame

    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    current_person_ids = set()
    current_items = []  # (item_id, cls_id, (x1,y1,x2,y2))

    # First pass: draw persons
    for i in range(len(xyxy)):
        track_id = int(ids[i])
        cid = int(cls[i])
        x1, y1, x2, y2 = map(int, xyxy[i])

        if cid == 0:
            current_person_ids.add(track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                annotated,
                f"Person {track_id}",
                (x1, max(0, y1 - 8)),
                font,
                0.5,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        elif cid in Config.ITEM_CLASS_IDS:
            current_items.append((track_id, cid, (x1, y1, x2, y2)))

    # Second pass: draw items with state + ownership
    for item_id, cid, (x1, y1, x2, y2) in current_items:
        state = tracker.get_item_state(item_id)
        owner_id = tracker.get_item_owner(item_id)
        class_name = tracker.class_names.get(cid, str(cid))

        if state == "ATTENDED":
            color = (0, 255, 0)      # green
        elif state == "AWAY":
            color = (0, 255, 255)    # yellow
        elif state == "UNATTENDED":
            color = (0, 0, 255)      # red
        else:
            color = (255, 255, 255)  # white

        label = f"{class_name}:{state}"
        if owner_id is not None:
            label += f" (Owner {owner_id})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            font,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        # Draw line from item to owner (if owner in current frame)
        if owner_id in current_person_ids and owner_id in tracker.person_tracks:
            item_cx = (x1 + x2) // 2
            item_cy = (y1 + y2) // 2
            owner_pos = tracker.person_tracks[owner_id]["pos"]
            ocx, ocy = int(owner_pos[0]), int(owner_pos[1])
            cv2.line(annotated, (item_cx, item_cy), (ocx, ocy), color, 1)

    return annotated


# -------------------------
# MAIN
# -------------------------

def main():
    # Ensure output directories exist
    os.makedirs(Config.ALL_FRAMES_DIR, exist_ok=True)
    os.makedirs(Config.DEBUG_FRAME_DIR, exist_ok=True)

    # Open video to read properties
    cap = cv2.VideoCapture(Config.INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {Config.INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Initialize YOLO model
    model = YOLO(Config.YOLO_MODEL)

    # Tracker (uses YOLO's class names)
    tracker = EnhancedItemTracker(fps, width, height, model.names)

    # Video writer
    out = cv2.VideoWriter(
        Config.OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Timeline data
    timeline_times = []
    timeline_unattended_counts = []

    frame_idx = 0

    # YOLO + ByteTrack streaming
    results_generator = model.track(
        source=Config.INPUT_VIDEO_PATH,
        conf=Config.CONF_THRESHOLD,
        classes=Config.TARGET_CLASS_IDS,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for result in results_generator:
        frame = result.orig_img
        frame_idx += 1
        now_ts = frame_idx / fps

        # Extract detections
        boxes = result.boxes
        if boxes is None or boxes.data is None or len(boxes) == 0:
            # no detections; just write original frame
            annotated = frame
            unattended_count = tracker.get_unattended_count()
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id
            if ids is None:
                # No tracking IDs; still draw detections but cannot do ownership logic
                annotated = frame
                unattended_count = tracker.get_unattended_count()
            else:
                ids = boxes.id.cpu().numpy().astype(int)

                # Update tracker
                tracker.update(frame_idx, now_ts, xyxy, cls, ids, frame)

                # Draw annotated frame
                annotated = draw_frame(frame, tracker, xyxy, cls, ids)
                unattended_count = tracker.get_unattended_count()

        # Write annotated frame to video
        out.write(annotated)

        # Save all frames as images (optional)
        if Config.SAVE_ALL_FRAMES:
            all_frame_name = os.path.join(
                Config.ALL_FRAMES_DIR,
                f"frame_{frame_idx:05d}.jpg",
            )
            cv2.imwrite(all_frame_name, annotated)

        # Update timeline
        timeline_times.append(now_ts)
        timeline_unattended_counts.append(unattended_count)

        # If you want to preview:
        # cv2.imshow("Video", annotated)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # Cleanup
    out.release()
    tracker.close()
    cv2.destroyAllWindows()

    # Timeline plot
    if timeline_times:
        plt.figure(figsize=(8, 4))
        plt.plot(timeline_times, timeline_unattended_counts, linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Number of unattended items")
        plt.title("Unattended items over time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Config.TIMELINE_PNG_PATH, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
