###############################################
# LOST & FOUND - MAIN PIPELINE FOR VS CODE
# End-to-end execution (no notebook required)
###############################################

import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

###############################################
# CONFIGURATION
###############################################

VIDEO_PATH = "sania_test1.mp4"       # <<< CHANGE HERE
OUTPUT_PATH = "output_detected_sania_test1.mp4"
UNATTENDED_THRESHOLD = 5                         # seconds
TARGET_CLASSES = ["backpack", "handbag", "suitcase", "laptop"]

###############################################
# LOAD YOLO MODEL
###############################################

print("[INFO] Loading YOLO model...")
model = YOLO("yolov8x.pt")   # use your trained model if custom

###############################################
# UTILITY FUNCTIONS
###############################################

def iou(boxA, boxB):
    """Intersection over Union for spatial matching."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    if interArea == 0:
        return 0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


###############################################
# OWNER–OBJECT MAPPING STATE
###############################################

object_owner = {}                  # object_id → person_id
object_last_seen_with_owner = {}   # timestamp
object_last_seen_time = {}         # last visible frame time
object_unattended_flag = {}        # boolean

next_object_id = 1                 # incremental unique ID

###############################################
# VIDEO SETUP
###############################################

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

print("[INFO] Processing video...")
start_time = time.time()

###############################################
# MAIN LOOP
###############################################

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_timestamp = time.time() - start_time

    # --------------------------------------
    # YOLO PREDICTION
    # --------------------------------------
    results = model(frame)[0]

    persons = []
    objects = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)

        if cls_name == "person":
            persons.append((x1, y1, x2, y2))

        if cls_name in TARGET_CLASSES:
            objects.append((x1, y1, x2, y2, cls_name))

    # --------------------------------------
    # ASSOCIATE OBJECTS TO PEOPLE (SPATIAL IOU)
    # --------------------------------------
    for obj in objects:
        x1, y1, x2, y2, obj_class = obj

        best_iou = 0
        best_person = None

        for p_idx, person_box in enumerate(persons):
            score = iou((x1, y1, x2, y2), person_box)
            if score > best_iou:
                best_iou = score
                best_person = p_idx

        # Assign object ID
        if obj not in object_owner:
            object_owner[obj] = best_person
            object_last_seen_with_owner[obj] = current_timestamp
            object_unattended_flag[obj] = False

        # Update last time object seen
        object_last_seen_time[obj] = current_timestamp

        # Draw the object box (yellow)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 255), 2)
        cv2.putText(frame, f"{obj_class} (Owner {best_person})",
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

    # --------------------------------------
    # DETECT UNATTENDED ITEMS
    # --------------------------------------
    for obj in list(object_owner.keys()):
        last_seen = object_last_seen_time.get(obj, None)

        if last_seen is None:
            continue

        time_since_seen = current_timestamp - last_seen

        owner_idx = object_owner[obj]

        # If owner is no longer in frame
        if owner_idx is None or owner_idx >= len(persons):
            # Object possibly unattended
            if time_since_seen > UNATTENDED_THRESHOLD:
                object_unattended_flag[obj] = True

        # Highlight unattended objects in RED
        if object_unattended_flag[obj]:
            x1, y1, x2, y2, cls_name = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), 3)
            cv2.putText(frame, "UNATTENDED!", (int(x1), int(y1) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    # --------------------------------------
    # WRITE FRAME TO OUTPUT VIDEO
    # --------------------------------------
    out.write(frame)

    # Debug view (optional)
    cv2.imshow("Lost & Found Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


###############################################
# CLEANUP
###############################################
cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] Processing complete.")
print(f"[INFO] Output saved at: {OUTPUT_PATH}")
