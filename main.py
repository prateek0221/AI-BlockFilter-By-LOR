# import os
# import cv2
# import shutil
# import torch
# import json
# from ultralytics import YOLO
# from datetime import datetime

# def log(msg: str):
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# def is_cuda_available():
#     return torch.cuda.is_available()

# def validate_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         log(f"❌ Cannot open video: {video_path}")
#         return False
#     cap.release()
#     return True

# def person_crossed_lor(results, lors):
#     """Check if any detected person crosses ANY of the provided LORs."""
#     try:
#         for frame_result in results:
#             if frame_result.boxes is None or len(frame_result.boxes.cls) == 0:
#                 continue
#             for box, cls in zip(frame_result.boxes.xyxy, frame_result.boxes.cls):
#                 if int(cls) != 0:  # only person class
#                     continue
#                 x1, y1, x2, y2 = box.tolist()
#                 cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # bbox center

#                 for lor in lors:
#                     lor_type = lor.get("type")
#                     lor_value = lor.get("value")
#                     if lor_type == "y" and cy > lor_value:
#                         return True
#                     if lor_type == "x" and cx > lor_value:
#                         return True
#         return False
#     except Exception as e:
#         log(f"⚠️ LOR check error: {e}")
#         return False

# def contains_person(video_path, model, device, lors=None):
#     try:
#         log(f"🔎 Analyzing: {os.path.basename(video_path)}")
#         results = model.predict(video_path, stream=True, device=device, verbose=False)

#         person_found = False
#         lor_crossed = False

#         for frame_result in results:
#             if frame_result.boxes is not None and len(frame_result.boxes.cls) > 0:
#                 if any(int(cls) == 0 for cls in frame_result.boxes.cls):  # person detected
#                     person_found = True

#             if lors:
#                 if person_crossed_lor([frame_result], lors):
#                     lor_crossed = True

#         if not person_found:
#             return False
#         if lors:
#             return lor_crossed
#         return True

#     except Exception as e:
#         log(f"⚠️ Detection error: {e}")
#         return False

# def process_videos(input_root, output_root, camera_folders, lor_config):
#     device = 0 if is_cuda_available() else 'cpu'
#     log(f"🚀 Loading YOLOv8 model on {'CUDA' if device == 0 else 'CPU'}...")

#     try:
#         model = YOLO('yolov8n.pt')
#     except Exception as e:
#         log(f"❌ Failed to load YOLO model: {e}")
#         return

#     total_videos = 0
#     retained_videos = 0

#     for cam_folder in camera_folders:
#         cam_path = os.path.join(input_root, cam_folder)
#         if not os.path.isdir(cam_path):
#             log(f"⚠️ Camera folder not found: {cam_path}")
#             continue

#         # Get LORs for this camera
#         lors = None
#         if lor_config and cam_folder in lor_config:
#             lors = lor_config[cam_folder]
#             log(f"📏 Using {len(lors)} LOR(s) for camera {cam_folder}: {lors}")

#         log(f"📷 Processing camera folder: {cam_folder}")
#         output_cam_path = os.path.join(output_root, cam_folder)
#         os.makedirs(output_cam_path, exist_ok=True)

#         video_files = [f for f in os.listdir(cam_path) if f.lower().endswith(".mp4")]
#         if not video_files:
#             log("⚠️ No video files found in this folder.")
#             continue

#         for video_file in sorted(video_files):
#             total_videos += 1
#             input_video_path = os.path.join(cam_path, video_file)

#             if not validate_video(input_video_path):
#                 continue

#             if contains_person(input_video_path, model, device, lors):
#                 try:
#                     shutil.copy2(input_video_path, os.path.join(output_cam_path, video_file))
#                     log(f"💾 Saved: {video_file}")
#                     retained_videos += 1
#                 except Exception as copy_err:
#                     log(f"❌ Failed to copy {video_file}: {copy_err}")
#             else:
#                 log(f"❌ No valid person/LOR in: {video_file}")

#     log("\n✅ Processing complete.")
#     log(f"📊 Total videos scanned: {total_videos}")
#     log(f"📦 Videos retained: {retained_videos}")
#     log(f"🗑️ Videos skipped: {total_videos - retained_videos}")

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Filter videos by person detection + per-camera multi-LOR crossing using YOLOv8.")
#     parser.add_argument("--input", required=True, help="Input root folder containing camera subfolders")
#     parser.add_argument("--output", required=True, help="Output folder for retained videos")
#     parser.add_argument("--cameras", nargs="+", required=True, help="Camera folders (e.g., 48 8 39)")
#     parser.add_argument("--lor_config", help="Path to JSON file with LOR config per camera")

#     args = parser.parse_args()

#     lor_config = None
#     if args.lor_config and os.path.isfile(args.lor_config):
#         with open(args.lor_config, "r") as f:
#             lor_config = json.load(f)

#     try:
#         process_videos(args.input, args.output, args.cameras, lor_config)
#     except KeyboardInterrupt:
#         log("🛑 Interrupted by user.")
#     except Exception as e:
#         log(f"❗ Unexpected error: {e}")


import os
import cv2
import json
import torch
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ---------------- Logger ----------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ---------------- LOR helpers ----------------
def load_lors(filepath):
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_lors(lors, filepath):
    if not filepath:
        return
    with open(filepath, "w") as f:
        json.dump(lors, f, indent=2)

def line_crossed(point, line, tol=10):
    """Check if a point (px,py) is within tol pixels of a line segment (x1,y1,x2,y2)."""
    px, py = point
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    seg_len2 = dx*dx + dy*dy
    if seg_len2 == 0:
        return False
    t = max(0.0, min(1.0, ((px - x1)*dx + (py - y1)*dy) / seg_len2))
    qx, qy = x1 + t*dx, y1 + t*dy
    dist = ((px - qx)**2 + (py - qy)**2) ** 0.5
    return dist < tol

# ---------------- Device management ----------------
class DeviceManager:
    def __init__(self, force_cpu=False):
        self.force_cpu = force_cpu
        self.device = self._choose_device()

    def _choose_device(self):
        if self.force_cpu:
            log("🧠 Forcing CPU as requested.")
            return "cpu"
        if torch.cuda.is_available():
            try:
                _ = torch.cuda.device_count()
                log("🚀 Using CUDA")
                return 0
            except Exception as e:
                log(f"⚠️ CUDA available but not usable ({e}). Falling back to CPU.")
        log("🧠 Using CPU")
        return "cpu"

    def fallback_to_cpu(self):
        if self.device != "cpu":
            log("↩️ CUDA error detected. Falling back to CPU for the rest of the run.")
            self.device = "cpu"

# ---------------- Mouse interaction ----------------
drawing_state = {"clicks": [], "done": False, "camera_id": None, "lors": None, "lor_file": None}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_state["done"]:
        drawing_state["clicks"].append((x, y))
        if len(drawing_state["clicks"]) == 2:
            # Save as LOR
            (x1, y1), (x2, y2) = drawing_state["clicks"]
            drawing_state["lors"][drawing_state["camera_id"]] = [[x1, y1, x2, y2]]
            save_lors(drawing_state["lors"], drawing_state["lor_file"])
            log(f"🖊️  LOR set for camera {drawing_state['camera_id']}: {(x1,y1)} → {(x2,y2)}")
            drawing_state["done"] = True

# ---------------- Video filter ----------------
def contains_person_and_lor(video_path, model, devmgr, lors, show=False, lor_file=None, use_half=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"❌ Failed to open {video_path}")
        return False

    camera_id = Path(video_path).parent.name
    lor_lines = lors.get(camera_id, [])

    # If no LOR exists yet, let user draw one on the first video
    if not lor_lines and show:
        log(f"✏️  No LOR found for camera {camera_id}. Please draw one by clicking two points.")
        drawing_state.update({"clicks": [], "done": False, "camera_id": camera_id, "lors": lors, "lor_file": lor_file})
        cv2.namedWindow("Set LOR")
        cv2.setMouseCallback("Set LOR", mouse_callback)

        while not drawing_state["done"]:
            ret, frame = cap.read()
            if not ret:
                break
            # Show temporary line if one click done
            if len(drawing_state["clicks"]) == 1:
                cv2.circle(frame, drawing_state["clicks"][0], 4, (0, 255, 255), -1)
            elif len(drawing_state["clicks"]) == 2:
                cv2.line(frame, drawing_state["clicks"][0], drawing_state["clicks"][1], (255, 0, 0), 2)

            cv2.imshow("Set LOR", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow("Set LOR")
        lor_lines = lors.get(camera_id, [])

        # Reset cap for detection
        cap.release()
        cap = cv2.VideoCapture(video_path)

    person_found, lor_crossed = False, False
    half = bool(use_half and devmgr.device != "cpu")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            results = model.predict(frame, device=devmgr.device, classes=[0], verbose=False, half=half)
        except RuntimeError as e:
            if "CUDA" in str(e) or "cublas" in str(e).lower() or "device-side" in str(e).lower():
                devmgr.fallback_to_cpu()
                half = False
                try:
                    results = model.predict(frame, device="cpu", classes=[0], verbose=False, half=False)
                except Exception as e2:
                    log(f"❌ Inference failed on CPU after CUDA fallback: {e2}")
                    break
            else:
                log(f"❌ Inference failed: {e}")
                break

        r0 = results[0]
        boxes = r0.boxes
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            person_found = True
            xyxy = boxes.xyxy.detach().cpu().numpy()
            for b in xyxy:
                x1, y1, x2, y2 = b.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
                for line in lor_lines:
                    lx1, ly1, lx2, ly2 = line
                    if line_crossed((cx, cy), (lx1, ly1, lx2, ly2)):
                        lor_crossed = True

        for line in lor_lines:
            lx1, ly1, lx2, ly2 = line
            cv2.line(frame, (lx1, ly1), (lx2, ly2), (255,0,0), 2)
        if lor_crossed:
            cv2.putText(frame, "CROSSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        if show:
            cv2.imshow("Video", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        if person_found and lor_crossed:
            break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    return person_found and lor_crossed

# ---------------- Processing loop ----------------
def process_videos(input_folder, output_folder, cameras, lor_file, show, force_cpu=False, half=False):
    devmgr = DeviceManager(force_cpu=force_cpu)
    log(f"🔧 Selected device: {'CUDA' if devmgr.device != 'cpu' else 'CPU'}")

    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        log(f"❌ Failed to load YOLO: {e}")
        return

    lors = load_lors(lor_file)
    total, retained, skipped = 0, 0, 0

    for cam in cameras:
        cam_folder = Path(input_folder) / str(cam)
        if not cam_folder.exists():
            log(f"⚠️ Camera folder missing: {cam_folder}")
            continue
        out_cam_folder = Path(output_folder) / str(cam)
        out_cam_folder.mkdir(parents=True, exist_ok=True)

        video_files = sorted([p for p in cam_folder.glob("*.mp4")])
        if not video_files:
            log(f"⚠️ No .mp4 files in {cam_folder}")
            continue

        log(f"📷 Processing camera {cam} with {len(lors.get(str(cam), []))} LOR(s)")
        for video_path in video_files:
            total += 1
            ok = contains_person_and_lor(str(video_path), model, devmgr, lors, show=show, lor_file=lor_file, use_half=half)
            if ok:
                try:
                    shutil.copy2(str(video_path), out_cam_folder / video_path.name)
                    log(f"💾 Saved: {video_path.name}")
                    retained += 1
                except Exception as copy_err:
                    log(f"❌ Copy failed for {video_path.name}: {copy_err}")
                    skipped += 1
            else:
                log(f"❌ Rejected: {video_path.name}")
                skipped += 1

    log("✅ Processing complete.")
    log(f"📊 Total videos scanned: {total}")
    log(f"📦 Videos retained (person + LOR crossed): {retained}")
    log(f"🗑️ Videos skipped: {skipped}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter videos by person & LOR crossing with live adjust and robust CUDA fallback.")
    parser.add_argument("--input", required=True, help="Input folder path containing camera subfolders")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--cameras", nargs="+", required=True, help="Camera folder names (e.g., 48 8 39)")
    parser.add_argument("--lors", default="lors.json", help="JSON file for LOR definitions per camera")
    parser.add_argument("--show", action="store_true", help="Show live video and allow LOR adjustment / setup")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--half", action="store_true", help="Use half precision on CUDA (ignored on CPU)")
    args = parser.parse_args()

    process_videos(args.input, args.output, args.cameras, args.lors, args.show, force_cpu=args.force_cpu, half=args.half)
