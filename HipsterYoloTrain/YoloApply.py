# inference_on_frame.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Path to best model checkpoint
CKPT = r"runs\detect\train12\weights\best.pt"

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Toggle printing raw model outputs for each frame here
# Set to True to enable printing, False to disable.
PRINT_OUTPUT = True

# load model (Ultralytics YOLO wrapper)
model = YOLO(CKPT)

# helper: run detection on a single OpenCV BGR frame and draw boxes
def detect_and_draw(frame, conf=0.25, print_output=False):
    """
    frame: BGR image (OpenCV)
    returns: annotated BGR image
    """
    # Convert to RGB for model (Ultralytics accepts numpy RGB arrays)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use model.predict or call model() depending on API; this uses predict which is stable
    results = model.predict(source=rgb, conf=conf, device=device, verbose=False)

    if not results:
        if print_output:
            print("No results returned by model.")
        return frame

    r = results[0]  # first (and only) result

    # Some ultralytics versions store boxes as torch tensors; handle both
    if hasattr(r, "boxes") and r.boxes is not None:
        boxes_obj = r.boxes
        # xyxy: (N,4)
        xyxy = getattr(boxes_obj, "xyxy", None)
        confs = getattr(boxes_obj, "conf", None)
        cls_ids = getattr(boxes_obj, "cls", None)

        # Convert to numpy if tensors
        def to_np(x):
            if x is None:
                return None
            try:
                return x.cpu().numpy()
            except Exception:
                return np.array(x)

        xyxy = to_np(xyxy)
        confs = to_np(confs)
        cls_ids = to_np(cls_ids)

        names = getattr(model, "names", {})

        if print_output:
            # concise summary printout for debugging / pipeline integration
            n = 0 if xyxy is None else xyxy.shape[0]
            print(f"Detections: {n} boxes")
            if xyxy is not None:
                print("xyxy:\n", xyxy)
            if confs is not None:
                print("confs:\n", confs)
            if cls_ids is not None:
                print("cls_ids:\n", cls_ids)
            print("class names mapping:\n", names)

        if xyxy is None:
            return frame

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            score = float(confs[i]) if confs is not None else 0.0
            cls_id = int(cls_ids[i]) if cls_ids is not None else -1
            label = f"{names.get(cls_id, str(cls_id))} {score:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:
        # If no boxes attribute, try string-based save option (older API) or skip
        print("No boxes found in results; check ultralytics version.")
    return frame

if __name__ == "__main__":
    # Example 1: single image
    image_path = r"..\Combined_Dataset\images\train\Dream_1.jpg" 
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    out = detect_and_draw(img, conf=0.25, print_output=PRINT_OUTPUT)
    cv2.imwrite("out_image.jpg", out)
    print("Wrote out_image.jpg")

    # Example 2: webcam / video (press q to quit)
    # cap = cv2.VideoCapture(0)  # 0 -> first webcam; replace with video path to process a file
    # if cap.isOpened():
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         out = detect_and_draw(frame, conf=0.25, print_output=PRINT_OUTPUT)
    #         cv2.imshow("Detections", out)
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()