import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS = "yolov8x.pt"
PLATE_WEIGHTS = "C:/CarCapital/best.pt"  # provide your plate weights file here
SAM_CHECKPOINT_PATH = "C:/CarCapital/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

input_folder = "C:/CarCapital/raw_images"
output_folder = "C:/CarCapital/output_images"
bg_path = "C:/CarCapital/Clean_Background_CC.png"
os.makedirs(output_folder, exist_ok=True)


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance_to_center(box, image_center):
    cx, cy = box_center(box)
    icx, icy = image_center
    return ((cx - icx) ** 2 + (cy - icy) ** 2) ** 0.5


def process_interior(image_rgb, output_path, sam_predictor):
    """Processes an interior image by whitening the windows using Grid-based SAM."""
    print(f"Processing {os.path.basename(output_path)} as interior image...")

    h, w = image_rgb.shape[:2]

    # 1. Generate Grid Points
    # Step size: 50 pixels (adjustable)
    step = 50
    # Only scan top 60%
    y_max = int(h * 0.6)

    grid_points = []

    # Pre-calculate HSV for fast filtering
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    for y in range(step // 2, y_max, step):
        for x in range(step // 2, w, step):
            # Check pixel color
            # Value > 190 (Bright) AND Saturation < 40 (White/Grey)
            if v_channel[y, x] > 190 and s_channel[y, x] < 40:
                grid_points.append([x, y])

    final_mask = np.zeros((h, w), dtype=np.uint8)

    if grid_points:
        print(f"Found {len(grid_points)} valid grid points.")
        sam_predictor.set_image(image_rgb)

        points_np = np.array(grid_points)

        # Transform ALL points once
        point_coords_all = sam_predictor.transform.apply_coords(points_np, (h, w))

        # Batch prediction: Treat each point as a separate prompt
        # Shape: (N, 1, 2) where N is number of points
        coords_torch = torch.as_tensor(point_coords_all, dtype=torch.float, device=sam_predictor.device)
        coords_torch = coords_torch[:, None, :]

        labels_torch = torch.ones(len(grid_points), 1, dtype=torch.int, device=sam_predictor.device)

        masks, scores, logits = sam_predictor.predict_torch(
            point_coords=coords_torch,
            point_labels=labels_torch,
            multimask_output=False,
        )

        # masks shape: (N, 1, H, W)
        # Combine all masks
        masks_np = masks.cpu().numpy().squeeze(1)  # (N, H, W)

        # Efficiently combine using max (equivalent to bitwise OR for binary masks)
        if masks_np.shape[0] > 0:
            combined_mask = np.max(masks_np, axis=0)
            final_mask = (combined_mask * 255).astype(np.uint8)

    else:
        print("No valid window points found in grid.")

    # Post-processing
    # Ensure bottom 50% is strictly clean (safety net)
    final_mask[int(h * 0.5):, :] = 0

    # Dilate slightly to close gaps between grid segments
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    interior_img = image_rgb.copy()
    interior_img[final_mask > 0] = [255, 255, 255]

    cv2.imwrite(output_path, cv2.cvtColor(interior_img, cv2.COLOR_RGB2BGR))
    print(f"Saved {os.path.basename(output_path)} (interior)")


# --- LOAD MODELS ---
print(f"Loading models on {DEVICE}...")
yolo_model = YOLO(YOLO_WEIGHTS)
plate_detector = YOLO(PLATE_WEIGHTS)
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(DEVICE)
sam_predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

if os.path.exists(bg_path):
    dealership_bg = cv2.imread(bg_path)
else:
    print(f"Warning: Background image not found at {bg_path}")
    dealership_bg = None

print("Starting processing...")

for img_path in glob.glob(os.path.join(input_folder, "*.jpg")):
    print(f"Processing {os.path.basename(img_path)}...")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Could not read: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    if dealership_bg is not None:
        resized_bg = cv2.resize(dealership_bg, (w, h))
    else:
        # Fallback if no bg image
        resized_bg = np.zeros_like(img_rgb)

    # Detect cars
    results = yolo_model(img_rgb, verbose=False)
    boxes = []
    scores = []
    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                              results[0].boxes.cls.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy()):
        if int(cls) == 2:  # car class id
            boxes.append(box)
            scores.append(conf)

    if not boxes:
        # No car detected at all, assume interior
        base_name = os.path.basename(img_path)
        out_name = os.path.splitext(base_name)[0] + "_interior_white_windows.png"
        process_interior(img_rgb, os.path.join(output_folder, out_name), sam_predictor)
        continue

    # Process exterior car images:
    image_center = (w / 2, h / 2)
    min_area = 0.2 * w * h
    filtered_boxes = []
    for box, score in zip(boxes, scores):
        area = box_area(box)
        dist = distance_to_center(box, image_center)
        if area > min_area:
            filtered_boxes.append((box, area, dist, score))

    if not filtered_boxes:
        print(f"No suitable exterior car found for {img_path} (too small or not centered).")
        # Fallback to interior processing
        base_name = os.path.basename(img_path)
        out_name = os.path.splitext(base_name)[0] + "_interior_white_windows.png"
        process_interior(img_rgb, os.path.join(output_folder, out_name), sam_predictor)
        continue

    filtered_boxes.sort(key=lambda x: (x[2], -x[1], -x[3]))
    selected_box = filtered_boxes[0][0]

    plate_results = plate_detector(img_rgb, verbose=False)
    plate_boxes = [box for box, cls in zip(plate_results[0].boxes.xyxy.cpu().numpy(),
                                           plate_results[0].boxes.cls.cpu().numpy()) if int(cls) == 0]

    sam_predictor.set_image(img_rgb)
    boxes_np = np.array([selected_box], dtype=np.float32)
    boxes_torch = torch.tensor(boxes_np, device=sam_predictor.device)
    input_boxes = sam_predictor.transform.apply_boxes_torch(boxes_torch, (h, w))

    empty_point_coords = torch.empty((1, 0, 2), device=sam_predictor.device)
    empty_point_labels = torch.empty((1, 0), dtype=torch.int, device=sam_predictor.device)

    masks, scores, logits = sam_predictor.predict_torch(
        empty_point_coords,
        empty_point_labels,
        input_boxes,
        multimask_output=False,
    )

    mask_np = masks[0].cpu().numpy()
    # Remove singleton dimensions if exist
    mask_np = np.squeeze(mask_np)

    # Convert float mask [0,1] to uint8 [0,255]
    mask_np = (mask_np * 255).astype(np.uint8)

    # Check if mask is not empty
    if mask_np.size == 0 or np.count_nonzero(mask_np) == 0:
        print("Empty mask, skipping")
        continue

    # Now apply morphology operations safely
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    blurred_mask = cv2.GaussianBlur(closed_mask, (7, 7), 0)

    # Full image alpha mask
    alpha_mask = blurred_mask.astype(np.float32) / 255

    x1, y1, x2, y2 = map(int, selected_box)

    # Ensure coordinates are within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    car_region = img_rgb[y1:y2, x1:x2]
    # Get the crop of the alpha mask corresponding to the car
    alpha_region = alpha_mask[y1:y2, x1:x2]

    # Apply plate patches
    for pbox in plate_boxes:
        px1, py1, px2, py2 = map(int, pbox)

        # Clip plate box to car box to avoid errors
        px1 = max(x1, px1)
        py1 = max(y1, py1)
        px2 = min(x2, px2)
        py2 = min(y2, py2)

        if px1 >= px2 or py1 >= py2:
            continue

        plate_crop = img_rgb[py1:py2, px1:px2]

        # Calculate relative coordinates within the car region
        rel_y1 = py1 - y1
        rel_y2 = py2 - y1
        rel_x1 = px1 - x1
        rel_x2 = px2 - x1

        car_region[rel_y1:rel_y2, rel_x1:rel_x2] = plate_crop
        alpha_region[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

    # Create 3-channel alpha mask for the CAR REGION only
    car_alpha_mask_3c = np.stack([alpha_region] * 3, axis=-1)

    # Now multiply. Shapes should match: (H_box, W_box, 3) * (H_box, W_box, 3)
    car_pixels = car_region.astype(np.float32) * car_alpha_mask_3c

    # Calculate positioning
    pos_x = (x1 + x2) // 2 - (x2 - x1) // 2
    pos_y = int(h * 0.9) - (y2 - y1)

    start_x = max(pos_x, 0)
    start_y = max(pos_y, 0)
    end_x = min(pos_x + (x2 - x1), w)
    end_y = min(pos_y + (y2 - y1), h)

    crop_x1 = start_x - pos_x
    crop_y1 = start_y - pos_y

    # Calculate dimensions of the region to update
    update_h = end_y - start_y
    update_w = end_x - start_x

    if update_h <= 0 or update_w <= 0:
        print("Target region is empty, skipping")
        continue

    background_region = resized_bg[start_y:end_y, start_x:end_x].astype(np.float32)

    # Extract the corresponding part of the car pixels and alpha mask
    car_pixels_crop = car_pixels[crop_y1:crop_y1 + update_h, crop_x1:crop_x1 + update_w]
    alpha_mask_crop = car_alpha_mask_3c[crop_y1:crop_y1 + update_h, crop_x1:crop_x1 + update_w]

    # Blend
    blended_region = car_pixels_crop + background_region * (1 - alpha_mask_crop)

    background_overlay = blended_region.clip(0, 255).astype(np.uint8)

    final_composite = resized_bg.copy()
    final_composite[start_y:end_y, start_x:end_x] = background_overlay

    base_name = os.path.basename(img_path)
    out_name = os.path.splitext(base_name)[0] + "_car_composite.png"
    cv2.imwrite(os.path.join(output_folder, out_name), cv2.cvtColor(final_composite, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_name}")
