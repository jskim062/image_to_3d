import os
import numpy as np
from PIL import Image

def _center_and_pad(rgba_img, target_size=(512, 512)):
    """RGBA 이미지를 알파 채널 기준으로 크롭 → 20% 패딩 → 흰 배경 정사각형으로 변환."""
    data = np.array(rgba_img.convert("RGBA"))
    alpha = data[:, :, 3]
    mask = alpha > 10

    if not np.any(mask):
        return Image.new("RGB", target_size, (255, 255, 255))

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = rgba_img.crop((x0, y0, x1, y1))
    w, h = cropped.size

    square_size = int(max(w, h) * 1.20)
    canvas = Image.new("RGBA", (square_size, square_size), (255, 255, 255, 255))
    canvas.paste(cropped, ((square_size - w) // 2, (square_size - h) // 2),
                 mask=cropped.getchannel("A"))
    return canvas.resize(target_size, Image.Resampling.LANCZOS).convert("RGB")


def remove_background_rembg(img, target_size=(512, 512)):
    """
    rembg AI 모델로 배경을 정밀하게 제거한다.
    threshold 방식보다 훨씬 깔끔한 외곽선을 생성해 3D 형상 품질을 높인다.
    """
    from rembg import remove as rembg_remove
    rgba = rembg_remove(img.convert("RGBA"))
    return _center_and_pad(rgba, target_size)


def detect_label_split(img, bg_threshold=240):
    """
    Analyzes the horizontal projection of non-white pixels to detect
    if there is a text label at the bottom of the cell, separated by a white gap.
    Returns split_y (int) if a label is detected, else None.
    """
    img_rgba = img.convert("RGBA")
    data = np.array(img_rgba)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    
    # Binary mask of non-white pixels
    mask = (r < bg_threshold) | (g < bg_threshold) | (b < bg_threshold)
    if "A" in img_rgba.mode:
        mask = mask & (a > 10)
        
    H, W = mask.shape
    
    # Vertical profile: sum along columns
    profile = np.sum(mask, axis=1)
    
    # We only look for a split in the bottom portion of the image (between 60% and 92% of height)
    start_y = int(H * 0.60)
    end_y = int(H * 0.92)
    
    best_gap_y = None
    
    # A valid gap row should have very low pixel density (e.g., < 1% of cell width)
    max_gap_density = max(1, int(W * 0.01))
    
    # Scan for the best split point in the region
    for y in range(start_y, end_y):
        if profile[y] <= max_gap_density:
            # Verify there is actual content above and below this row
            content_above = np.any(profile[:y] > max_gap_density * 2)
            content_below = np.any(profile[y+1:] > max_gap_density * 2)
            
            if content_above and content_below:
                # Prefer gap closest to 80% height of the cell
                if best_gap_y is None or abs(y - H * 0.80) < abs(best_gap_y - H * 0.80):
                    best_gap_y = y
                    
    return best_gap_y


def slice_turnaround_sheet(sheet_path, num_views=None, bg_threshold=240,
                           target_size=(512, 512), use_rembg=False, return_labels=False):
    """
    Slices a turnaround sheet image into individual views.
    Supports both:
      1. Single-row layout (Aspect Ratio >= 3.5, e.g., 4 or 6 views side-by-side)
      2. 2-row grid layout (Aspect Ratio < 3.5, Row 1 = 4 horizontal views, Row 2 = 2 vertical views)
      
    If return_labels is True, returns (views, labels) where labels is a list of cropped
    text label images (or None if no label detected for a view).
    """
    if not os.path.exists(sheet_path):
        raise FileNotFoundError(f"Turnaround sheet not found at: {sheet_path}")
        
    sheet = Image.open(sheet_path)
    width, height = sheet.size
    aspect_ratio = width / height
    
    views = []
    labels = []

    def _process(segment):
        # Detect if there's a text label at the bottom
        split_y = detect_label_split(segment, bg_threshold=bg_threshold)
        if split_y is not None:
            w, h = segment.size
            object_segment = segment.crop((0, 0, w, split_y))
            label_segment = segment.crop((0, split_y, w, h))
            print(f"  [Label Detection] Detected text label! Splitting at y={split_y}")
        else:
            object_segment = segment
            label_segment = None

        if use_rembg:
            try:
                processed_view = remove_background_rembg(object_segment, target_size)
            except Exception as e:
                print(f"  [rembg] 실패 ({e}), threshold 방식으로 대체합니다.")
                processed_view = remove_background_and_center(object_segment, bg_threshold=bg_threshold, target_size=target_size)
        else:
            processed_view = remove_background_and_center(object_segment, bg_threshold=bg_threshold, target_size=target_size)
        return processed_view, label_segment

    if use_rembg:
        print("[*] AI 배경 제거 모드 (rembg) 활성화")

    # Detect if it is a 2-row grid layout (Aspect ratio is usually ~2.0 for a 4x2 grid)
    # If the user explicitly requested 4 views, it cannot be a 2-row grid (which has 6 views).
    is_2row_grid = aspect_ratio < 3.5 and num_views != 4

    if is_2row_grid:
        print(f"[*] Detected 2-row grid layout based on Aspect Ratio: {aspect_ratio:.2f}")
        row_height = height // 2

        segment_width = width / 4
        print("[*] Slicing Row 1 into 4 horizontal views...")
        for i in range(4):
            left = int(i * segment_width)
            right = int((i + 1) * segment_width)
            v, l = _process(sheet.crop((left, 0, right, row_height)))
            views.append(v)
            labels.append(l)

        print("[*] Slicing Row 2 into 2 vertical views (Top, Bottom)...")
        bottom_segment_width = width / 2
        for i in range(2):
            left = int(i * bottom_segment_width)
            right = int((i + 1) * bottom_segment_width)
            v, l = _process(sheet.crop((left, row_height, right, height)))
            views.append(v)
            labels.append(l)

        print(f"[+] Successfully extracted 6 views from 2-row grid layout.")
    else:
        if num_views is None:
            if aspect_ratio >= 5.0:
                num_views = 6
                print(f"[*] Auto-detected single-row 6-view turnaround sheet (Aspect Ratio: {aspect_ratio:.2f})")
            else:
                num_views = 4
                print(f"[*] Auto-detected single-row 4-view turnaround sheet (Aspect Ratio: {aspect_ratio:.2f})")
        else:
            print(f"[*] Using user-specified {num_views}-view configuration")

        segment_width = width / num_views
        print(f"[*] Slicing single-row into {num_views} views...")
        for i in range(num_views):
            left = int(i * segment_width)
            right = width if i == num_views - 1 else int((i + 1) * segment_width)
            v, l = _process(sheet.crop((left, 0, right, height)))
            views.append(v)
            labels.append(l)

    if return_labels:
        return views, labels
    return views

def remove_background_and_center(img, bg_threshold=240, target_size=(512, 512)):
    """
    Finds the bounding box of non-background pixels, crops the object tightly,
    pads it by 10% to prevent border clipping, centers it, and resizes to a square.
    """
    img = img.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    
    # Create mask where pixels are NOT white/light gray
    mask = (r < bg_threshold) | (g < bg_threshold) | (b < bg_threshold)
    
    # If the segment is completely white or empty, return centered empty white image
    if not np.any(mask):
        return Image.new("RGB", target_size, (255, 255, 255))
        
    # Get bounding box coordinates of the subject
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    # Crop the subject
    cropped = img.crop((x0, y0, x1, y1))
    w, h = cropped.size
    
    # Add a 10% protective margin to prevent edge-stretching in 3D projection
    max_dim = max(w, h)
    square_size = int(max_dim * 1.20)
    
    # Create clean white canvas
    centered_img = Image.new("RGBA", (square_size, square_size), (255, 255, 255, 255))
    
    # Paste subject in the center
    offset_x = (square_size - w) // 2
    offset_y = (square_size - h) // 2
    
    # Use alpha channel as mask if available
    alpha_mask = cropped.getchannel("A") if "A" in cropped.mode else None
    centered_img.paste(cropped, (offset_x, offset_y), mask=alpha_mask)
    
    # Resize to standardized square target and return as RGB
    return centered_img.resize(target_size).convert("RGB")

def match_histograms(source, template):
    """
    Adjust the pixel values of an RGB source image to match the histogram
    of an RGB template image. Eliminates lighting/shadow gradients.
    """
    src_arr = np.array(source)
    tmpl_arr = np.array(template)
    
    matched = np.zeros_like(src_arr)
    
    for channel in range(3):
        src_c = src_arr[:, :, channel]
        tmpl_c = tmpl_arr[:, :, channel]
        
        # Calculate source and template histograms
        src_values, src_unique_indices, src_counts = np.unique(src_c, return_inverse=True, return_counts=True)
        tmpl_values, tmpl_counts = np.unique(tmpl_c, return_counts=True)
        
        # Calculate cumulative sums
        src_quantiles = np.cumsum(src_counts).astype(np.float64) / src_c.size
        tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float64) / tmpl_c.size
        
        # Interpolate source values to match template quantiles
        interp_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        matched[:, :, channel] = interp_values[src_unique_indices].reshape(src_c.shape)
        
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))

def align_multiview_colors(views):
    """
    Aligns the lighting and color profiles of all views in a list to match
    the canonical Front view (first image in the list).
    """
    if not views:
        return []
        
    front_view = views[0]
    aligned_views = [front_view]
    
    for i in range(1, len(views)):
        aligned = match_histograms(views[i], front_view)
        aligned_views.append(aligned)
        
    return aligned_views
