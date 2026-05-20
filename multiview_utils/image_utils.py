import os
import numpy as np
from PIL import Image

def slice_turnaround_sheet(sheet_path, num_views=None, bg_threshold=240, target_size=(512, 512)):
    """
    Slices a horizontal turnaround sheet image into individual views.
    Automatically detects 4 or 6 views if num_views is not specified.
    """
    if not os.path.exists(sheet_path):
        raise FileNotFoundError(f"Turnaround sheet not found at: {sheet_path}")
        
    sheet = Image.open(sheet_path)
    width, height = sheet.size
    aspect_ratio = width / height
    
    # Auto-detect number of views based on aspect ratio
    if num_views is None:
        if aspect_ratio >= 5.0:
            num_views = 6
            print(f"[*] Auto-detected 6-view turnaround sheet (Aspect Ratio: {aspect_ratio:.2f})")
        else:
            num_views = 4
            print(f"[*] Auto-detected 4-view turnaround sheet (Aspect Ratio: {aspect_ratio:.2f})")
    else:
        print(f"[*] Using user-specified {num_views}-view configuration")
        
    segment_width = width / num_views
    views = []
    
    for i in range(num_views):
        left = int(i * segment_width)
        right = int((i + 1) * segment_width)
        box = (left, 0, right, height)
        
        # Crop segment
        segment = sheet.crop(box)
        
        # Clean background, center object, and pad
        processed = remove_background_and_center(segment, bg_threshold=bg_threshold, target_size=target_size)
        views.append(processed)
        
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
