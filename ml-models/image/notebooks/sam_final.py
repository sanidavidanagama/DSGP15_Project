import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter


# =============================================================================
# GEOMETRIC FILTERING FUNCTIONS
# =============================================================================

def is_paper_like(mask: np.ndarray, img_shape: Tuple[int, int],
                  min_area_ratio: float = 0.05,
                  max_area_ratio: float = 0.85,
                  min_aspect_ratio: float = 0.5,
                  max_aspect_ratio: float = 2.5,
                  min_rectangularity: float = 0.7) -> Tuple[bool, Dict]:
    """Check if a mask represents a paper-like object"""
    img_area = img_shape[0] * img_shape[1]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, {}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    area_ratio = area / img_area
    if area_ratio < min_area_ratio:
        return False, {}
    if area_ratio > max_area_ratio:
        return False, {}

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]

    if width == 0 or height == 0:
        return False, {}

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False, {}

    bbox_area = width * height
    rectangularity = area / bbox_area if bbox_area > 0 else 0
    if rectangularity < min_rectangularity:
        return False, {}

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_corners = len(approx)
    if num_corners < 4 or num_corners > 8:
        return False, {}

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.8:
        return False, {}

    metrics = {
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
        "num_corners": num_corners,
        "solidity": solidity,
        "area_pixels": int(area),
        "score": (rectangularity + solidity) / 2
    }

    return True, metrics


def find_best_paper_mask(all_masks: List[Dict], img_shape: Tuple[int, int],
                         verbose: bool = True) -> Tuple[Optional[np.ndarray], Dict]:
    """Find the best paper candidate from all SAM masks"""

    paper_candidates = []

    for i, mask_dict in enumerate(all_masks):
        mask = (mask_dict['segmentation'].astype(np.uint8)) * 255
        is_paper, metrics = is_paper_like(mask, img_shape)

        if is_paper:
            paper_candidates.append({
                'index': i,
                'mask': mask,
                'metrics': metrics,
                'sam_area': mask_dict['area']
            })


    if not paper_candidates:
        return None, {}

    paper_candidates.sort(key=lambda x: x['metrics']['score'], reverse=True)
    best = paper_candidates[0]

    return best['mask'], best['metrics']


# PERSPECTIVE CORRECTION FUNCTIONS

def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners as: [top-left, top-right, bottom-right, bottom-left]"""
    corners = corners.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]  # top-left
    ordered[2] = corners[np.argmax(s)]  # bottom-right

    diff = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(diff)]  # top-right
    ordered[3] = corners[np.argmax(diff)]  # bottom-left

    return ordered


def find_paper_corners(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the 4 corners of the paper from the mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    paper_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(paper_contour, True)
    approx = cv2.approxPolyDP(paper_contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2)
    elif len(approx) > 4:
        hull = cv2.convexHull(paper_contour)
        hull_points = hull.reshape(-1, 2)

        top_left = hull_points[np.argmin(hull_points.sum(axis=1))]
        bottom_right = hull_points[np.argmax(hull_points.sum(axis=1))]
        top_right = hull_points[np.argmin(hull_points[:, 0] - hull_points[:, 1])]
        bottom_left = hull_points[np.argmax(hull_points[:, 0] - hull_points[:, 1])]

        corners = np.array([top_left, top_right, bottom_right, bottom_left])
    else:
        x, y, w, h = cv2.boundingRect(paper_contour)
        corners = np.array([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ])

    return corners


def correct_perspective(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to unwarp the paper
    Preserves the natural aspect ratio of the paper (doesn't force dimensions)
    """
    corners = order_corners(corners)

    # Calculate the actual width and height of the paper
    # Use maximum of top/bottom widths and left/right heights
    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    width = max(width_top, width_bottom)

    height_left = np.linalg.norm(corners[3] - corners[0])
    height_right = np.linalg.norm(corners[2] - corners[1])
    height = max(height_left, height_right)

    # Define destination corners (perfect rectangle with natural dimensions)
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)

    # Apply transformation
    corrected = cv2.warpPerspective(img, M, (int(width), int(height)))

    return corrected

# SAM PAPER DETECTOR CLASS
class SAM_PaperDetector:
    """SAM-based paper detector with geometric filtering"""

    def __init__(self, model_type="vit_b"):

        self.model_type = model_type

        checkpoint_path = self._download_checkpoint(model_type)

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,  # Optimized for speed
            pred_iou_thresh=0.90,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000,
        )


    def _download_checkpoint(self, model_type):
        checkpoint_urls = {
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        }

        cache_dir = Path.home() / ".cache" / "sam_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / f"sam_{model_type}.pth"

        if not checkpoint_path.exists():
            urllib.request.urlretrieve(
                checkpoint_urls[model_type],
                checkpoint_path,
                reporthook=self._download_progress
            )

        return str(checkpoint_path)

    # def _download_progress(self, block_num, block_size, total_size):
    #     downloaded = block_num * block_size
    #     percent = min(downloaded / total_size * 100, 100)
    #     print(f"\r   Progress: {percent:.1f}%", end='', flush=True)

    def detect_paper_smart(self, img: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Detect paper using SAM + geometric filtering"""

        all_masks = self.mask_generator.generate(img)

        if len(all_masks) == 0:
            raise RuntimeError("No objects detected by SAM")


        best_mask, metrics = find_best_paper_mask(all_masks, img_shape=img.shape[:2], verbose=verbose)

        if best_mask is None:
            raise RuntimeError("No paper-like object detected")

        return best_mask, all_masks


# ORIENTATION NORMALIZATION
def normalize_orientation(img: np.ndarray) -> np.ndarray:
    """
    Force portrait → landscape orientation
    Dataset uses landscape (512×362), so all images must be landscape
    """
    h, w = img.shape[:2]

    if h > w:
        # Portrait - rotate 90° clockwise to landscape
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img



# WHITE BALANCE CORRECTION
def fix_white_balance_aggressive(img: np.ndarray) -> np.ndarray:
    """
    AGGRESSIVE white balance correction
    Removes cyan/gray/blue color casts from background
    Forces white areas to be PURE WHITE
    """
    # Find areas that SHOULD be white (bright areas)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    white_areas = gray > 200

    if np.sum(white_areas) == 0:
        return img

    # Calculate average color of "white" areas
    white_r = np.mean(img[white_areas, 0])
    white_g = np.mean(img[white_areas, 1])
    white_b = np.mean(img[white_areas, 2])


    # Calculate correction factors to make white areas pure white (255, 255, 255)
    max_channel = max(white_r, white_g, white_b)

    if max_channel > 0:
        r_correction = 255.0 / white_r if white_r > 0 else 1.0
        g_correction = 255.0 / white_g if white_g > 0 else 1.0
        b_correction = 255.0 / white_b if white_b > 0 else 1.0

        # Limit extreme corrections
        r_correction = min(r_correction, 1.5)
        g_correction = min(g_correction, 1.5)
        b_correction = min(b_correction, 1.5)

        # Apply correction to entire image
        result = img.astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] * r_correction, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * g_correction, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * b_correction, 0, 255)

        return result.astype(np.uint8)

    return img


# BACKGROUND PROCESSING FUNCTIONS

def apply_pure_white_background(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace all non-paper pixels with PURE white (255, 255, 255)"""
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    result = np.ones_like(img, dtype=np.uint8) * 255

    paper_pixels = binary_mask == 255
    result[paper_pixels] = img[paper_pixels]

    return result


def clean_white_background_aggressive(img: np.ndarray) -> np.ndarray:
    """
    AGGRESSIVE white background cleaning
    Any pixel with all channels >= 220 becomes PURE WHITE
    """
    # Much lower threshold - more aggressive
    white_mask = np.all(img >= 220, axis=2)
    img[white_mask] = 255
    return img


def force_near_white_to_white_aggressive(img: np.ndarray, threshold: int = 230) -> np.ndarray:
    """
    AGGRESSIVE white enforcement
    Forces any near-white pixels to pure white
    """
    near_white = np.all(img >= threshold, axis=2)
    img[near_white] = 255
    return img


def enhance_crayon_colors_strong(img: np.ndarray) -> np.ndarray:
    """
    STRONG color enhancement to match dataset vibrancy
    Increased from previous version
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # STRONGER saturation boost (was 1.40 + 15, now 1.6 + 20)
    s_channel = s_channel.astype(np.float32)
    s_channel = s_channel * 1.6 + 20
    s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)

    # Increased brightness boost (was 1.08, now 1.12)
    v_channel = v_channel.astype(np.float32)
    v_channel = v_channel * 1.12
    v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)

    hsv_enhanced = cv2.merge((h_channel, s_channel, v_channel))
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    return result



# MINIMAL TEXTURE

def add_minimal_crayon_texture_strict(img: np.ndarray, grain_intensity: float = 0.08) -> np.ndarray:
    """
    Add MINIMAL crayon texture with STRICT masking
    NEVER touches white or near-white areas

    Args:
        img: Input image
        grain_intensity: Very subtle grain strength
    """
    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # STRICT masking - only very colored pixels (threshold lowered from 240 to 230)
    # This ensures texture NEVER touches background
    drawing_mask = np.all(img < 230, axis=2)

    # Count how many pixels will get texture
    texture_pixel_count = np.sum(drawing_mask)
    total_pixels = h * w
    texture_percentage = (texture_pixel_count / total_pixels) * 100

    # MINIMAL GRAIN
    grain = np.random.normal(0, 2.5, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0)

    # Apply very light grain ONLY to colored areas
    for c in range(3):
        channel = result[:, :, c]
        channel[drawing_mask] += grain[drawing_mask] * grain_intensity
        result[:, :, c] = np.clip(channel, 0, 255)

    return result.astype(np.uint8)



# COMPLETE PIPELINE


def full_pipeline_sam_aggressive(image_path, detector, debug=False, save_path="output.png",
                                 texture_grain=0.08,
                                 target_width=512, target_height=362):
    # === STEP 1: Load Image ===
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Downscale large images for faster SAM processing
    max_dimension = 2048
    if max(img.shape[:2]) > max_dimension:
        scale = max_dimension / max(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # === STEP 2: SAM Paper Detection ===
    mask, all_masks = detector.detect_paper_smart(img, verbose=True)

    # === STEP 3: Find Paper Corners ===
    corners = find_paper_corners(mask)

    if corners is None:
        raise RuntimeError("Could not find paper corners")

    # === STEP 4: PERSPECTIVE CORRECTION ===
    corrected_img = correct_perspective(img, corners)

    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    corrected_mask = correct_perspective(mask_3channel, corners)
    corrected_mask = cv2.cvtColor(corrected_mask, cv2.COLOR_RGB2GRAY)

    # === STEP 5: ORIENTATION NORMALIZATION ===
    corrected_img = normalize_orientation(corrected_img)
    corrected_mask = normalize_orientation(corrected_mask)

    # === STEP 6: Erode Mask ===
    kernel = np.ones((3, 3), np.uint8)
    corrected_mask = cv2.erode(corrected_mask, kernel, iterations=2)

    # === STEP 7: Pure White Background ===
    paper = apply_pure_white_background(corrected_img, corrected_mask)

    # === STEP 8: WHITE BALANCE CORRECTION ===
    paper = fix_white_balance_aggressive(paper)

    # === STEP 9: Background Cleaning ===
    paper = clean_white_background_aggressive(paper)

    # === STEP 10: STRONGER Color Enhancement ===
    paper = enhance_crayon_colors_strong(paper)

    # === STEP 11: STRICT Texture Application ===
    paper = add_minimal_crayon_texture_strict(paper, grain_intensity=texture_grain)

    # === STEP 12: Resize to Dataset Dimensions ===
    current_ratio = paper.shape[1] / paper.shape[0]
    target_ratio = target_width / target_height
    stretch_amount = abs(current_ratio - target_ratio) / target_ratio * 100
    final = cv2.resize(paper, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # === STEP 13: TRIPLE White Enforcement ===
    for i in range(3):
        final = force_near_white_to_white_aggressive(final, threshold=230)

    # === Save Output ===
    output_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, output_bgr)

    # === Debug Visualization ===
    if debug:
        fig = plt.figure(figsize=(20, 14))

        steps = [
            (img, "1. Original"),
            (mask, "2. SAM Mask"),
            (corrected_img, "3. Perspective Corrected"),
            (normalize_orientation(corrected_img.copy()), "4. Orientation Normalized"),
            (apply_pure_white_background(corrected_img, corrected_mask), "5. White Background"),
            (fix_white_balance_aggressive(apply_pure_white_background(corrected_img, corrected_mask)),
             "6. White Balance Fixed"),
            (enhance_crayon_colors_strong(
                fix_white_balance_aggressive(apply_pure_white_background(corrected_img, corrected_mask))),
             "7. Colors Enhanced"),
            (final, f"8. Final {target_width}×{target_height}")
        ]

        for i, (step_img, title) in enumerate(steps, 1):
            plt.subplot(2, 4, i)
            if len(step_img.shape) == 2:
                plt.imshow(step_img, cmap='gray')
            else:
                plt.imshow(step_img)
            plt.title(title, fontsize=10, fontweight='bold')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return final




# MAIN EXECUTION
if __name__ == "__main__":

    # Initialize SAM Detector (Do this ONCE)
    detector = SAM_PaperDetector(model_type='vit_b')

    try:
        final_result = full_pipeline_sam_aggressive(
            image_path="../data/black bg2.jpeg",
            detector=detector,
            debug=True,  # Show all processing steps
            save_path="../data/output/output_blackbg2.jpg",
            texture_grain=0.08,  # Minimal texture
            target_width=512,  # Dataset width
            target_height=362  # Dataset height
        )

        # Display final result
        plt.figure(figsize=(12, 8))
        plt.imshow(final_result)
        plt.title("Ready for Dataset", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback

        traceback.print_exc()
