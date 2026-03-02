"""
Child Drawing Preprocessor Service
Production-ready module for web application integration

Usage:
    # Initialize once when server starts
    preprocessor = ChildDrawingPreprocessor()

    # Process uploaded images
    processed_image = preprocessor.process(uploaded_image_bytes)

    # Or get bytes for API response
    image_bytes = preprocessor.process_to_bytes(uploaded_image_bytes)
"""

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import io
from PIL import Image


class ChildDrawingPreprocessor:
    """
    Production-ready preprocessor for child drawings

    Features:
    - SAM-based paper detection
    - Perspective correction
    - White background normalization
    - Color enhancement for crayon drawings
    - Output ready for mood detection model

    Input formats supported:
    - File path (str)
    - Bytes (uploaded file)
    - PIL Image
    - NumPy array

    Output format:
    - NumPy array (RGB, 512×362 by default)
    """

    def __init__(self, model_type: str = "vit_b", verbose: bool = False):
        """
        Initialize the preprocessor (call once when server starts)

        Args:
            model_type: SAM model size ('vit_b', 'vit_l', or 'vit_h')
                       vit_b is recommended for speed
            verbose: Print processing steps
        """
        self.verbose = verbose
        self.model_type = model_type


        # Download and load SAM model
        checkpoint_path = self._download_checkpoint(model_type)
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000,
        )


    def process(self,
                image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
                target_width: int = 512,
                target_height: int = 362,
                texture_grain: float = 0.08) -> np.ndarray:
        """
        Main processing pipeline - call this for each uploaded image

        Args:
            image_input: Uploaded image in any supported format
            target_width: Output width (default 512)
            target_height: Output height (default 362)
            texture_grain: Texture intensity (0.0 - 0.2, default 0.08)

        Returns:
            Processed image as RGB NumPy array, ready for mood model

        Raises:
            ValueError: If image cannot be loaded
            RuntimeError: If paper detection fails
        """


        # === STEP 1: Load and convert image ===
        img = self._load_image(image_input)

        # === STEP 2: Resize if too large (for faster SAM processing) ===
        img = self._resize_if_large(img)

        # === STEP 3: Detect paper using SAM ===
        mask = self._detect_paper(img)

        # === STEP 4: Find paper corners ===
        corners = self._find_paper_corners(mask)
        if corners is None:
            raise RuntimeError("Could not find paper corners")

        # === STEP 5: Perspective correction ===
        corrected_img = self._apply_perspective_transform(img, corners)

        # Also transform mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        corrected_mask = self._apply_perspective_transform(mask_3ch, corners)
        corrected_mask = cv2.cvtColor(corrected_mask, cv2.COLOR_RGB2GRAY)

        # === STEP 6: Normalize orientation ===
        corrected_img = self._normalize_orientation(corrected_img)
        corrected_mask = self._normalize_orientation(corrected_mask)

        # === STEP 7: Erode mask ===
        kernel = np.ones((3, 3), np.uint8)
        corrected_mask = cv2.erode(corrected_mask, kernel, iterations=2)

        # === STEP 8: Apply white background ===
        paper = self._apply_white_background(corrected_img, corrected_mask)

        # === STEP 9: White balance correction ===
        paper = self._fix_white_balance(paper)

        # === STEP 10: Clean background ===
        paper = self._clean_background(paper)

        # === STEP 11: Enhance colors ===
        paper = self._enhance_colors(paper)

        # === STEP 12: Add texture ===
        paper = self._add_texture(paper, grain_intensity=texture_grain)

        # === STEP 13: Resize to target dimensions ===
        final = cv2.resize(paper, (target_width, target_height),
                           interpolation=cv2.INTER_AREA)

        # === STEP 14: Final white enforcement ===
        for _ in range(3):
            final = self._enforce_white_background(final, threshold=230)

        return final

    def process_to_bytes(self,
                         image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
                         format: str = 'JPEG',
                         **kwargs) -> bytes:
        """
        Process image and return as bytes (useful for API responses)

        Args:
            image_input: Uploaded image
            format: Output format ('JPEG', 'PNG', etc.)
            **kwargs: Additional arguments passed to process()

        Returns:
            Image bytes that can be sent over HTTP
        """
        processed = self.process(image_input, **kwargs)

        # Convert to PIL and then to bytes
        pil_img = Image.fromarray(processed)
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format=format, quality=95)
        img_bytes.seek(0)

        return img_bytes.getvalue()

    def process_to_pil(self,
                       image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
                       **kwargs) -> Image.Image:
        """
        Process image and return as PIL Image

        Args:
            image_input: Uploaded image
            **kwargs: Additional arguments passed to process()

        Returns:
            PIL Image object
        """
        processed = self.process(image_input, **kwargs)
        return Image.fromarray(processed)

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """Convert various input types to RGB numpy array"""

        # Case 1: Already a numpy array
        if isinstance(image_input, np.ndarray):
            img = image_input.copy()
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:  # Could be BGR or RGB
                # Assume BGR if from cv2, but we'll handle both
                return img
            return img

        # Case 2: File path (string or Path)
        elif isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Cannot read image from path: {image_input}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Case 3: Bytes (uploaded file from web)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image from bytes")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Case 4: PIL Image
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert('RGB'))

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _resize_if_large(self, img: np.ndarray, max_dim: int = 2048) -> np.ndarray:
        """Downscale large images for faster SAM processing"""
        if max(img.shape[:2]) > max_dim:
            scale = max_dim / max(img.shape[:2])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            return img_resized
        return img

    def _detect_paper(self, img: np.ndarray) -> np.ndarray:
        """Run SAM and find best paper mask"""
        all_masks = self.mask_generator.generate(img)

        if not all_masks:
            raise RuntimeError("No objects detected by SAM")

        # Find best paper candidate
        best_mask, metrics = self._find_best_paper_mask(all_masks, img.shape[:2])

        if best_mask is None:
            raise RuntimeError("No paper-like object detected. Please ensure the paper is clearly visible.")

        return best_mask

    def _find_best_paper_mask(self, all_masks: List[Dict],
                              img_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """Find the best paper candidate from all SAM masks"""

        paper_candidates = []

        for i, mask_dict in enumerate(all_masks):
            mask = (mask_dict['segmentation'].astype(np.uint8)) * 255
            is_paper, metrics = self._is_paper_like(mask, img_shape)

            if is_paper:
                paper_candidates.append({
                    'index': i,
                    'mask': mask,
                    'metrics': metrics,
                    'sam_area': mask_dict['area']
                })

        if not paper_candidates:
            return None, {}

        # Sort by score and return best
        paper_candidates.sort(key=lambda x: x['metrics']['score'], reverse=True)
        best = paper_candidates[0]

        return best['mask'], best['metrics']

    def _is_paper_like(self, mask: np.ndarray, img_shape: Tuple[int, int],
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

        # Check area ratio
        area_ratio = area / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False, {}

        # Check aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        if width == 0 or height == 0:
            return False, {}

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            return False, {}

        # Check rectangularity
        bbox_area = width * height
        rectangularity = area / bbox_area if bbox_area > 0 else 0
        if rectangularity < min_rectangularity:
            return False, {}

        # Check corners
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_corners = len(approx)
        if num_corners < 4 or num_corners > 8:
            return False, {}

        # Check solidity
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

    def _find_paper_corners(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the 4 corners of the paper from the mask"""

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        paper_contour = max(contours, key=cv2.contourArea)

        # Try to approximate to 4 corners
        epsilon = 0.02 * cv2.arcLength(paper_contour, True)
        approx = cv2.approxPolyDP(paper_contour, epsilon, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2)
        elif len(approx) > 4:
            # Find 4 extreme points from convex hull
            hull = cv2.convexHull(paper_contour)
            hull_points = hull.reshape(-1, 2)

            top_left = hull_points[np.argmin(hull_points.sum(axis=1))]
            bottom_right = hull_points[np.argmax(hull_points.sum(axis=1))]
            top_right = hull_points[np.argmin(hull_points[:, 0] - hull_points[:, 1])]
            bottom_left = hull_points[np.argmax(hull_points[:, 0] - hull_points[:, 1])]

            corners = np.array([top_left, top_right, bottom_right, bottom_left])
        else:
            # Fallback to bounding box
            x, y, w, h = cv2.boundingRect(paper_contour)
            corners = np.array([
                [x, y], [x + w, y], [x + w, y + h], [x, y + h]
            ])

        return corners

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: [top-left, top-right, bottom-right, bottom-left]"""

        corners = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Sum and diff methods
        s = corners.sum(axis=1)
        ordered[0] = corners[np.argmin(s)]  # top-left
        ordered[2] = corners[np.argmax(s)]  # bottom-right

        diff = np.diff(corners, axis=1)
        ordered[1] = corners[np.argmin(diff)]  # top-right
        ordered[3] = corners[np.argmax(diff)]  # bottom-left

        return ordered

    def _apply_perspective_transform(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective transformation to unwarp the paper"""

        corners = self._order_corners(corners)

        # Calculate actual width and height
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = max(width_top, width_bottom)

        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = max(height_left, height_right)

        # Define destination corners (perfect rectangle)
        dst_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Calculate and apply transformation
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        corrected = cv2.warpPerspective(img, M, (int(width), int(height)))

        return corrected

    def _normalize_orientation(self, img: np.ndarray) -> np.ndarray:
        """Force portrait → landscape orientation"""

        h, w = img.shape[:2]

        if h > w:
            # Portrait - rotate 90° clockwise to landscape
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        return img

    def _apply_white_background(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Replace all non-paper pixels with pure white (255, 255, 255)"""

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        result = np.ones_like(img, dtype=np.uint8) * 255

        paper_pixels = binary_mask == 255
        result[paper_pixels] = img[paper_pixels]

        return result

    def _fix_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Aggressive white balance correction to remove color casts"""

        # Find areas that should be white (bright areas)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        white_areas = gray > 200

        if np.sum(white_areas) == 0:
            return img

        # Calculate average color of "white" areas
        white_r = np.mean(img[white_areas, 0])
        white_g = np.mean(img[white_areas, 1])
        white_b = np.mean(img[white_areas, 2])

        # Calculate correction factors
        max_channel = max(white_r, white_g, white_b)

        if max_channel > 0:
            r_correction = 255.0 / white_r if white_r > 0 else 1.0
            g_correction = 255.0 / white_g if white_g > 0 else 1.0
            b_correction = 255.0 / white_b if white_b > 0 else 1.0

            # Limit extreme corrections
            r_correction = min(r_correction, 1.5)
            g_correction = min(g_correction, 1.5)
            b_correction = min(b_correction, 1.5)

            # Apply correction
            result = img.astype(np.float32)
            result[:, :, 0] = np.clip(result[:, :, 0] * r_correction, 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * g_correction, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * b_correction, 0, 255)

            return result.astype(np.uint8)

        return img

    def _clean_background(self, img: np.ndarray) -> np.ndarray:
        """Aggressive white background cleaning"""

        # Any pixel with all channels >= 220 becomes pure white
        white_mask = np.all(img >= 220, axis=2)
        img[white_mask] = 255
        return img

    def _enhance_colors(self, img: np.ndarray) -> np.ndarray:
        """Strong color enhancement to match dataset vibrancy"""

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # Strong saturation boost
        s_channel = s_channel.astype(np.float32)
        s_channel = s_channel * 1.6 + 20
        s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)

        # Brightness boost
        v_channel = v_channel.astype(np.float32)
        v_channel = v_channel * 1.12
        v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)

        hsv_enhanced = cv2.merge((h_channel, s_channel, v_channel))
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

        return result

    def _add_texture(self, img: np.ndarray, grain_intensity: float = 0.08) -> np.ndarray:
        """Add minimal crayon texture with strict masking"""

        h, w = img.shape[:2]
        result = img.astype(np.float32)

        # Strict masking - only very colored pixels
        drawing_mask = np.all(img < 230, axis=2)

        # Minimal grain
        grain = np.random.normal(0, 2.5, (h, w)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (3, 3), 0)

        # Apply very light grain only to colored areas
        for c in range(3):
            channel = result[:, :, c]
            channel[drawing_mask] += grain[drawing_mask] * grain_intensity
            result[:, :, c] = np.clip(channel, 0, 255)

        return result.astype(np.uint8)

    def _enforce_white_background(self, img: np.ndarray, threshold: int = 230) -> np.ndarray:
        """Force near-white pixels to pure white"""

        near_white = np.all(img >= threshold, axis=2)
        img[near_white] = 255
        return img

    def _download_checkpoint(self, model_type: str) -> str:
        """Download SAM checkpoint if needed"""

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
                checkpoint_path
            )

        return str(checkpoint_path)


# ============================================================================
# CONVENIENCE FUNCTION FOR SIMPLE USAGE
# ============================================================================

def preprocess_drawing(image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
                       target_width: int = 512,
                       target_height: int = 362) -> np.ndarray:
    """
    Convenience function for quick preprocessing without class instantiation

    WARNING: This creates a new preprocessor each time (slow!)
    For production use, instantiate ChildDrawingPreprocessor once and reuse it.

    Args:
        image_input: Image to process
        target_width: Output width
        target_height: Output height

    Returns:
        Processed image as RGB NumPy array
    """
    preprocessor = ChildDrawingPreprocessor(verbose=False)
    return preprocessor.process(image_input, target_width, target_height)