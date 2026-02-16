"""
TrueWordCloud - Value-Proportional Word Cloud Generator
========================================================

A word cloud generator that maintains TRUE proportional relationships between values.
Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud
ensures font sizes are ALWAYS proportional to the input values.

Key Features:
- Font sizes are ALWAYS proportional to input values (no squeezing/normalization)
- Three layout algorithms: 'greedy' (fast, deterministic), 'square' (compact, randomized),
  and 'distance_transform' (compact packing using distance transform)
- Dynamic canvas sizing (content determines size, not pre-fixed dimensions)
- Mask support (black allowed, white forbidden), with unified mask-scaling attempts across all methods
"""

from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_closing
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import math
import random
import numpy as np
import os
import platform


@dataclass
class WordBox:
    """Represents a word with its rendering properties and position"""

    word: str
    frequency: float
    font_size: int
    width: int
    height: int
    x: float  # Center position (wordcloud coords)
    y: float
    color: Tuple[int, int, int] = (0, 0, 0)
    bbox_offset_x: int = 0  # font.getbbox left offset
    bbox_offset_y: int = 0  # font.getbbox top offset

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """
        Return the *ink* bounding box (left, top, right, bottom) in wordcloud coords.
        This matches the rectangle used during drawing (accounts for font.getbbox offsets).
        """
        left = self.x - self.width / 2 - self.bbox_offset_x
        top = self.y - self.height / 2 - self.bbox_offset_y
        return (left, top, left + self.width, top + self.height)


class TrueWordCloud:
    """
    Value-proportional word cloud generator.

    Methods:
      - greedy: spiral search, deterministic-ish (angle step depends on size)
      - square: compact-ish packing, randomized
      - distance_transform: DT-based packing, compact-ish

    Mask convention (all methods):
      - black (0)   = allowed
      - white (255) = forbidden
    """

    def __init__(
        self,
        values: Dict[str, float],
        method: str = "distance_transform",
        base_font_size: int = 100,
        font_path: Optional[str] = None,
        min_font_size: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        margin: int = 2,
        color_func: Optional[callable] = None,
        angle_divisor: float = 3.0,
        seed: Optional[int] = None,
        max_attempts: int = 20,
        scale_factor: float = 1.2,
        use_mask_colors: bool = False,
        mask_color_mode: str = "mean_ink",
        mask_shape_mode: str = "no-colors",
    ):
        self.use_mask_colors = use_mask_colors
        self.mask_shape_mode = mask_shape_mode  # "black_allowed" | "nonwhite_allowed"
        self.mask_color_mode = mask_color_mode  # "center" | "mean_ink"
        self._mask_used_color_arr = None
        self.rng = random.Random(seed)
        self.values = values
        self.method = method.lower()
        if self.method not in ["greedy", "square", "distance_transform"]:
            raise ValueError(
                f"method must be 'greedy', 'square', or 'distance_transform', got '{method}'"
            )

        self.base_font_size = base_font_size
        self.font_path = font_path or self._get_default_font()
        self.min_font_size = min_font_size
        self.background_color = background_color
        self.margin = margin
        self.color_func = color_func or self._default_color_func
        self.angle_divisor = angle_divisor

        # Unified attempts for mask-scaling (and for DT even without a mask image)
        self.max_attempts = max_attempts
        self.scale_factor = scale_factor

        # Outputs/debug
        self.words: List[WordBox] = []
        self.failure: Optional[dict] = None
        self._last_bbox = (0, 0, 0, 0)
        self._last_canvas_size = (0, 0)
        self._last_offsets = (0, 0, 0)

        # The mask actually used for the final successful (or last) attempt
        self._mask_used_img: Optional[PILImage.Image] = None
        self._mask_used_allowed: Optional[np.ndarray] = None

        # Track the number of attempts used in the last generation
        self._last_attempts = 0

        # Glyph cache
        self._glyph_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def reseed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def _get_default_font(self) -> str:
        """Find a suitable default TrueType font path for the system."""
        system = platform.system()
        font_paths = []

        if system == "Windows":
            font_paths = [
                os.path.join(
                    os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"
                ),
                os.path.join(
                    os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "segoeui.ttf"
                ),
                os.path.join(
                    os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "calibri.ttf"
                ),
                os.path.join(
                    os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "verdana.ttf"
                ),
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSDisplay.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        else:  # Linux and others
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]

        for path in font_paths:
            if os.path.exists(path):
                return path

        # Fallback: PIL's default bitmap font (not scalable, but always available)
        # Return a special marker string to indicate default font usage
        return "__PIL_DEFAULT__"

    def _default_color_func(
        self, word: str, freq: float, norm_freq: float
    ) -> Tuple[int, int, int]:
        return (0, 0, 0)

    def _calculate_font_sizes(self) -> Dict[str, int]:
        if not self.values:
            return {}

        max_value = max(self.values.values())
        if max_value == 0:
            return {word: self.min_font_size for word in self.values}

        font_sizes: Dict[str, int] = {}
        for word, value in self.values.items():
            size = int(self.base_font_size * (value / max_value))
            font_sizes[word] = max(size, self.min_font_size)
        return font_sizes

    def _measure_words(self, font_sizes: Dict[str, int]) -> List[WordBox]:
        word_boxes: List[WordBox] = []
        max_freq = max(self.values.values())

        for word, font_size in font_sizes.items():
            frequency = self.values[word]
            norm_freq = frequency / max_freq

            if self.font_path == "__PIL_DEFAULT__":
                font = PILImageFont.load_default()
                # PIL's default font is fixed size (10x10), so bbox is always (0,0,6*len(word),11)
                bbox = font.getbbox(word)
            else:
                font = PILImageFont.truetype(self.font_path, font_size)
                bbox = font.getbbox(word)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            color = self.color_func(word, frequency, norm_freq)

            word_boxes.append(
                WordBox(
                    word=word,
                    frequency=frequency,
                    font_size=font_size,
                    width=width,
                    height=height,
                    x=0.0,
                    y=0.0,
                    color=color,
                    bbox_offset_x=bbox[0],
                    bbox_offset_y=bbox[1],
                )
            )

        word_boxes.sort(key=lambda wb: wb.frequency, reverse=True)
        return word_boxes

    def _clone_word_boxes(self, word_boxes: List[WordBox]) -> List[WordBox]:
        """Fresh boxes per attempt so old positions don't leak across attempts."""
        clones: List[WordBox] = []
        for wb in word_boxes:
            clones.append(
                WordBox(
                    word=wb.word,
                    frequency=wb.frequency,
                    font_size=wb.font_size,
                    width=wb.width,
                    height=wb.height,
                    x=0.0,
                    y=0.0,
                    color=wb.color,
                    bbox_offset_x=wb.bbox_offset_x,
                    bbox_offset_y=wb.bbox_offset_y,
                )
            )
        return clones

    # ----------------------------
    # Glyph masks (ink-aware mask fitting and stamping)
    # ----------------------------

    def _get_glyph_mask(self, word_box: WordBox) -> np.ndarray:
        """
        Returns a boolean mask (h,w) where True means 'ink pixel' for this word.
        Cached by (word, font_size).
        """
        key = (word_box.word, word_box.font_size)
        if key in self._glyph_cache:
            return self._glyph_cache[key]

        if self.font_path == "__PIL_DEFAULT__":
            font = PILImageFont.load_default()
        else:
            font = PILImageFont.truetype(self.font_path, word_box.font_size)
        w, h = word_box.width, word_box.height

        img = PILImage.new("L", (w, h), 0)
        draw = PILImageDraw.Draw(img)

        draw.text(
            (-word_box.bbox_offset_x, -word_box.bbox_offset_y),
            word_box.word,
            font=font,
            fill=255,
        )

        glyph = np.array(img) > 0
        self._glyph_cache[key] = glyph
        return glyph

    def _glyph_fits_mask_px(
        self, word_box: WordBox, cx: int, cy: int, mask_allowed: np.ndarray
    ) -> bool:
        """
        Check if the word's ink pixels fit inside allowed mask pixels,
        using mask pixel coords (cx,cy) as the intended *center*.
        """
        h, w = mask_allowed.shape

        left = int(cx - word_box.width / 2 - word_box.bbox_offset_x)
        top = int(cy - word_box.height / 2 - word_box.bbox_offset_y)
        right = left + word_box.width
        bottom = top + word_box.height

        if left < 0 or top < 0 or right > w or bottom > h:
            return False

        region = mask_allowed[top:bottom, left:right]
        glyph = self._get_glyph_mask(word_box)
        return region[glyph].all()

    def _glyph_fits_mask(
        self, word_box: WordBox, x: float, y: float, mask_allowed: np.ndarray
    ) -> bool:
        """
        Check if the word's ink pixels fit inside allowed mask pixels,
        using wordcloud coords (x,y) with mask centered at (0,0).
        """
        mask_h, mask_w = mask_allowed.shape
        mask_cx, mask_cy = mask_w // 2, mask_h // 2

        cx = int(round(mask_cx + x))
        cy = int(round(mask_cy + y))
        return self._glyph_fits_mask_px(word_box, cx, cy, mask_allowed)

    def _stamp_word_on_mask_px(
        self, word_box: WordBox, cx: int, cy: int, mask_working: np.ndarray
    ) -> None:
        """
        Mark the word's ink pixels as occupied (set to False) in mask_working.
        Assumes bounds are already valid.
        """
        left = int(cx - word_box.width / 2 - word_box.bbox_offset_x)
        top = int(cy - word_box.height / 2 - word_box.bbox_offset_y)
        right = left + word_box.width
        bottom = top + word_box.height

        region = mask_working[top:bottom, left:right]
        glyph = self._get_glyph_mask(word_box)

        # Only stamp ink pixels (frees space compared to stamping the full rectangle)
        region[glyph] = False
        mask_working[top:bottom, left:right] = region

    def _apply_colors_from_mask(
        self,
        placed_words: List[WordBox],
        mask_color_arr: np.ndarray,
        mask_allowed: Optional[np.ndarray] = None,
        *,
        strategy: str = "median",  # "median" or "mean"
    ) -> None:
        """
        Assign word_box.color by sampling the resized color mask under the word.

        Coordinates:
        - placed_words are in wordcloud coords centered at (0,0)
        - mask_color_arr is in mask pixel coords [0..w-1, 0..h-1]
        - we map via mask center (mask_cx, mask_cy)

        If mask_allowed is provided, we only sample pixels that are allowed (True).
        We also sample only glyph 'ink' pixels for better color fidelity.
        """
        if mask_color_arr is None:
            return

        h, w, _ = mask_color_arr.shape
        mask_cx, mask_cy = w // 2, h // 2

        for wb in placed_words:
            # Region in mask pixel coords corresponding to this word's ink bbox
            left = int(mask_cx + wb.x - wb.width / 2 - wb.bbox_offset_x)
            top = int(mask_cy + wb.y - wb.height / 2 - wb.bbox_offset_y)
            right = left + wb.width
            bottom = top + wb.height

            # Skip if out of bounds (shouldn't happen if placement used same checks, but safe)
            if left < 0 or top < 0 or right > w or bottom > h:
                continue

            region_rgb = mask_color_arr[top:bottom, left:right, :]  # (hh, ww, 3)

            # Glyph mask: True where ink exists
            glyph = self._get_glyph_mask(wb)  # shape (hh, ww)

            # Optionally restrict to allowed region too
            if mask_allowed is not None:
                allowed_region = mask_allowed[top:bottom, left:right]
                sel = glyph & allowed_region
            else:
                sel = glyph

            if not np.any(sel):
                # Fallback: sample center pixel
                cx = int(mask_cx + wb.x)
                cy = int(mask_cy + wb.y)
                if 0 <= cx < w and 0 <= cy < h:
                    wb.color = tuple(int(c) for c in mask_color_arr[cy, cx])
                continue

            pixels = region_rgb[sel]  # (N,3)

            if strategy == "mean":
                rgb = np.mean(pixels, axis=0)
            else:
                rgb = np.median(pixels, axis=0)

            wb.color = tuple(int(c) for c in rgb)

    # ----------------------------
    # Overlap + bbox helpers
    # ----------------------------

    def _boxes_overlap(self, box1: WordBox, box2: WordBox) -> bool:
        """Check if two word boxes overlap (including margin) using ink bbox."""
        left1, top1, right1, bottom1 = box1.bbox
        left2, top2, right2, bottom2 = box2.bbox

        left1 -= self.margin
        top1 -= self.margin
        right1 += self.margin
        bottom1 += self.margin

        left2 -= self.margin
        top2 -= self.margin
        right2 += self.margin
        bottom2 += self.margin

        return not (
            right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1
        )

    def _calculate_bounding_box(
        self, word_boxes: List[WordBox]
    ) -> Tuple[float, float, float, float]:
        if not word_boxes:
            return (0, 0, 0, 0)

        min_x = min(wb.bbox[0] for wb in word_boxes)
        min_y = min(wb.bbox[1] for wb in word_boxes)
        max_x = max(wb.bbox[2] for wb in word_boxes)
        max_y = max(wb.bbox[3] for wb in word_boxes)

        return (min_x, min_y, max_x, max_y)

    # ----------------------------
    # Unified attempts + mask scaling
    # ----------------------------

    def _estimate_initial_mask_size(self, word_boxes: List[WordBox], mask_img=None):
        """
        Estimate initial mask/canvas size from total rectangle area and mask aspect ratio.
        """
        total_area = sum(wb.width * wb.height for wb in word_boxes)
        if mask_img is not None:
            aspect = mask_img.width / mask_img.height
            h = int((total_area / aspect) ** 0.5)
            w = int(h * aspect)
        else:
            w = h = int(total_area**0.5)

        return max(w, 1), max(h, 1)

    def _make_working_mask(self, boxes, mask_img, w, h):
        """
        Returns:
        mask_resized_img: resized PIL image (RGBA for colors mode, L for no-colors mode)
        mask_allowed:     bool array (h,w) True=allowed
        mask_color_arr:   uint8 (h,w,3) or None
        """
        if mask_img is None:
            return None, np.ones((h, w), dtype=bool), None

        # New simplified mode: "no-colors" or "colors"
        mode = getattr(self, "mask_shape_mode", "no-colors")
        if mode not in ("no-colors", "colors"):
            raise ValueError("mask_shape_mode must be 'no-colors' or 'colors'")

        if mode == "no-colors":
            # Binary mask: black allowed, white not allowed
            # Use NEAREST to preserve crisp edges
            mask_resized = mask_img.convert("L").resize(
                (w, h), resample=PILImage.NEAREST
            )
            gray = np.array(mask_resized, dtype=np.uint8)
            mask_allowed = gray < 128
            mask_color_arr = None  # no colors used in this mode

        else:  # mode == "colors"
            # Color mask: non-transparent allowed, transparent not allowed
            # Use NEAREST for alpha to avoid "phantom" semi-transparent fringes
            mask_resized = mask_img.convert("RGBA").resize(
                (w, h), resample=PILImage.NEAREST
            )
            rgba = np.array(mask_resized, dtype=np.uint8)
            alpha = rgba[..., 3]
            mask_allowed = alpha > 0

            # If you want word colors sampled from mask:
            mask_color_arr = None
            if getattr(self, "use_mask_colors", False):
                mask_color_arr = rgba[..., :3].copy()

        return mask_resized, mask_allowed, mask_color_arr

    def _layout_with_attempts(
        self, base_word_boxes: List[WordBox], mask_img: Optional[PILImage.Image]
    ) -> List[WordBox]:
        """
        Unified mask scaling attempts for ALL three methods.

        - If mask_img is provided: all methods are constrained by it; we start with a resized
          minimal-ish mask and scale up on failure.
        - If mask_img is None:
            - greedy/square run unbounded (no attempts loop).
            - distance_transform uses an "all-allowed canvas" and scales up on failure.
        """
        # No-mask + non-DT: keep original behavior (unbounded canvas)
        if mask_img is None and self.method in ("greedy", "square"):
            self.failure = None
            boxes = self._clone_word_boxes(base_word_boxes)
            if self.method == "greedy":
                return self._layout_greedy(boxes, mask_allowed=None)
            else:
                return self._layout_square(boxes, mask_allowed=None)

        # Masked OR distance_transform without mask => attempts loop with a working canvas
        init_w, init_h = self._estimate_initial_mask_size(base_word_boxes, mask_img)
        w, h = init_w, init_h

        last_partial: List[WordBox] = []
        last_failure: Optional[dict] = None

        for attempt in range(self.max_attempts):
            boxes = self._clone_word_boxes(base_word_boxes)

            mask_resized_img, mask_allowed, mask_color_arr = self._make_working_mask(
                boxes, mask_img, w, h
            )

            self._mask_used_img = mask_resized_img
            self._mask_used_allowed = mask_allowed
            self._mask_used_color_arr = mask_color_arr  # <-- NEW

            # Remember the mask/canvas used for *this attempt* (and ultimately the final one)
            self._mask_used_img = mask_resized_img
            self._mask_used_allowed = mask_allowed

            self.failure = None

            if self.method == "greedy":
                placed = self._layout_greedy(boxes, mask_allowed=mask_allowed)
            elif self.method == "square":
                placed = self._layout_square(boxes, mask_allowed=mask_allowed)
            elif self.method == "distance_transform":
                placed = self._layout_distance_transform_on_mask(boxes, mask_allowed)
            else:
                raise ValueError(self.method)

            last_partial = placed

            # Success = all words placed
            if len(placed) == len(base_word_boxes) and self.failure is None:

                # Apply mask-based colors if enabled
                if (
                    getattr(self, "use_mask_colors", False)
                    and self._mask_used_color_arr is not None
                ):
                    self._apply_colors_from_mask(
                        placed,
                        self._mask_used_color_arr,
                        mask_allowed=self._mask_used_allowed,
                        strategy="median",
                    )

                self._last_attempts = attempt + 1

                # if mask_allowed is not None:
                #     mask_img_to_save = PILImage.fromarray(
                #         np.where(mask_allowed, 0, 255).astype(np.uint8),
                #         mode="L",
                #     )
                #     mask_img_to_save.save("truewordcloud_mask_allowed.png")
                return placed

            # Record failure info (non-fatal), then scale up and retry
            fail = self.failure or {
                "method": self.method,
                "word_index": len(placed),
                "word": (
                    None
                    if len(placed) >= len(base_word_boxes)
                    else base_word_boxes[len(placed)].word
                ),
                "reason": "not_all_words_placed",
            }
            fail = dict(fail)
            fail["attempt"] = attempt
            fail["mask_size"] = (w, h)
            last_failure = fail

            w = int(w * self.scale_factor)
            h = int(h * self.scale_factor)

        # Out of attempts: return best partial
        self._last_attempts = self.max_attempts
        self.failure = last_failure

        if (
            getattr(self, "use_mask_colors", False)
            and self._mask_used_color_arr is not None
        ):
            self._apply_colors_from_mask(
                last_partial,
                self._mask_used_color_arr,
                mask_allowed=self._mask_used_allowed,
                strategy="median",
            )

        return last_partial

    # ----------------------------
    # Greedy layout
    # ----------------------------

    def _layout_greedy(
        self, word_boxes: List[WordBox], mask_allowed: Optional[np.ndarray] = None
    ) -> List[WordBox]:
        """
        Greedy spiral placement algorithm (FAIL-FAST).
        If mask_allowed is provided, ink pixels must lie in allowed region.
        """
        if not word_boxes:
            return []

        placed_words: List[WordBox] = []

        first_word = word_boxes[0]
        if mask_allowed is not None:
            if self._glyph_fits_mask(first_word, 0.0, 0.0, mask_allowed):
                first_word.x = 0.0
                first_word.y = 0.0
            else:
                pos = self._find_position_spiral_with_mask(first_word, [], mask_allowed)
                if pos is None:
                    self.failure = {
                        "method": "greedy",
                        "word_index": 0,
                        "word": first_word.word,
                        "font_size": first_word.font_size,
                        "box_size": (first_word.width, first_word.height),
                        "reason": "largest_word_no_position_found",
                    }
                    return []
                first_word.x, first_word.y = pos
        else:
            first_word.x = 0.0
            first_word.y = 0.0

        placed_words.append(first_word)

        for word_index, word_box in enumerate(word_boxes[1:], start=1):
            if mask_allowed is not None:
                pos = self._find_position_spiral_with_mask(
                    word_box, placed_words, mask_allowed
                )
            else:
                pos = self._find_position_spiral(word_box, placed_words)

            if pos is None:
                self.failure = {
                    "method": "greedy",
                    "word_index": word_index,
                    "word": word_box.word,
                    "font_size": word_box.font_size,
                    "box_size": (word_box.width, word_box.height),
                    "reason": "no_position_found",
                }
                return placed_words  # FAIL-FAST

            word_box.x, word_box.y = pos
            placed_words.append(word_box)

        return placed_words

    def _find_position_spiral_with_mask(
        self,
        word_box: WordBox,
        placed_words: List[WordBox],
        mask_allowed: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Spiral search constrained by mask (ink pixels must lie within allowed region)."""
        max_radius = int(0.75 * max(mask_allowed.shape))

        word_size = math.sqrt(word_box.width * word_box.height)
        # Finer radial sampling for large words -> higher packing density
        radius_step = max(1, min(5, int(word_size / 80)))

        word_size = math.sqrt(word_box.width * word_box.height)
        angle_step = max(3, min(15, int(word_size / self.angle_divisor)))

        for radius in range(0, max_radius, radius_step):
            for angle in range(0, 360, angle_step):
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                # overlap
                overlap = False
                for placed in placed_words:
                    if self._boxes_overlap(word_box, placed):
                        overlap = True
                        break
                if overlap:
                    continue

                # mask fit
                if self._glyph_fits_mask(word_box, x, y, mask_allowed):
                    return (x, y)

        return None

    def _find_position_spiral(
        self, word_box: WordBox, placed_words: List[WordBox]
    ) -> Optional[Tuple[float, float]]:
        """Spiral search without mask."""
        max_radius = 2000

        word_size = math.sqrt(word_box.width * word_box.height)
        # Finer radial sampling for large words -> higher packing density
        radius_step = max(1, min(5, int(word_size / 80)))

        radius_step = 5
        word_size = math.sqrt(word_box.width * word_box.height)
        angle_step = max(5, min(30, int(word_size / self.angle_divisor)))

        for radius in range(0, max_radius, radius_step):
            for angle in range(0, 360, angle_step):
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                overlap = False
                for placed in placed_words:
                    if self._boxes_overlap(word_box, placed):
                        overlap = True
                        break

                if not overlap:
                    return (x, y)

        return None

    # ----------------------------
    # Square layout
    # ----------------------------

    def _layout_square(
        self, word_boxes: List[WordBox], mask_allowed: Optional[np.ndarray] = None
    ) -> List[WordBox]:
        """
        Center-outward square-ish packing (FAIL-FAST).
        If mask_allowed is provided, ink pixels must lie in allowed region.
        """
        if not word_boxes:
            return []

        placed_words: List[WordBox] = []

        # Place first word
        first_word = word_boxes[0]
        if mask_allowed is not None:
            if self._glyph_fits_mask(first_word, 0.0, 0.0, mask_allowed):
                first_word.x, first_word.y = 0.0, 0.0
            else:
                pos = self._find_position_spiral_with_mask(first_word, [], mask_allowed)
                if pos is None:
                    self.failure = {
                        "method": "square",
                        "word_index": 0,
                        "word": first_word.word,
                        "font_size": first_word.font_size,
                        "box_size": (first_word.width, first_word.height),
                        "reason": "largest_word_no_position_found",
                    }
                    return []
                first_word.x, first_word.y = pos
        else:
            first_word.x, first_word.y = 0.0, 0.0

        placed_words.append(first_word)

        # Place remaining words
        for word_index, word_box in enumerate(word_boxes[1:], start=1):
            best_position = None

            min_x = min(wb.x - wb.width / 2 - self.margin for wb in placed_words)
            max_x = max(wb.x + wb.width / 2 + self.margin for wb in placed_words)
            min_y = min(wb.y - wb.height / 2 - self.margin for wb in placed_words)
            max_y = max(wb.y + wb.height / 2 + self.margin for wb in placed_words)

            candidates: List[Tuple[float, float]] = []

            step = max(20, int(word_box.width / 2))
            for x in range(int(min_x), int(max_x) + 1, step):
                for y in range(int(min_y), int(max_y) + 1, step):
                    pos = self._find_non_overlapping_position(
                        word_box, x, y, placed_words, mask_allowed
                    )
                    if pos is not None:
                        candidates.append(pos)

            x_right = max_x + word_box.width / 2 + self.margin
            for y in range(int(min_y), int(max_y) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x_right, y, placed_words, mask_allowed
                )
                if pos is not None:
                    candidates.append(pos)

            x_left = min_x - word_box.width / 2 - self.margin
            for y in range(int(min_y), int(max_y) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x_left, y, placed_words, mask_allowed
                )
                if pos is not None:
                    candidates.append(pos)

            y_bottom = max_y + word_box.height / 2 + self.margin
            for x in range(int(min_x), int(max_x) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x, y_bottom, placed_words, mask_allowed
                )
                if pos is not None:
                    candidates.append(pos)

            y_top = min_y - word_box.height / 2 - self.margin
            for x in range(int(min_x), int(max_x) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x, y_top, placed_words, mask_allowed
                )
                if pos is not None:
                    candidates.append(pos)

            self.rng.shuffle(candidates)

            if candidates:
                metrics = []
                for x, y in candidates:
                    new_min_x = min(min_x, x - word_box.width / 2 - self.margin)
                    new_max_x = max(max_x, x + word_box.width / 2 + self.margin)
                    new_min_y = min(min_y, y - word_box.height / 2 - self.margin)
                    new_max_y = max(max_y, y + word_box.height / 2 + self.margin)

                    new_w = new_max_x - new_min_x
                    new_h = new_max_y - new_min_y

                    squareness = abs(new_w - new_h)
                    distance = (x * x + y * y) ** 0.5
                    metrics.append((squareness, distance, x, y))

                best_sq = min(m[0] for m in metrics)
                tol = best_sq * 0.05 + 1.0
                bests = [m for m in metrics if abs(m[0] - best_sq) <= tol]

                best_dist = min(m[1] for m in bests)
                dtol = best_dist * 0.05 + 1.0
                opts = [m for m in bests if abs(m[1] - best_dist) <= dtol]

                _, _, x, y = self.rng.choice(opts)
                best_position = (x, y)

            if best_position is None:
                self.failure = {
                    "method": "square",
                    "word_index": word_index,
                    "word": word_box.word,
                    "font_size": word_box.font_size,
                    "box_size": (word_box.width, word_box.height),
                    "reason": "no_position_found",
                }
                return placed_words  # FAIL-FAST

            word_box.x, word_box.y = best_position
            placed_words.append(word_box)

        return placed_words

    def _find_non_overlapping_position(
        self,
        word_box: WordBox,
        target_x: float,
        target_y: float,
        placed_words: List[WordBox],
        mask_allowed: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        """Try to place near (target_x,target_y), with optional mask constraint."""

        def mask_ok(x: float, y: float) -> bool:
            if mask_allowed is None:
                return True
            return self._glyph_fits_mask(word_box, x, y, mask_allowed)

        word_box.x = target_x
        word_box.y = target_y

        if not any(self._boxes_overlap(word_box, p) for p in placed_words) and mask_ok(
            target_x, target_y
        ):
            return (target_x, target_y)

        for radius in range(5, 50, 5):
            for angle in range(0, 360, 30):
                x = target_x + radius * math.cos(math.radians(angle))
                y = target_y + radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                if not any(
                    self._boxes_overlap(word_box, p) for p in placed_words
                ) and mask_ok(x, y):
                    return (x, y)

        return None

    # ----------------------------
    # Distance transform layout on a boolean allowed mask (no resizing here)
    # ----------------------------

    def _layout_distance_transform_on_mask(
        self, word_boxes: List[WordBox], mask_allowed: np.ndarray
    ) -> List[WordBox]:
        """
        Distance transform packing constrained to a boolean allowed mask (True=allowed).

        FAIL-FAST: stops at the first word that cannot be placed and sets self.failure.
        """
        if not word_boxes:
            return []

        placed_words: List[WordBox] = []
        mask_working = mask_allowed.copy()

        for word_index, word_box in enumerate(word_boxes):
            # DT: distance to forbidden/occupied (False)
            dist = distance_transform_edt(mask_working)

            # Loose threshold; exact fit decided by glyph check
            min_dist = min(word_box.width, word_box.height) / 2
            candidates = np.argwhere(dist >= min_dist)
            if len(candidates) == 0:
                self.failure = {
                    "method": "distance_transform",
                    "word_index": word_index,
                    "word": word_box.word,
                    "font_size": word_box.font_size,
                    "box_size": (word_box.width, word_box.height),
                    "reason": "no_candidates",
                }
                return placed_words  # FAIL-FAST

            # Prefer fattest spots first
            cand_dist = dist[candidates[:, 0], candidates[:, 1]]
            order = np.argsort(-cand_dist)

            found = False
            for idx in order:
                cy, cx = candidates[idx]  # NOTE: argwhere gives (row=y, col=x)

                # Must fit mask by ink pixels
                if not self._glyph_fits_mask_px(word_box, cx, cy, mask_working):
                    continue

                # Convert to wordcloud coords
                word_box.x = cx - mask_working.shape[1] // 2
                word_box.y = cy - mask_working.shape[0] // 2

                # Overlap with previously placed boxes
                overlap = False
                for placed in placed_words:
                    if self._boxes_overlap(word_box, placed):
                        overlap = True
                        break
                if overlap:
                    continue

                # Place it
                placed_words.append(word_box)
                self._stamp_word_on_mask_px(word_box, cx, cy, mask_working)
                found = True
                break

            if not found:
                self.failure = {
                    "method": "distance_transform",
                    "word_index": word_index,
                    "word": word_box.word,
                    "font_size": word_box.font_size,
                    "box_size": (word_box.width, word_box.height),
                    "reason": "no_position_found",
                }
                return placed_words  # FAIL-FAST

        return placed_words

    # ----------------------------
    # Rendering
    # ----------------------------

    def generate(
        self,
        mask: Optional[PILImage.Image] = None,
        mask_outline: bool = False,
        mask_outline_color=(0, 0, 0),
        mask_outline_width: int = 1,
    ) -> PILImage.Image:
        """
        Generate the word cloud image.

        Mask (if provided) is used by all methods, with unified scaling attempts.
        For distance_transform, mask may be None (uses an all-allowed canvas with attempts).
        """
        self.failure = None
        self._mask_used_img = None
        self._mask_used_allowed = None

        font_sizes = self._calculate_font_sizes()
        base_word_boxes = self._measure_words(font_sizes)

        # Unified attempts + scaling
        word_boxes = self._layout_with_attempts(base_word_boxes, mask)

        # If nothing placed, return a small diagnostic image
        if not word_boxes:
            if mask is not None:
                width, height = mask.size
            else:
                width = height = 300

            image = PILImage.new(
                "RGB", (max(1, width), max(1, height)), self.background_color
            )
            draw = PILImageDraw.Draw(image)
            msg = "No words could be placed."
            draw.text((10, 10), msg, fill=(150, 150, 150))
            self.words = []
            self._last_bbox = (0, 0, 0, 0)
            self._last_canvas_size = image.size
            self._last_offsets = (0, 0, 0)
            return image

        # Persist final layout
        self.words = word_boxes

        # Calculate bounding box
        min_x, min_y, max_x, max_y = self._calculate_bounding_box(word_boxes)

        # If drawing an outline, ensure bbox includes the mask extents too
        # if mask_outline and self._mask_used_allowed is not None:
        #     mh, mw = self._mask_used_allowed.shape
        #     mask_cx, mask_cy = mw // 2, mh // 2

        #     # Mask extents in wordcloud coords.
        #     # Left/top are -center; right/bottom are + (size - center)
        #     # Use -1 on right/bottom if you want exact pixel extent; leaving it as below is fine with ceil/+1.
        #     mask_min_x = -mask_cx
        #     mask_min_y = -mask_cy
        #     mask_max_x = mw - mask_cx
        #     mask_max_y = mh - mask_cy

        #     min_x = min(min_x, mask_min_x)
        #     min_y = min(min_y, mask_min_y)
        #     max_x = max(max_x, mask_max_x)
        #     max_y = max(max_y, mask_max_y)

        if mask_outline and self._mask_used_allowed is not None:
            mh, mw = self._mask_used_allowed.shape
            mask_cx, mask_cy = mw // 2, mh // 2

            # Tight bbox around the allowed region (or use outline if you prefer)
            ys, xs = np.where(self._mask_used_allowed)
            if len(xs) > 0:
                min_mx, max_mx = xs.min(), xs.max()
                min_my, max_my = ys.min(), ys.max()

                # Convert mask pixel coords -> wordcloud coords
                min_x = min_mx - mask_cx
                max_x = max_mx - mask_cx
                min_y = min_my - mask_cy
                max_y = max_my - mask_cy

        # Canonicalize bbox to integer pixel bounds (avoids truncation + off-by-one)
        min_xi = int(np.floor(min_x))
        min_yi = int(np.floor(min_y))
        max_xi = int(np.ceil(max_x))
        max_yi = int(np.ceil(max_y))

        self._last_bbox = (min_xi, min_yi, max_xi, max_yi)

        base_padding = 20

        # If drawing an outline, reserve extra margin so it can't be clipped
        outline_extra = (mask_outline_width // 2) + 2 if mask_outline else 0
        pad = base_padding + outline_extra

        width = (max_xi - min_xi + 1) + 2 * pad
        height = (max_yi - min_yi + 1) + 2 * pad

        image = PILImage.new("RGB", (width, height), self.background_color)
        draw = PILImageDraw.Draw(image)

        offset_x = -min_xi + pad
        offset_y = -min_yi + pad
        self._last_offsets = (offset_x, offset_y, pad)  # store actual pad used
        self._last_canvas_size = (width, height)
        self._last_bbox = (min_xi, min_yi, max_xi, max_yi)

        for wb in word_boxes:
            if self.font_path == "__PIL_DEFAULT__":
                font = PILImageFont.load_default()
            else:
                font = PILImageFont.truetype(self.font_path, wb.font_size)
            x = wb.x - wb.width / 2 + offset_x - wb.bbox_offset_x
            y = wb.y - wb.height / 2 + offset_y - wb.bbox_offset_y
            draw.text((x, y), wb.word, font=font, fill=wb.color)

        # Mask outline should match the *mask actually used* in the final attempt
        if mask_outline:
            outline_src = (
                self._mask_used_img if self._mask_used_img is not None else mask
            )
            if outline_src is not None:
                image = self._overlay_mask_outline(
                    image=image,
                    mask_allowed=self._mask_used_allowed,
                    used_bbox=(min_xi, min_yi, max_xi, max_yi),
                    padding=pad,  # <-- IMPORTANT: use pad, not base_padding
                    outline_color=mask_outline_color,
                    outline_width=mask_outline_width,
                )

        return image

    def save_mask_outline_png(
        self, mask_allowed: np.ndarray, filename: str = "mask_outline_debug.png"
    ):
        """
        Save the outline of the allowed mask as a PNG for visual debugging.
        White = outline, black = background.
        """
        from PIL import Image as PILImage
        from scipy.ndimage import binary_erosion
        import numpy as np

        eroded = binary_erosion(mask_allowed)
        outline = mask_allowed ^ eroded
        img = np.zeros(mask_allowed.shape, dtype=np.uint8)
        img[outline] = 255
        PILImage.fromarray(img, mode="L").save(filename)

    def _overlay_mask_outline(
        self,
        image: PILImage.Image,
        mask_allowed: np.ndarray,
        used_bbox: Tuple[int, int, int, int],  # <-- make it int
        padding: int,
        outline_color=(0, 0, 0),
        outline_width: int = 1,
    ) -> PILImage.Image:
        def hex_to_rgb(hex_color: str):
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join([c * 2 for c in hex_color])
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        color = outline_color
        if isinstance(color, str):
            color = hex_to_rgb(color)

        mask_h, mask_w = mask_allowed.shape
        mask_cx, mask_cy = mask_w // 2, mask_h // 2

        eroded = binary_erosion(mask_allowed)
        outline = mask_allowed ^ eroded
        ys, xs = np.where(outline)

        min_x, min_y, max_x, max_y = used_bbox
        width, height = image.size
        img_arr = np.array(image)

        # mask pixels -> wordcloud coords
        x_wc = xs - mask_cx
        y_wc = ys - mask_cy

        # wordcloud -> image coords (USE THE SAME padding you rendered with!)
        px = np.rint(x_wc - min_x + padding).astype(int)
        py = np.rint(y_wc - min_y + padding).astype(int)

        in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px = px[in_bounds]
        py = py[in_bounds]

        half = outline_width // 2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                fx = px + dx
                fy = py + dy
                ok = (fx >= 0) & (fx < width) & (fy >= 0) & (fy < height)
                img_arr[fy[ok], fx[ok], :] = color

        return PILImage.fromarray(img_arr, mode="RGB")

    def generate_with_stats(
        self,
        mask: Optional[PILImage.Image] = None,
        mask_outline: bool = False,
        mask_outline_color=(0, 0, 0),
        mask_outline_width: int = 1,
    ) -> Tuple[PILImage.Image, Dict]:
        image = self.generate(
            mask=mask,
            mask_outline=mask_outline,
            mask_outline_color=mask_outline_color,
            mask_outline_width=mask_outline_width,
        )

        font_sizes = self._calculate_font_sizes()
        stats = {
            "num_words": len(self.values),
            "size_range": (
                (min(font_sizes.values()), max(font_sizes.values()))
                if font_sizes
                else (0, 0)
            ),
            "canvas_size": image.size,
            "method": self.method,
            "placed_words": len(self.words),
            "success": (self.failure is None and len(self.words) == len(self.values)),
            "failure": self.failure,
            "mask_used_size": (
                None
                if self._mask_used_allowed is None
                else (
                    int(self._mask_used_allowed.shape[1]),
                    int(self._mask_used_allowed.shape[0]),
                )
            ),
            "attempts": self._last_attempts,
        }
        return image, stats


def main():

    values = {
        "python": 100,
        "data": 98,
        "science": 96,
        "visualization": 94,
        "frequency": 92,
        "proportional": 90,
        "word": 88,
        "cloud": 86,
        "layout": 84,
        "algorithm": 82,
        "packing": 80,
        "greedy": 78,
        "dynamic": 76,
        "analysis": 74,
        "text": 72,
        "font": 70,
        "size": 68,
        "canvas": 66,
        "mask": 64,
        "image": 62,
        "draw": 60,
        "color": 58,
        "random": 56,
        "spiral": 54,
        "square": 52,
        "compact": 50,
        "gap": 48,
        "fill": 46,
        "center": 44,
        "outward": 42,
        "method": 40,
        "test": 38,
        "true": 36,
        "proportion": 34,
        "relationship": 32,
        "input": 30,
        "output": 28,
        "statistics": 26,
        "resize": 22,
        "transform": 20,
        "distance": 18,
        "edge": 16,
        "background": 14,
        "white": 12,
        "black": 10,
    }

    mask_img = PILImage.open("examples/assets/cloud.png").convert("L")
    color_mask_img = PILImage.open("examples/assets/cloud.png")

    print("=" * 70)
    print("TrueWordCloud Color Mask Test (Greedy)")
    print("=" * 70)

    twc_color = TrueWordCloud(
        values=values,
        method="greedy",  # try also: "square", "distance_transform"
        base_font_size=150,
        min_font_size=1,
        margin=3,
        max_attempts=20,
        scale_factor=1.1,
        seed=123,
        use_mask_colors=False,  # or use_color=True depending on your naming
        mask_shape_mode="no-colors",
    )

    image, stats = twc_color.generate_with_stats(
        mask=mask_img,
        mask_outline=True,
        mask_outline_color="#00AAFF",
        mask_outline_width=2,
    )

    image.save("truewordcloud_greedy_color_mask_test.png")

    print(f"Canvas size: {stats['canvas_size']}")
    print(f"Font range: {stats['size_range']}")
    print(f"Method: {stats['method']}")
    print(f"Placed words: {stats['placed_words']} / {stats['num_words']}")
    print(f"Success: {stats['success']}")
    print(f"Mask used size: {stats['mask_used_size']}")
    print(f"Failure: {stats['failure']}")
    print(f"Attempts: {stats['attempts']}")
    print("Saved: truewordcloud_greedy_color_mask_test.png")

    # print("=" * 70)
    # print("TrueWordCloud Color Mask Test (Square)")
    # print("=" * 70)

    # twc_color = TrueWordCloud(
    #     values=values,
    #     method="square",  # try also: "greedy", "distance_transform"
    #     base_font_size=50,
    #     min_font_size=1,
    #     margin=3,
    #     max_attempts=20,
    #     scale_factor=1.3,
    #     seed=123,
    #     use_mask_colors=True,  # or use_color=True depending on your naming
    #     mask_shape_mode="colors",
    # )

    # image, stats = twc_color.generate_with_stats(
    #     mask=color_mask_img,
    #     mask_outline=True,
    #     mask_outline_color="#00AAFF",
    #     mask_outline_width=2,
    # )

    # image.save("truewordcloud_square_color_mask_test.png")

    # print(f"Canvas size: {stats['canvas_size']}")
    # print(f"Font range: {stats['size_range']}")
    # print(f"Method: {stats['method']}")
    # print(f"Placed words: {stats['placed_words']} / {stats['num_words']}")
    # print(f"Success: {stats['success']}")
    # print(f"Mask used size: {stats['mask_used_size']}")
    # print(f"Failure: {stats['failure']}")
    # print(f"Attempts: {stats['attempts']}")
    # print("Saved: truewordcloud_square_color_mask_test.png")

    print("=" * 70)
    print("TrueWordCloud Color Mask Test (Distance Transform)")
    print("=" * 70)

    twc_color = TrueWordCloud(
        values=values,
        method="distance_transform",  # try also: "greedy", "square"
        base_font_size=150,
        min_font_size=1,
        margin=3,
        max_attempts=20,
        scale_factor=1.2,
        seed=123,
        use_mask_colors=False,  # or use_color=True depending on your naming
        mask_shape_mode="no-colors",
    )

    image, stats = twc_color.generate_with_stats(
        mask=mask_img,
        mask_outline=True,
        mask_outline_color="#00AAFF",
        mask_outline_width=2,
    )

    image.save("truewordcloud_distance_transform_mask_test.png")
    print(f"Canvas size: {stats['canvas_size']}")
    print(f"Font range: {stats['size_range']}")
    print(f"Method: {stats['method']}")
    print(f"Placed words: {stats['placed_words']} / {stats['num_words']}")
    print(f"Success: {stats['success']}")
    print(f"Mask used size: {stats['mask_used_size']}")
    print(f"Failure: {stats['failure']}")
    print(f"Attempts: {stats['attempts']}")
    print("Saved: truewordcloud_distance_transform_mask_test.png")


if __name__ == "__main__":
    main()
