"""
TrueWordCloud - Value-Proportional Word Cloud Generator
========================================================

A word cloud generator that maintains TRUE proportional relationships between values.
Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud
ensures font sizes are ALWAYS proportional to the input values.

Key Features:
- Font sizes are ALWAYS proportional to input values (no squeezing/normalization)
- Two layout algorithms: 'greedy' (fast, deterministic) and 'square' (compact, randomized)
- Dynamic canvas sizing (content determines size, not pre-fixed dimensions)
- Works with any numeric values: frequencies, keyness scores, TF-IDF, etc.

Usage:
    from truewordcloud import TrueWordCloud

    # Simple usage
    twc = TrueWordCloud(values={'hello': 100, 'world': 50})
    image = twc.generate()
    image.save('wordcloud.png')

    # With options
    twc = TrueWordCloud(
        values={'hello': 100, 'world': 50},
        method='square',  # or 'greedy'
        base_font_size=100,
        font_path='path/to/font.ttf'
    )
    image, stats = twc.generate_with_stats()
"""

from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
from scipy.ndimage import distance_transform_edt, binary_erosion
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import math
import random
import numpy as np


@dataclass
class WordBox:
    """Represents a word with its rendering properties and position"""

    word: str
    frequency: float
    font_size: int
    width: int
    height: int
    x: float  # Center position
    y: float
    color: Tuple[int, int, int] = (0, 0, 0)
    bbox_offset_x: int = 0  # getbbox left offset
    bbox_offset_y: int = 0  # getbbox top offset

    # @property
    # def bbox(self) -> Tuple[float, float, float, float]:
    #     """Return (left, top, right, bottom)"""
    #     half_w = self.width / 2
    #     half_h = self.height / 2
    #     return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)

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

    This generator maintains TRUE proportional relationships - font sizes are always
    proportional to input values, never arbitrarily resized.

    Two layout methods available:

    'greedy' (default):
        - Fast spiral placement from center outward
        - Deterministic (same input → same output)
        - Creates radial/circular patterns
        - Best for: speed, reproducibility, circular aesthetics

    'square':
        - Center-outward square packing with gap filling
        - Randomized (varied layouts each run)
        - Creates compact, roughly square layouts
        - Best for: compact layouts, gap filling, visual variety

    Args:
        values: Dictionary mapping words to numeric values (frequencies, scores, etc.)
        method: Layout algorithm - 'greedy' or 'square' (default: 'greedy')
        base_font_size: Font size for maximum value word (default: 100)
        font_path: Path to TrueType font file (default: system font)
        min_font_size: Minimum font size for words (default: 10)
        background_color: RGB tuple for background (default: white)
        margin: Pixels between words (default: 2)
        color_func: Function to generate colors, receives (word, freq, norm_freq)
        angle_divisor: For greedy method, controls angular granularity based on word size.
                    Smaller values = coarser search (faster), larger = finer (better gap-filling).
                    (default: 3)
    """

    def __init__(
        self,
        values: Dict[str, float],
        method: str = "greedy",
        base_font_size: int = 100,
        font_path: Optional[str] = None,
        min_font_size: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        margin: int = 2,
        color_func: Optional[callable] = None,
        angle_divisor: float = 3.0,
        seed: Optional[int] = None,
    ):
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

        self.words: List[WordBox] = []

    def _get_default_font(self) -> str:
        """Try to find a suitable default font"""
        import matplotlib.font_manager as fm
        import platform

        system = platform.system()

        # Platform-specific fonts (without hyphens to avoid parse errors)
        if system == "Windows":
            candidates = ["Arial", "Segoe UI", "Calibri", "Verdana"]
        elif system == "Darwin":  # macOS
            candidates = ["Helvetica", "Arial", "SF Pro Display"]
        else:  # Linux
            candidates = ["DejaVu Sans", "Liberation Sans", "FreeSans"]

        # Try each candidate
        for font_name in candidates:
            try:
                matches = [
                    f.fname
                    for f in fm.fontManager.ttflist
                    if font_name.lower() in f.name.lower()
                ]
                if matches:
                    return matches[0]
            except:
                continue

        # Fallback to any font
        if fm.fontManager.ttflist:
            return fm.fontManager.ttflist[0].fname

        raise RuntimeError("No TrueType fonts found on system")

    def _default_color_func(
        self, word: str, freq: float, norm_freq: float
    ) -> Tuple[int, int, int]:
        """Default color function - black"""
        return (0, 0, 0)

    def _calculate_font_sizes(self) -> Dict[str, int]:
        """Calculate true proportional font sizes"""
        if not self.values:
            return {}

        max_value = max(self.values.values())
        if max_value == 0:
            return {word: self.min_font_size for word in self.values}

        font_sizes = {}
        for word, value in self.values.items():
            # True proportional scaling
            size = int(self.base_font_size * (value / max_value))
            # Enforce minimum
            font_sizes[word] = max(size, self.min_font_size)

        return font_sizes

    def _mask_to_allowed(self, mask_img: PILImage.Image) -> np.ndarray:
        """
        Convert a PIL mask image to a boolean array where:
        - True  = allowed (black)
        - False = forbidden (white)
        """
        return np.array(mask_img.convert("L")) < 128

    def _measure_words(self, font_sizes: Dict[str, int]) -> List[WordBox]:
        """Measure word dimensions at their true font sizes"""
        word_boxes = []
        max_freq = max(self.values.values())

        for word, font_size in font_sizes.items():
            frequency = self.values[word]
            norm_freq = frequency / max_freq

            # Measure text size
            font = PILImageFont.truetype(self.font_path, font_size)
            bbox = font.getbbox(word)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Get color
            color = self.color_func(word, frequency, norm_freq)

            word_box = WordBox(
                word=word,
                frequency=frequency,
                font_size=font_size,
                width=width,
                height=height,
                x=0,
                y=0,
                color=color,
                bbox_offset_x=bbox[0],
                bbox_offset_y=bbox[1],
            )
            word_boxes.append(word_box)

        # Sort by size (largest first) for better packing
        word_boxes.sort(key=lambda wb: wb.frequency, reverse=True)

        return word_boxes

    def _get_glyph_mask(self, word_box: WordBox) -> np.ndarray:
        """
        Returns a boolean mask (h,w) where True means 'ink pixel' for this word.
        Cached by (word, font_size).
        """
        key = (word_box.word, word_box.font_size)
        if not hasattr(self, "_glyph_cache"):
            self._glyph_cache = {}
        if key in self._glyph_cache:
            return self._glyph_cache[key]

        font = PILImageFont.truetype(self.font_path, word_box.font_size)
        w, h = word_box.width, word_box.height

        img = PILImage.new("L", (w, h), 0)
        draw = PILImageDraw.Draw(img)

        # Draw so that the bbox aligns with [0:w, 0:h]
        draw.text(
            (-word_box.bbox_offset_x, -word_box.bbox_offset_y),
            word_box.word,
            font=font,
            fill=255,
        )

        glyph = np.array(img) > 0
        self._glyph_cache[key] = glyph
        return glyph

    def reseed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def _rect_fits_mask_px(
        self,
        word_box: WordBox,
        cx: int,
        cy: int,
        mask_allowed: np.ndarray,
    ) -> bool:
        """
        Check if word_box's axis-aligned rectangle fits entirely within allowed pixels,
        using mask pixel coordinates (cx, cy are center indices in the mask array).
        """
        h, w = mask_allowed.shape

        left = int(cx - word_box.width / 2 - word_box.bbox_offset_x)
        top = int(cy - word_box.height / 2 - word_box.bbox_offset_y)
        right = left + word_box.width
        bottom = top + word_box.height

        # Must be fully inside bounds
        if left < 0 or top < 0 or right > w or bottom > h:
            return False

        # Must be fully allowed
        return mask_allowed[top:bottom, left:right].all()

    def _rect_fits_mask(
        self, word_box: WordBox, x: float, y: float, mask: np.ndarray
    ) -> bool:
        """Return True if the word's bounding rectangle at (x,y) lies entirely in allowed mask pixels."""
        mask_h, mask_w = mask.shape
        mask_cx, mask_cy = mask_w // 2, mask_h // 2

        left = int(mask_cx + x - word_box.width / 2 - word_box.bbox_offset_x)
        top = int(mask_cy + y - word_box.height / 2 - word_box.bbox_offset_y)
        right = left + word_box.width
        bottom = top + word_box.height

        if left < 0 or top < 0 or right > mask_w or bottom > mask_h:
            return False

        return mask[top:bottom, left:right].all()

    def _layout_greedy(
        self, word_boxes: List[WordBox], mask: Optional[np.ndarray] = None
    ) -> List[WordBox]:
        """
        Greedy spiral placement algorithm.

        Fast and deterministic. Places largest word near center, then spirals
        outward to find non-overlapping positions for remaining words.
        """
        if not word_boxes:
            return []

        placed_words: List[WordBox] = []

        # Place first (largest) word
        first_word = word_boxes[0]

        if mask is not None:
            # Try exact center first (cheap)
            center_ok = self._rect_fits_mask(first_word, 0.0, 0.0, mask)
            if center_ok:
                first_word.x = 0.0
                first_word.y = 0.0
            else:
                # Fall back to mask-aware spiral search
                pos = self._find_position_spiral_with_mask(first_word, [], mask)
                if pos is None:
                    self.failure = {
                        "method": "greedy",
                        "word_index": 0,
                        "word": first_word.word,
                        "font_size": first_word.font_size,
                        "box_size": (first_word.width, first_word.height),
                        "reason": "largest_word_no_position_found",
                    }
                    return []  # nothing could be placed
                first_word.x, first_word.y = pos
        else:
            first_word.x = 0.0
            first_word.y = 0.0

        placed_words.append(first_word)

        # Place remaining words using spiral search
        for word_box_id, word_box in enumerate(word_boxes[1:], 1):
            print(word_box_id, word_box.word)

            if mask is not None:
                position = self._find_position_spiral_with_mask(
                    word_box, placed_words, mask
                )
            else:
                position = self._find_position_spiral(word_box, placed_words)

            if position is None:
                self.failure = {
                    "method": "greedy",
                    "word_index": word_box_id,
                    "word": word_box.word,
                    "font_size": word_box.font_size,
                    "box_size": (word_box.width, word_box.height),
                    "reason": "no_position_found",
                }
                return placed_words  # FAIL-FAST: stop placing further words

            word_box.x, word_box.y = position
            placed_words.append(word_box)

        return placed_words

    def _find_position_spiral_with_mask(
        self, word_box: WordBox, placed_words: List[WordBox], mask: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Find a non-overlapping position using spiral search, constrained by a mask (True=allowed, False=forbidden)."""
        mask_h, mask_w = mask.shape
        mask_cx, mask_cy = mask_w // 2, mask_h // 2

        def mask_check(x, y):
            left = int(mask_cx + x - word_box.width / 2 - word_box.bbox_offset_x)
            top = int(mask_cy + y - word_box.height / 2 - word_box.bbox_offset_y)
            right = left + word_box.width
            bottom = top + word_box.height

            if left < 0 or top < 0 or right > mask_w or bottom > mask_h:
                return False

            glyph = self._get_glyph_mask(word_box)  # True where ink exists
            region = mask[top:bottom, left:right]  # True where allowed

            # Only require allowed where glyph has ink
            return region[glyph].all()

        max_radius = int(0.75 * max(mask.shape))

        radius_step = 5
        word_size = math.sqrt(word_box.width * word_box.height)
        angle_step = max(5, min(30, int(word_size / self.angle_divisor)))
        for radius in range(0, max_radius, radius_step):
            for angle in range(0, 360, angle_step):
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))
                word_box.x = x
                word_box.y = y
                has_overlap = False
                for placed_word in placed_words:
                    if self._boxes_overlap(word_box, placed_word):
                        has_overlap = True
                        break
                if not has_overlap and mask_check(x, y):
                    return (x, y)
        return None

    def _find_position_spiral(
        self, word_box: WordBox, placed_words: List[WordBox]
    ) -> Optional[Tuple[float, float]]:
        """Find a non-overlapping position using spiral search with adaptive angle step"""
        max_radius = 2000
        radius_step = 5

        # Adaptive angle step based on word dimensions
        # Smaller words use finer angular granularity to find gaps
        # Larger words use coarser steps for speed
        word_size = math.sqrt(word_box.width * word_box.height)
        angle_step = max(5, min(30, int(word_size / self.angle_divisor)))

        for radius in range(0, max_radius, radius_step):
            for angle in range(0, 360, angle_step):
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                # Check for overlaps
                has_overlap = False
                for placed_word in placed_words:
                    if self._boxes_overlap(word_box, placed_word):
                        has_overlap = True
                        break

                if not has_overlap:
                    return (x, y)

        return None

    def _layout_square(
        self, word_boxes: List[WordBox], mask: Optional[np.ndarray] = None
    ) -> List[WordBox]:
        """
        Center-outward square packing algorithm.

        Maintains roughly square aspect ratio by trying positions in all
        four directions and choosing the one that keeps layout most square.
        Includes interior gap-filling for compact layouts.
        """
        if not word_boxes:
            return []

        placed_words = []

        # Place first (largest) word
        first_word = word_boxes[0]

        if mask is not None:
            # Try exact center first
            if self._rect_fits_mask(first_word, 0.0, 0.0, mask):
                first_word.x = 0.0
                first_word.y = 0.0
            else:
                # Fall back to mask-aware spiral search to find *some* legal starting point
                pos = self._find_position_spiral_with_mask(first_word, [], mask)
                if pos is None:
                    return []  # mask too restrictive for even the largest word
                first_word.x, first_word.y = pos
        else:
            first_word.x = 0.0
            first_word.y = 0.0

        placed_words.append(first_word)

        # Place remaining words
        for word_box_id, word_box in enumerate(word_boxes[1:], start=1):
            print(word_box_id, word_box.word)
            best_position = None

            # Calculate current bounding box
            min_x = min(wb.x - wb.width / 2 - self.margin for wb in placed_words)
            max_x = max(wb.x + wb.width / 2 + self.margin for wb in placed_words)
            min_y = min(wb.y - wb.height / 2 - self.margin for wb in placed_words)
            max_y = max(wb.y + wb.height / 2 + self.margin for wb in placed_words)

            # Try many positions: interior positions first (to fill gaps), then edges
            candidates = []

            # INTERIOR positions: try a grid throughout the entire bounding box
            step = max(20, int(word_box.width / 2))
            for x in range(int(min_x), int(max_x) + 1, step):
                for y in range(int(min_y), int(max_y) + 1, step):
                    pos = self._find_non_overlapping_position(
                        word_box, x, y, placed_words, mask=mask
                    )
                    if pos:
                        candidates.append(pos)

            # RIGHT edge
            x_right = max_x + word_box.width / 2 + self.margin
            for y in range(int(min_y), int(max_y) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x_right, y, placed_words, mask=mask
                )
                if pos:
                    candidates.append(pos)

            # LEFT edge
            x_left = min_x - word_box.width / 2 - self.margin
            for y in range(int(min_y), int(max_y) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x_left, y, placed_words, mask=mask
                )
                if pos:
                    candidates.append(pos)

            # BOTTOM edge
            y_bottom = max_y + word_box.height / 2 + self.margin
            for x in range(int(min_x), int(max_x) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x, y_bottom, placed_words, mask=mask
                )
                if pos:
                    candidates.append(pos)

            # TOP edge
            y_top = min_y - word_box.height / 2 - self.margin
            for x in range(int(min_x), int(max_x) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x, y_top, placed_words, mask=mask
                )
                if pos:
                    candidates.append(pos)

            # Shuffle for randomness
            self.rng.shuffle(candidates)

            # Evaluate candidates and choose best
            if candidates:
                candidate_metrics = []
                for x, y in candidates:
                    new_min_x = min(min_x, x - word_box.width / 2 - self.margin)
                    new_max_x = max(max_x, x + word_box.width / 2 + self.margin)
                    new_min_y = min(min_y, y - word_box.height / 2 - self.margin)
                    new_max_y = max(max_y, y + word_box.height / 2 + self.margin)

                    new_width = new_max_x - new_min_x
                    new_height = new_max_y - new_min_y

                    # Squareness metric
                    squareness = abs(new_width - new_height)
                    distance = (x**2 + y**2) ** 0.5

                    candidate_metrics.append((squareness, distance, x, y))

                # Find best with tolerance for randomization
                best_squareness = min(m[0] for m in candidate_metrics)
                tolerance = best_squareness * 0.05 + 1.0

                best_candidates = [
                    m
                    for m in candidate_metrics
                    if abs(m[0] - best_squareness) <= tolerance
                ]

                best_distance = min(m[1] for m in best_candidates)
                distance_tolerance = best_distance * 0.05 + 1.0

                optimal_candidates = [
                    m
                    for m in best_candidates
                    if abs(m[1] - best_distance) <= distance_tolerance
                ]

                _, _, x, y = self.rng.choice(optimal_candidates)
                best_position = (x, y)

            if best_position:
                word_box.x, word_box.y = best_position
                placed_words.append(word_box)

        return placed_words

    def _find_non_overlapping_position(
        self,
        word_box: WordBox,
        target_x: float,
        target_y: float,
        placed_words: List[WordBox],
        mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """Try to find a non-overlapping position near the target, optionally constrained by a mask (True=allowed, False=forbidden)."""
        word_box.x = target_x
        word_box.y = target_y

        def mask_check(x, y):
            if mask is None:
                return True
            return self._rect_fits_mask(word_box, x, y, mask)

        if not any(
            self._boxes_overlap(word_box, placed) for placed in placed_words
        ) and mask_check(target_x, target_y):
            return (target_x, target_y)

        # Try small adjustments in spiral pattern
        for radius in range(5, 50, 5):
            for angle in range(0, 360, 30):
                x = target_x + radius * math.cos(math.radians(angle))
                y = target_y + radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                if not any(
                    self._boxes_overlap(word_box, placed) for placed in placed_words
                ) and mask_check(x, y):
                    return (x, y)

        return None

    def _boxes_overlap(self, box1: WordBox, box2: WordBox) -> bool:
        """Check if two word boxes overlap (including margin)"""
        left1, top1, right1, bottom1 = box1.bbox
        left2, top2, right2, bottom2 = box2.bbox

        # Expand by margin
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
        """Calculate minimum bounding box containing all words"""
        if not word_boxes:
            return (0, 0, 0, 0)

        min_x = min(wb.bbox[0] for wb in word_boxes)
        min_y = min(wb.bbox[1] for wb in word_boxes)
        max_x = max(wb.bbox[2] for wb in word_boxes)
        max_y = max(wb.bbox[3] for wb in word_boxes)

        return (min_x, min_y, max_x, max_y)

    def generate(
        self,
        mask: Optional[PILImage.Image] = None,
        mask_outline: bool = False,
        mask_outline_color=(0, 0, 0),
        mask_outline_width: int = 1,
    ) -> PILImage.Image:
        """
        Generate the word cloud image. For 'distance_transform', mask is optional: if not supplied, use a square canvas.
        If mask_outline is True, overlay the outline of the mask on the output image.
        """
        self.failure = None

        font_sizes = self._calculate_font_sizes()
        word_boxes = self._measure_words(font_sizes)

        if self.method == "greedy":
            print(f"Generating with greedy spiral algorithm...")
            mask_array = None
            if mask is not None:
                mask_array = self._mask_to_allowed(mask)
            word_boxes = self._layout_greedy(word_boxes, mask=mask_array)
        elif self.method == "square":
            print(f"Generating with center-outward square packing...")
            mask_array = None
            if mask is not None:
                mask_array = self._mask_to_allowed(mask)
            word_boxes = self._layout_square(word_boxes, mask=mask_array)
        elif self.method == "distance_transform":
            print(f"Generating with distance transform packing...")
            word_boxes, mask_shape = self._layout_distance_transform(word_boxes, mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if not word_boxes:
            if mask is not None:
                width, height = mask.size
            else:
                width = height = 300

            image = PILImage.new(
                "RGB", (max(1, width), max(1, height)), self.background_color
            )
            draw = PILImageDraw.Draw(image)
            msg = "No words could be placed (mask too restrictive / font too large)."
            draw.text((10, 10), msg, fill=(150, 150, 150))
            self.words = []
            self._last_bbox = (0, 0, 0, 0)
            self._last_canvas_size = image.size
            self._last_offsets = (0, 0, 0)
            return image

        # NEW: persist final layout
        self.words = word_boxes

        # Calculate minimum bounding box
        min_x, min_y, max_x, max_y = self._calculate_bounding_box(word_boxes)
        self._last_bbox = (min_x, min_y, max_x, max_y)

        # Add padding
        padding = 20
        width = int(max_x - min_x) + 2 * padding
        height = int(max_y - min_y) + 2 * padding
        self._last_canvas_size = (width, height)

        # Render to image
        image = PILImage.new("RGB", (width, height), self.background_color)
        draw = PILImageDraw.Draw(image)

        offset_x = -min_x + padding
        offset_y = -min_y + padding
        self._last_offsets = (offset_x, offset_y, padding)

        for word_box in word_boxes:
            font = PILImageFont.truetype(self.font_path, word_box.font_size)
            x = word_box.x - word_box.width / 2 + offset_x - word_box.bbox_offset_x
            y = word_box.y - word_box.height / 2 + offset_y - word_box.bbox_offset_y
            draw.text((x, y), word_box.word, font=font, fill=word_box.color)

        # Overlay mask outline if requested
        if mask is not None and mask_outline:
            # Support hex string for mask_outline_color
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip("#")
                if len(hex_color) == 3:
                    hex_color = "".join([c * 2 for c in hex_color])
                return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

            color = mask_outline_color
            if isinstance(color, str):
                color = hex_to_rgb(color)

            mask_bin = self._mask_to_allowed(mask)
            mask_h, mask_w = mask_bin.shape
            mask_cx, mask_cy = mask_w // 2, mask_h // 2

            # Outline of the mask (in mask pixel coords)
            eroded = binary_erosion(mask_bin)
            outline = mask_bin ^ eroded
            ys, xs = np.where(outline)

            # Only draw outline pixels that fall inside the USED wordcloud bbox
            # (this is the "crop"). No scaling.
            half = mask_outline_width // 2

            # Convert output image to a writable numpy array view
            img_arr = np.array(image)

            # Map mask outline pixels -> wordcloud coords
            x_wc = xs - mask_cx
            y_wc = ys - mask_cy

            # Crop in wordcloud coords to the used bbox
            inside = (
                (x_wc >= min_x) & (x_wc <= max_x) & (y_wc >= min_y) & (y_wc <= max_y)
            )
            x_wc = x_wc[inside]
            y_wc = y_wc[inside]

            # Wordcloud coords -> image coords
            px = np.rint(x_wc - min_x + padding).astype(int)
            py = np.rint(y_wc - min_y + padding).astype(int)

            # Clip to image bounds (safety)
            in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px = px[in_bounds]
            py = py[in_bounds]

            # Apply thickness by stamping neighbors
            half = mask_outline_width // 2
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    fx = px + dx
                    fy = py + dy
                    ok = (fx >= 0) & (fx < width) & (fy >= 0) & (fy < height)
                    img_arr[fy[ok], fx[ok], :] = (
                        color  # broadcast RGB into selected pixels
                    )

            # Convert back to PIL
            image = PILImage.fromarray(img_arr, mode="RGB")

        return image

    def generate_with_stats(
        self,
        mask: Optional[PILImage.Image] = None,
        mask_outline: bool = False,
        mask_outline_color=(0, 0, 0),
        mask_outline_width: int = 1,
    ) -> Tuple[PILImage.Image, Dict]:
        """Generate word cloud and return statistics. For 'distance_transform', mask is optional: if not supplied, use a square canvas."""
        image = self.generate(
            mask=mask,
            mask_outline=mask_outline,
            mask_outline_color=mask_outline_color,
            mask_outline_width=mask_outline_width,
        )
        font_sizes = self._calculate_font_sizes()

        stats = {
            "num_words": len(self.values),
            "size_range": (min(font_sizes.values()), max(font_sizes.values())),
            "canvas_size": image.size,
            "method": self.method,
            "placed_words": len(self.words),
            "success": (self.failure is None),
            "failure": self.failure,
        }

        return image, stats

    def _estimate_initial_mask_size(self, word_boxes, mask_img=None):
        """
        Estimate initial mask/canvas size based on total word area and mask aspect ratio.
        If mask_img is provided, preserve its aspect ratio.
        Returns (width, height)
        """
        total_area = sum(wb.width * wb.height for wb in word_boxes)
        if mask_img is not None:
            aspect = mask_img.width / mask_img.height
            h = int((total_area / aspect) ** 0.5)
            w = int(h * aspect)
        else:
            # Square canvas
            w = h = int(total_area**0.5)
        return max(w, 1), max(h, 1)

    def _layout_distance_transform(self, word_boxes, mask_img):
        """
        Distance transform packing with dynamic mask/canvas resizing.

        Mask convention (consistent with greedy/square):
        - black (0)   = allowed region
        - white (255) = forbidden region

        Internally we convert the mask to a boolean array where True=allowed.
        If mask_img is None, we use a square canvas with all pixels allowed.
        """

        # 1. Estimate initial mask size
        if mask_img is not None:
            init_w, init_h = self._estimate_initial_mask_size(word_boxes, mask_img)
            mask_resized = mask_img.resize((init_w, init_h), resample=PILImage.NEAREST)
            mask = self._mask_to_allowed(mask_resized)
        else:
            # No mask: use a square canvas, all allowed
            init_w, init_h = self._estimate_initial_mask_size(word_boxes, None)
            mask = np.ones((init_h, init_w), dtype=bool)

        placed_words = []
        scale_factor = 1.1  # 10% grow each time if needed
        max_attempts = 20
        attempt = 0

        while attempt < max_attempts:
            mask_working = mask.copy()
            placed_words.clear()
            failed = False

            for word_box_id, word_box in enumerate(word_boxes):
                print("attempt", attempt, word_box_id, word_box.word)
                # Compute distance transform (distance to forbidden)
                dist = distance_transform_edt(mask_working)
                # Find all positions where word fits (distance >= min required)
                min_dist = (
                    min(word_box.width, word_box.height) / 2
                )  # looser, rely on rect check
                # min_dist = 0.5 * math.hypot(word_box.width, word_box.height)

                candidates = np.argwhere(dist >= min_dist)
                if len(candidates) == 0:
                    failed = True
                    break
                # Sort candidates by distance (fattest spots first)
                candidate_distances = dist[candidates[:, 0], candidates[:, 1]]
                sorted_indices = np.argsort(-candidate_distances)
                found_position = False
                for idx in sorted_indices:

                    cy, cx = candidates[idx]

                    # NEW: ensure rectangle fits allowed region (not just circle distance)
                    if not self._rect_fits_mask_px(word_box, cx, cy, mask_working):
                        continue

                    # Set word position in word cloud coordinates
                    word_box.x = cx - mask_working.shape[1] // 2
                    word_box.y = cy - mask_working.shape[0] // 2

                    # Check for overlap with already placed words
                    has_overlap = False
                    for placed_word in placed_words:
                        if self._boxes_overlap(word_box, placed_word):
                            has_overlap = True
                            break

                    if not has_overlap:
                        placed_words.append(word_box)

                        # Mark region as filled (set to False) - now safe to do without clipping
                        left = int(cx - word_box.width / 2 - word_box.bbox_offset_x)
                        top = int(cy - word_box.height / 2 - word_box.bbox_offset_y)
                        right = left + word_box.width
                        bottom = top + word_box.height
                        mask_working[top:bottom, left:right] = False

                        found_position = True
                        break

            if not failed:
                # All words placed
                return placed_words, mask.shape
            # If failed, scale up mask and try again
            attempt += 1
            new_w = int(mask.shape[1] * scale_factor)
            new_h = int(mask.shape[0] * scale_factor)
            if mask_img is not None:
                mask_img = mask_img.resize((new_w, new_h), resample=PILImage.NEAREST)
                mask = self._mask_to_allowed(mask_img)
            else:
                mask = np.ones((new_h, new_w), dtype=bool)

        raise RuntimeError(
            "Could not fit all words in mask after multiple attempts. Try increasing max_attempts or check mask/word sizes."
        )


def main():
    """Test function for TrueWordCloud"""
    print("=" * 70)
    print("TrueWordCloud Test")
    print("=" * 70)

    # Load mask (black=allowed, white=forbidden)
    mask_img = PILImage.open("examples/assets/mask_alice.png").convert("L")
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
        "dynamic": 24,
        "resize": 22,
        "transform": 20,
        "distance": 18,
        "edge": 16,
        "background": 14,
        "white": 12,
        "black": 10,
    }

    print(f"\nInput: {len(values)} words")
    print(f"Value range: {min(values.values())} to {max(values.values())}")
    # values = dict(list(values.items())[0:20])  # Use only top 20 for testing to speed up

    # Test greedy method
    print("\n" + "=" * 70)
    print("Testing GREEDY method...")
    print("=" * 70)
    print(values)
    twc_greedy = TrueWordCloud(
        values=values,
        method="greedy",
        base_font_size=50,
        min_font_size=1,
        background_color=(255, 255, 255),
        margin=3,
    )
    image, stats = twc_greedy.generate_with_stats(
        mask=mask_img,
        mask_outline=True,
        mask_outline_color="#FF0000",
        mask_outline_width=2,
    )
    image.save("truewordcloud_greedy_test.png")
    print(f"✓ Canvas size: {stats['canvas_size'][0]}×{stats['canvas_size'][1]} pixels")
    print(f"✓ Font range: {stats['size_range'][0]}-{stats['size_range'][1]}pt")
    print(f"✓ Method: {stats['method']}")
    print(f"✓ Placed words: {stats['placed_words']} / {stats['num_words']}")
    print(f"✓ Success: {stats['success']}")
    print(f"✓ Failure: {stats['failure']}")
    print(f"✓ Saved: truewordcloud_greedy_test.png")

    # # Test square method
    # print("\n" + "=" * 70)
    # print("Testing SQUARE method...")
    # print("=" * 70)

    # twc_square = TrueWordCloud(
    #     values=values,
    #     method="square",
    #     base_font_size=100,
    #     min_font_size=10,
    #     background_color=(255, 255, 255),
    #     margin=3,
    # )
    # image, stats = twc_square.generate_with_stats(
    #     mask=mask_img,
    #     mask_outline=True,
    #     mask_outline_color="#FF0000",
    #     mask_outline_width=2,
    # )
    # image.save("truewordcloud_square_test.png")
    # print(f"✓ Canvas size: {stats['canvas_size'][0]}×{stats['canvas_size'][1]} pixels")
    # print(f"✓ Font range: {stats['size_range'][0]}-{stats['size_range'][1]}pt")
    # print(f"✓ Method: {stats['method']}")
    # print(f"✓ Saved: truewordcloud_square_test.png")

    # Test distance_transform method

    # print("\n" + "=" * 70)
    # print("Testing DISTANCE TRANSFORM method...")
    # print("=" * 70)

    # twc_dist = TrueWordCloud(
    #     values=values,
    #     method="distance_transform",
    #     base_font_size=100,
    #     min_font_size=10,
    #     background_color=(255, 255, 255),
    #     margin=3,
    # )
    # image, stats = twc_dist.generate_with_stats(
    #     mask=mask_img,
    #     mask_outline=True,
    #     mask_outline_color="#FF0000",
    #     mask_outline_width=2,
    # )
    # image.save("truewordcloud_distance_transform_test.png")
    # print(f"✓ Canvas size: {stats['canvas_size'][0]}×{stats['canvas_size'][1]} pixels")
    # print(f"✓ Font range: {stats['size_range'][0]}-{stats['size_range'][1]}pt")
    # print(f"✓ Method: {stats['method']}")
    # print(f"✓ Saved: truewordcloud_distance_transform_test.png")

    # # Test distance_transform method
    # print("\n" + "=" * 70)
    # print("Testing DISTANCE TRANSFORM method...")
    # print("=" * 70)

    # # Create a simple circular mask (black=allowed, white=forbidden)
    # mask_size = 400
    # # Start with a white image (all forbidden)
    # mask_img = PILImage.new(
    #     "L", (mask_size, mask_size), 255
    # )  # white background (forbidden)
    # draw = PILImageDraw.Draw(mask_img)
    # # Draw a black circle in the center (allowed region)
    # draw.ellipse(
    #     (20, 20, mask_size - 20, mask_size - 20), fill=0
    # )  # black circle (allowed)
    # # Optionally save the mask for inspection
    # # mask_img.save("truewordcloud_test_mask_circle.png")

    # twc_dist = TrueWordCloud(
    #     values=values,
    #     method="distance_transform",
    #     base_font_size=100,
    #     min_font_size=10,
    #     background_color=(255, 255, 255),
    #     margin=3,
    # )
    # image, stats = twc_dist.generate_with_stats(
    #     mask=mask_img,
    #     mask_outline=False,
    #     mask_outline_color="#FF0000",
    #     mask_outline_width=2,
    # )
    # image.save("truewordcloud_distance_transform_test.png")
    # print(f"✓ Canvas size: {stats['canvas_size'][0]}×{stats['canvas_size'][1]} pixels")
    # print(f"✓ Font range: {stats['size_range'][0]}-{stats['size_range'][1]}pt")
    # print(f"✓ Method: {stats['method']}")
    # print(f"✓ Saved: truewordcloud_distance_transform_test.png")

    # # Test special case: applying a mask to the greedy and square method
    # # Power transform to reduce Zipfian distribution severity
    # POWER_TRANSFORM = 0.6
    # values = {word: freq**POWER_TRANSFORM for word, freq in values.items()}

    # # Generate word cloud with square layout and mask
    # twc = TrueWordCloud(
    #     values=values, method="square", base_font_size=80, min_font_size=10
    # )
    # image = twc.generate(
    #     mask=mask_img,
    #     mask_outline=False,
    #     mask_outline_color="#FF0000",
    #     mask_outline_width=2,
    # )
    # image.save("truewordcloud_square_mask_alice_test.png")
    # print("✓ Saved: truewordcloud_square_mask_alice_test.png")


if __name__ == "__main__":
    main()
