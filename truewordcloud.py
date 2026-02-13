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

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import math
import random


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

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return (left, top, right, bottom)"""
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)

    def overlaps(self, other: "WordBox", margin: int = 2) -> bool:
        """Check if this word overlaps with another word"""
        l1, t1, r1, b1 = self.bbox
        l2, t2, r2, b2 = other.bbox

        # Add margin for visual separation
        l1 -= margin
        t1 -= margin
        r1 += margin
        b1 += margin

        return not (r1 < l2 or r2 < l1 or b1 < t2 or b2 < t1)


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
    ):
        self.values = values
        self.method = method.lower()
        if self.method not in ['greedy', 'square']:
            raise ValueError(f"method must be 'greedy' or 'square', got '{method}'")
        
        self.base_font_size = base_font_size
        self.font_path = font_path or self._get_default_font()
        self.min_font_size = min_font_size
        self.background_color = background_color
        self.margin = margin
        self.color_func = color_func or self._default_color_func

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

    def _default_color_func(self, word: str, freq: float, norm_freq: float) -> Tuple[int, int, int]:
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

    def _measure_words(self, font_sizes: Dict[str, int]) -> List[WordBox]:
        """Measure word dimensions at their true font sizes"""
        word_boxes = []
        max_freq = max(self.values.values())

        for word, font_size in font_sizes.items():
            frequency = self.values[word]
            norm_freq = frequency / max_freq

            # Measure text size
            font = ImageFont.truetype(self.font_path, font_size)
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
            )
            word_boxes.append(word_box)

        # Sort by size (largest first) for better packing
        word_boxes.sort(key=lambda wb: wb.frequency, reverse=True)

        return word_boxes

    def _layout_greedy(self, word_boxes: List[WordBox]) -> List[WordBox]:
        """
        Greedy spiral placement algorithm.
        
        Fast and deterministic. Places largest word at center, then spirals
        outward to find non-overlapping positions for remaining words.
        """
        if not word_boxes:
            return []

        placed_words = []

        # Place first (largest) word at center
        first_word = word_boxes[0]
        first_word.x = 0
        first_word.y = 0
        placed_words.append(first_word)

        # Place remaining words using spiral search
        for word_box in word_boxes[1:]:
            position = self._find_position_spiral(word_box, placed_words)
            if position:
                word_box.x, word_box.y = position
                placed_words.append(word_box)

        return placed_words

    def _find_position_spiral(
        self, word_box: WordBox, placed_words: List[WordBox]
    ) -> Optional[Tuple[float, float]]:
        """Find a non-overlapping position using spiral search"""
        max_radius = 2000
        radius_step = 5
        angle_step = 30  # degrees

        for radius in range(0, max_radius, radius_step):
            for angle in range(0, 360, angle_step):
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                # Check for overlaps
                has_overlap = False
                for placed_word in placed_words:
                    if word_box.overlaps(placed_word, self.margin):
                        has_overlap = True
                        break

                if not has_overlap:
                    return (x, y)

        return None

    def _layout_square(self, word_boxes: List[WordBox]) -> List[WordBox]:
        """
        Center-outward square packing algorithm.
        
        Maintains roughly square aspect ratio by trying positions in all
        four directions and choosing the one that keeps layout most square.
        Includes interior gap-filling for compact layouts.
        """
        if not word_boxes:
            return []

        placed_words = []

        # Place first (largest) word at center
        first_word = word_boxes[0]
        first_word.x = 0
        first_word.y = 0
        placed_words.append(first_word)

        # Place remaining words
        for word_box in word_boxes[1:]:
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
                        word_box, x, y, placed_words
                    )
                    if pos:
                        candidates.append(pos)

            # RIGHT edge
            x_right = max_x + word_box.width / 2 + self.margin
            for y in range(int(min_y), int(max_y) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x_right, y, placed_words
                )
                if pos:
                    candidates.append(pos)

            # LEFT edge
            x_left = min_x - word_box.width / 2 - self.margin
            for y in range(int(min_y), int(max_y) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x_left, y, placed_words
                )
                if pos:
                    candidates.append(pos)

            # BOTTOM edge
            y_bottom = max_y + word_box.height / 2 + self.margin
            for x in range(int(min_x), int(max_x) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x, y_bottom, placed_words
                )
                if pos:
                    candidates.append(pos)

            # TOP edge
            y_top = min_y - word_box.height / 2 - self.margin
            for x in range(int(min_x), int(max_x) + 1, 20):
                pos = self._find_non_overlapping_position(
                    word_box, x, y_top, placed_words
                )
                if pos:
                    candidates.append(pos)

            # Shuffle for randomness
            random.shuffle(candidates)

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
                    m for m in candidate_metrics if abs(m[0] - best_squareness) <= tolerance
                ]

                best_distance = min(m[1] for m in best_candidates)
                distance_tolerance = best_distance * 0.05 + 1.0

                optimal_candidates = [
                    m for m in best_candidates if abs(m[1] - best_distance) <= distance_tolerance
                ]

                _, _, x, y = random.choice(optimal_candidates)
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
    ) -> Optional[Tuple[float, float]]:
        """Try to find a non-overlapping position near the target"""
        word_box.x = target_x
        word_box.y = target_y

        if not any(self._boxes_overlap(word_box, placed) for placed in placed_words):
            return (target_x, target_y)

        # Try small adjustments in spiral pattern
        for radius in range(5, 50, 5):
            for angle in range(0, 360, 30):
                x = target_x + radius * math.cos(math.radians(angle))
                y = target_y + radius * math.sin(math.radians(angle))

                word_box.x = x
                word_box.y = y

                if not any(self._boxes_overlap(word_box, placed) for placed in placed_words):
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

        return not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)

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

    def generate(self) -> Image.Image:
        """Generate the word cloud image."""
        # Calculate true proportional font sizes
        font_sizes = self._calculate_font_sizes()

        # Measure words at their true sizes
        word_boxes = self._measure_words(font_sizes)

        # Layout using selected method
        if self.method == 'greedy':
            print(f"Generating with greedy spiral algorithm...")
            word_boxes = self._layout_greedy(word_boxes)
        else:  # square
            print(f"Generating with center-outward square packing...")
            word_boxes = self._layout_square(word_boxes)

        # Calculate minimum bounding box
        min_x, min_y, max_x, max_y = self._calculate_bounding_box(word_boxes)

        # Add padding
        padding = 20
        width = int(max_x - min_x) + 2 * padding
        height = int(max_y - min_y) + 2 * padding

        # Render to image
        image = Image.new("RGB", (width, height), self.background_color)
        draw = ImageDraw.Draw(image)

        offset_x = -min_x + padding
        offset_y = -min_y + padding

        for word_box in word_boxes:
            font = ImageFont.truetype(self.font_path, word_box.font_size)
            x = word_box.x - word_box.width / 2 + offset_x
            y = word_box.y - word_box.height / 2 + offset_y
            draw.text((x, y), word_box.word, font=font, fill=word_box.color)

        return image

    def generate_with_stats(self) -> Tuple[Image.Image, Dict]:
        """Generate word cloud and return statistics."""
        image = self.generate()
        font_sizes = self._calculate_font_sizes()

        stats = {
            "num_words": len(self.values),
            "size_range": (min(font_sizes.values()), max(font_sizes.values())),
            "canvas_size": image.size,
            "method": self.method,
        }

        return image, stats


def main():
    """Test function for TrueWordCloud"""
    print("=" * 70)
    print("TrueWordCloud Test")
    print("=" * 70)

    values = {
        "python": 100,
        "data": 80,
        "science": 75,
        "visualization": 60,
        "frequency": 50,
        "proportional": 45,
        "word": 40,
        "cloud": 40,
        "layout": 30,
        "algorithm": 25,
        "packing": 20,
        "greedy": 15,
        "dynamic": 10,
    }

    print(f"\nInput: {len(values)} words")
    print(f"Value range: {min(values.values())} to {max(values.values())}")

    # Test greedy method
    print("\n" + "=" * 70)
    print("Testing GREEDY method...")
    print("=" * 70)
    twc_greedy = TrueWordCloud(
        values=values,
        method='greedy',
        base_font_size=100,
        min_font_size=10,
        background_color=(255, 255, 255),
        margin=3,
    )
    image, stats = twc_greedy.generate_with_stats()
    image.save("truewordcloud_greedy_test.png")
    print(f"✓ Canvas size: {stats['canvas_size'][0]}×{stats['canvas_size'][1]} pixels")
    print(f"✓ Font range: {stats['size_range'][0]}-{stats['size_range'][1]}pt")
    print(f"✓ Method: {stats['method']}")
    print(f"✓ Saved: truewordcloud_greedy_test.png")

    # Test square method
    print("\n" + "=" * 70)
    print("Testing SQUARE method...")
    print("=" * 70)
    twc_square = TrueWordCloud(
        values=values,
        method='square',
        base_font_size=100,
        min_font_size=10,
        background_color=(255, 255, 255),
        margin=3,
    )
    image, stats = twc_square.generate_with_stats()
    image.save("truewordcloud_square_test.png")
    print(f"✓ Canvas size: {stats['canvas_size'][0]}×{stats['canvas_size'][1]} pixels")
    print(f"✓ Font range: {stats['size_range'][0]}-{stats['size_range'][1]}pt")
    print(f"✓ Method: {stats['method']}")
    print(f"✓ Saved: truewordcloud_square_test.png")


if __name__ == "__main__":
    main()
