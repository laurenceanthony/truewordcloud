"""
TrueWordCloud Mask Example
=========================

This file demonstrates using a mask image to constrain the word cloud shape.
"""

import csv
import numpy as np
from PIL import Image
from truewordcloud import TrueWordCloud


def example_spiral_with_mask():
    """Word cloud using spiral (greedy) approach with a mask."""
    print("\n" + "=" * 70)
    print("Example: Spiral with Mask (alice_mask.png)")
    print("=" * 70)

    # Load stoplist
    stoplist = set()
    with open("examples/sample_stoplist.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stoplist.add(word)

    # Read CSV file
    values = {}
    with open("examples/sample_wordlist.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            word = row["Type"]
            freq = int(row["Freq"])
            if word.isalpha() and len(word) > 1 and word.lower() not in stoplist:
                values[word] = freq

    # Use top 60 words for better visualization
    sorted_words = sorted(values.items(), key=lambda x: x[1], reverse=True)[:180]
    values = dict(sorted_words)

    # Power transform to reduce Zipfian distribution severity
    # Tune this value: 1.0 = no change, 0.5 = square root, 0.6-0.7 = recommended range
    POWER_TRANSFORM = 0.6
    values = {word: freq**POWER_TRANSFORM for word, freq in values.items()}

    def run_with_mask(mask, outname):
        # mask: True for black (foreground), False for white (background)
        mask_h, mask_w = mask.shape
        mask_cx, mask_cy = mask_w // 2, mask_h // 2

        def mask_check(x, y, width, height):
            # x, y are word cloud coordinates (centered at 0,0)
            # Map to mask image coordinates
            left = int(mask_cx + x - width / 2)
            top = int(mask_cy + y - height / 2)
            right = left + width
            bottom = top + height
            if left < 0 or top < 0 or right > mask_w or bottom > mask_h:
                return False
            return mask[top:bottom, left:right].all()

        orig_find_position_spiral = TrueWordCloud._find_position_spiral

        def _find_position_spiral_with_mask(self, word_box, placed_words):
            max_radius = 2000
            radius_step = 5
            word_size = np.sqrt(word_box.width * word_box.height)
            angle_step = max(
                5, min(30, int(word_size / getattr(self, "angle_divisor", 3)))
            )
            for radius in range(0, max_radius, radius_step):
                for angle in range(0, 360, angle_step):
                    x = radius * np.cos(np.radians(angle))
                    y = radius * np.sin(np.radians(angle))
                    word_box.x = x
                    word_box.y = y
                    has_overlap = False
                    for placed_word in placed_words:
                        if word_box.overlaps(placed_word, self.margin):
                            has_overlap = True
                            break
                    if not has_overlap and mask_check(
                        x, y, word_box.width, word_box.height
                    ):
                        return (x, y)
            return None

        TrueWordCloud._find_position_spiral = _find_position_spiral_with_mask
        twc = TrueWordCloud(
            values=values, method="greedy", base_font_size=80, min_font_size=10
        )
        image = twc.generate()
        image.save(outname)
        print(f"✓ Saved: {outname}")
        TrueWordCloud._find_position_spiral = orig_find_position_spiral

    # 1. Alice PNG mask
    mask_img = Image.open("examples/mask_alice.png").convert("L")
    mask = np.array(mask_img) < 128  # True for black (foreground)
    run_with_mask(mask, "examples/example_mask_spiral_alice.png")

    # 2. Heart PNG mask
    heart_img = Image.open("examples/mask_heart.png").convert("L")
    heart_mask = np.array(heart_img) < 128  # True for black (foreground)
    run_with_mask(heart_mask, "examples/example_mask_spiral_heart.png")


if __name__ == "__main__":
    example_spiral_with_mask()
