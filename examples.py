"""
TrueWordCloud Examples
======================

This file demonstrates various use cases for TrueWordCloud.
"""

import csv
from truewordcloud import TrueWordCloud


def example_basic():
    """Basic usage with word frequencies"""
    print("\n" + "=" * 70)
    print("Example 1: Basic Word Frequencies")
    print("=" * 70)

    word_frequencies = {
        "python": 1000,
        "data": 800,
        "science": 700,
        "machine": 600,
        "learning": 600,
        "analysis": 500,
        "visualization": 400,
        "algorithm": 300,
        "statistics": 250,
        "model": 200,
    }

    # Greedy layout
    twc_greedy = TrueWordCloud(values=word_frequencies, method="greedy")
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_basic.png")
    print("✓ Saved: examples/greedy_basic.png")

    # Square layout
    twc_square = TrueWordCloud(values=word_frequencies, method="square")
    image_square = twc_square.generate()
    image_square.save("examples/square_basic.png")
    print("✓ Saved: examples/square_basic.png")

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(values=word_frequencies, method="distance_transform")
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_basic.png")
    print("✓ Saved: examples/distance_transform_basic.png")


def example_keyness():
    """Using keyness scores (statistical significance)"""
    print("\n" + "=" * 70)
    print("Example 2: Keyness Scores")
    print("=" * 70)

    keyness_scores = {
        "significant": 15.7,
        "corpus": 12.3,
        "analysis": 10.8,
        "frequency": 8.9,
        "linguistic": 7.4,
        "text": 6.2,
        "distribution": 5.1,
        "pattern": 4.3,
    }

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=keyness_scores, method="greedy", base_font_size=80, margin=4
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_keyness.png")
    print("✓ Saved: examples/greedy_keyness.png")

    # Square layout
    twc_square = TrueWordCloud(
        values=keyness_scores, method="square", base_font_size=80, margin=4
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_keyness.png")
    print("✓ Saved: examples/square_keyness.png")

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(
        values=keyness_scores, method="distance_transform", base_font_size=80, margin=4
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_keyness.png")
    print("✓ Saved: examples/distance_transform_keyness.png")


def example_custom_colors():
    """Using custom color function"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Colors (Red-Blue Gradient)")
    print("=" * 70)

    def red_blue_gradient(word, freq, norm_freq):
        """Color gradient from blue (low) to red (high)"""
        red = int(255 * norm_freq)
        blue = int(255 * (1 - norm_freq))
        return (red, 0, blue)

    values = {"hot": 100, "warm": 80, "moderate": 60, "cool": 40, "cold": 20}

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=values,
        method="greedy",
        color_func=red_blue_gradient,
        background_color=(255, 255, 255),
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_colors.png")
    print("✓ Saved: examples/greedy_colors.png")

    # Square layout
    twc_square = TrueWordCloud(
        values=values,
        method="square",
        color_func=red_blue_gradient,
        background_color=(255, 255, 255),
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_colors.png")
    print("✓ Saved: examples/square_colors.png")

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
        color_func=red_blue_gradient,
        background_color=(255, 255, 255),
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_colors.png")
    print("✓ Saved: examples/distance_transform_colors.png")


def example_categorical_colors():
    """Different colors for different categories"""
    print("\n" + "=" * 70)
    print("Example 4: Categorical Colors")
    print("=" * 70)

    # Words with their frequencies and categories
    data = {
        "python": (100, "language"),
        "java": (80, "language"),
        "javascript": (75, "language"),
        "numpy": (60, "library"),
        "pandas": (55, "library"),
        "matplotlib": (50, "library"),
        "algorithm": (45, "concept"),
        "optimization": (40, "concept"),
        "recursion": (35, "concept"),
    }

    # Extract values
    values = {word: freq for word, (freq, _) in data.items()}
    categories = {word: cat for word, (_, cat) in data.items()}

    def category_color(word, freq, norm_freq):
        """Color by category"""
        colors = {
            "language": (255, 0, 0),  # Red
            "library": (0, 0, 255),  # Blue
            "concept": (0, 128, 0),  # Green
        }
        return colors.get(categories[word], (0, 0, 0))

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=values, method="greedy", color_func=category_color
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_categorical.png")
    print("✓ Saved: examples/greedy_categorical.png")

    # Square layout
    twc_square = TrueWordCloud(
        values=values, method="square", color_func=category_color
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_categorical.png")
    print("✓ Saved: examples/square_categorical.png")

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(
        values=values, method="distance_transform", color_func=category_color
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_categorical.png")
    print("✓ Saved: examples/distance_transform_categorical.png")


def example_large_dataset():
    """Large dataset with many words"""
    print("\n" + "=" * 70)
    print("Example 6: Large Dataset (50 words)")
    print("=" * 70)

    # Simulate word frequencies with decreasing values
    import random

    words = [f"word{i:02d}" for i in range(1, 51)]
    values = {word: int(100 * (0.95**i)) for i, word in enumerate(words)}

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=values, method="greedy", base_font_size=80, min_font_size=8, margin=2
    )
    image_greedy, stats_greedy = twc_greedy.generate_with_stats()
    image_greedy.save("examples/greedy_large.png")
    print(
        f"✓ Greedy: {stats_greedy['num_words']} words, {stats_greedy['canvas_size'][0]}×{stats_greedy['canvas_size'][1]}px"
    )

    # Square layout
    twc_square = TrueWordCloud(
        values=values, method="square", base_font_size=80, min_font_size=8, margin=2
    )
    image_square, stats_square = twc_square.generate_with_stats()
    image_square.save("examples/square_large.png")
    print(
        f"✓ Square: {stats_square['num_words']} words, {stats_square['canvas_size'][0]}×{stats_square['canvas_size'][1]}px"
    )

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
        base_font_size=80,
        min_font_size=8,
        margin=2,
    )
    image_dist, stats_dist = twc_dist.generate_with_stats()
    image_dist.save("examples/distance_transform_large.png")
    print(
        f"✓ Distance Transform: {stats_dist['num_words']} words, {stats_dist['canvas_size'][0]}×{stats_dist['canvas_size'][1]}px"
    )


def example_custom_font():
    """Using a custom font"""
    print("\n" + "=" * 70)
    print("Example 7: Custom Font")
    print("=" * 70)

    values = {
        "Typography": 100,
        "Design": 80,
        "Fonts": 70,
        "Style": 60,
        "Creative": 50,
    }

    # Try to find a custom font (fallback to default if not found)
    import matplotlib.font_manager as fm

    # Look for a specific font style
    serif_fonts = [f.fname for f in fm.fontManager.ttflist if "times" in f.name.lower()]
    font_path = serif_fonts[0] if serif_fonts else None

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=values, method="greedy", font_path=font_path, base_font_size=90
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_custom_font.png")
    print(f"✓ Greedy: Saved with {'custom' if font_path else 'default'} font")

    # Square layout
    twc_square = TrueWordCloud(
        values=values, method="square", font_path=font_path, base_font_size=90
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_custom_font.png")
    print(f"✓ Square: Saved with {'custom' if font_path else 'default'} font")

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
        font_path=font_path,
        base_font_size=90,
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_custom_font.png")
    print(
        f"✓ Distance Transform: Saved with {'custom' if font_path else 'default'} font"
    )


def example_from_csv_with_stoplist():
    """Load word frequencies from CSV file with stoplist filtering"""
    print("\n" + "=" * 70)
    print("Example 8: Loading from CSV with Stoplist")
    print("=" * 70)

    # Power transform to reduce Zipfian distribution severity
    # Real word frequencies follow a Zipfian distribution (few very common words,
    # many rare words). Applying a power transform makes visualizations more balanced.
    # Tune this value: 1.0 = no change, 0.5 = square root, 0.6-0.7 = recommended range
    POWER_TRANSFORM = 0.6

    # Load stoplist
    stoplist = set()
    with open("examples/assets/sample_stoplist.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stoplist.add(word)

    print(f"Loaded {len(stoplist)} stopwords from sample_stoplist.txt")

    # Read CSV file
    values = {}
    with open("examples/assets/sample_wordlist.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            row = {k.strip(): v.strip() for k, v in row.items()}
            word = row["Type"]
            freq = int(row["Freq"])
            # Filter: keep only alphabetic words (no punctuation) and not in stoplist
            if word.isalpha() and len(word) > 1 and word.lower() not in stoplist:
                values[word] = freq

    # Use top 40 words for better visualization
    sorted_words = sorted(values.items(), key=lambda x: x[1], reverse=True)[:40]
    values = dict(sorted_words)

    # Apply power transform to reduce frequency range
    values_transformed = {word: freq**POWER_TRANSFORM for word, freq in values.items()}

    print(f"Loaded {len(values)} words from CSV (after filtering stopwords)")
    print(
        f"Applied power transform (exponent={POWER_TRANSFORM}) to reduce Zipfian drop-off"
    )
    print(
        f"Applied power transform (exponent={POWER_TRANSFORM}) to reduce Zipfian drop-off"
    )

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=values_transformed, method="greedy", base_font_size=70, min_font_size=12
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_csv.png")
    print("✓ Greedy: examples/greedy_csv.png")

    # Square layout
    twc_square = TrueWordCloud(
        values=values_transformed, method="square", base_font_size=70, min_font_size=12
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_csv.png")
    print("✓ Square: examples/square_csv.png")

    # Distance transform layout (no mask)
    twc_dist = TrueWordCloud(
        values=values_transformed,
        method="distance_transform",
        base_font_size=70,
        min_font_size=12,
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_csv.png")
    print("✓ Distance Transform: examples/distance_transform_csv.png")


def example_distance_transform_with_mask():
    """Distance transform layout with a heart-shaped mask using CSV and stoplist."""
    print("\n" + "=" * 70)
    print("Example 9: Distance Transform with Heart Mask")
    print("=" * 70)

    POWER_TRANSFORM = 0.6

    # Load stoplist
    stoplist = set()
    with open("examples/assets/sample_stoplist.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stoplist.add(word)

    # Read CSV file
    values = {}
    with open("examples/assets/sample_wordlist.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            word = row["Type"]
            freq = int(row["Freq"])
            if word.isalpha() and len(word) > 1 and word.lower() not in stoplist:
                values[word] = freq

    # Use top 40 words for better visualization
    sorted_words = sorted(values.items(), key=lambda x: x[1], reverse=True)[:40]
    values = dict(sorted_words)

    # Apply power transform
    values_transformed = {word: freq**POWER_TRANSFORM for word, freq in values.items()}

    print(f"Loaded {len(values)} words from CSV (after filtering stopwords)")
    print(
        f"Applied power transform (exponent={POWER_TRANSFORM}) to reduce Zipfian drop-off"
    )

    # Load heart-shaped mask (black=allowed, white=forbidden)
    from PIL import Image as PILImage

    # Distance greedy layout with mask
    twc_greedy = TrueWordCloud(
        values=values_transformed, method="greedy", base_font_size=70, min_font_size=12
    )
    mask_img = PILImage.open("examples/assets/mask_heart.png").convert("L")
    image_greedy = twc_greedy.generate(mask=mask_img)
    image_greedy.save("examples/greedy_mask_heart.png")
    print("✓ Greedy with Inverted Heart Mask: examples/greedy_mask_heart.png")

    mask_img = PILImage.open("examples/assets/mask_alice.png").convert("L")
    image_greedy = twc_greedy.generate(mask=mask_img)
    image_greedy.save("examples/greedy_mask_alice.png")
    print("✓ Greedy with Inverted Alice Mask: examples/greedy_mask_alice.png")

    # Distance transform layout with mask
    twc_dist = TrueWordCloud(
        values=values_transformed,
        method="distance_transform",
        base_font_size=70,
        min_font_size=12,
    )

    mask_img = PILImage.open("examples/assets/mask_heart.png").convert("L")
    image_dist = twc_dist.generate(mask=mask_img)
    image_dist.save("examples/distance_transform_mask_heart.png")
    print(
        "✓ Distance Transform with Inverted Heart Mask: examples/distance_transform_mask_heart.png"
    )

    mask_img = PILImage.open("examples/assets/mask_alice.png").convert("L")
    image_dist = twc_dist.generate(mask=mask_img)
    image_dist.save("examples/distance_transform_mask_alice.png")
    print(
        "✓ Distance Transform with Inverted Alice Mask: examples/distance_transform_mask_alice.png"
    )


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("TrueWordCloud Examples")
    print("=" * 70)
    print("\nGenerating example word clouds...")
    print("\nAll examples generate BOTH greedy and square layouts for comparison:")
    print("  • Greedy: faster, deterministic, radial pattern")
    print("  • Square: compact, randomized, fills gaps")

    example_basic()
    example_keyness()
    example_custom_colors()
    example_categorical_colors()
    example_large_dataset()
    example_custom_font()
    example_from_csv_with_stoplist()
    example_distance_transform_with_mask()

    print("\n" + "=" * 70)
    print("✓ All examples generated successfully!")
    print("✓ Compare greedy vs square outputs to see layout differences")
    print("=" * 70)


if __name__ == "__main__":
    main()
