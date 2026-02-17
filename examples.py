"""
TrueWordCloud Examples
======================

This file demonstrates various use cases for TrueWordCloud.
"""

import csv
from truewordcloud import TrueWordCloud
from PIL import Image as PILImage


def get_example_data():
    """Helper function to load example data from CSV and stoplist"""
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
    sorted_words = sorted(values.items(), key=lambda x: x[1], reverse=True)[:80]
    values = dict(sorted_words)

    # Apply power transform to reduce frequency range
    POWER_TRANSFORM = 0.6
    values = {word: freq**POWER_TRANSFORM for word, freq in values.items()}

    return values


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

    # Distance transform layout
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

    # Distance transform layout
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

    # Greedy layoutd
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

    # Distance transform layout
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

    # Distance transform layout
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

    # Distance transform layout
    twc_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
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

    font_path = "examples/assets/roboto_font/Roboto_Condensed-Black.ttf"

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

    # Distance transform layout
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
    """Using CSV input with a stoplist to filter out common words"""

    values = get_example_data()

    # Greedy layout
    twc_greedy = TrueWordCloud(
        values=values, method="greedy", base_font_size=70, min_font_size=12
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_csv.png")
    print("✓ Greedy: examples/greedy_csv.png")

    # Square layout
    twc_square = TrueWordCloud(
        values=values, method="square", base_font_size=70, min_font_size=12
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_csv.png")
    print("✓ Square: examples/square_csv.png")

    # Distance transform layout
    twc_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_csv.png")
    print("✓ Distance Transform: examples/distance_transform_csv.png")


def example_with_mask():
    """Using a heart-shaped mask to constrain word placement"""
    values = get_example_data()

    # Load heart-shaped mask (black=allowed, white=forbidden)
    mask_img = PILImage.open("examples/assets/mask_heart.png").convert("L")

    # Distance greedy layout with mask
    twc_greedy = TrueWordCloud(
        values=values,
        method="greedy",
        base_font_size=70,
        min_font_size=12,
        mask=mask_img,
        mask_shape_transparency=False,
        show_mask_outline=True,
        mask_outline_color="#000000",
        mask_outline_width=2,
    )
    image_greedy = twc_greedy.generate()
    image_greedy.save("examples/greedy_mask_heart.png")
    print("✓ Greedy with Heart Mask: examples/greedy_mask_heart.png")

    # Distance square layout with mask
    twc_square = TrueWordCloud(
        values=values,
        method="square",
        base_font_size=70,
        min_font_size=12,
        mask=mask_img,
        mask_shape_transparency=False,
        show_mask_outline=True,
        mask_outline_color="#000000",
        mask_outline_width=2,
    )
    image_square = twc_square.generate()
    image_square.save("examples/square_mask_heart.png")
    print("✓ Square with Heart Mask: examples/square_mask_heart.png")

    # Distance transform layout with mask
    twc_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
        mask=mask_img,
        mask_shape_transparency=False,
        show_mask_outline=True,
        mask_outline_color="#000000",
        mask_outline_width=2,
    )
    image_dist = twc_dist.generate()
    image_dist.save("examples/distance_transform_mask_heart.png")
    print(
        "✓ Distance Transform with Heart Mask: examples/distance_transform_mask_heart.png"
    )


def example_with_colored_mask():
    """Square layout with a colored mask and mask-derived word colors."""
    values = get_example_data()

    # Load heart-shaped mask (black=allowed, white=forbidden)
    color_mask_img = PILImage.open("examples/assets/mask_heart_color.png")

    # Greedy layout with colored mask
    twc_color = TrueWordCloud(
        values=values,
        method="greedy",  # try also: "greedy", "distance_transform"
        seed=123,
        mask=color_mask_img,
        mask_shape_transparency=True,
        show_mask_outline=True,
        mask_outline_color="#F80202",
        mask_outline_width=2,
        use_mask_colors=True,
    )

    image, stats = twc_color.generate_with_stats()
    image.save("examples/greedy_mask_heart_colored.png")
    print("✓ Greedy with colored Heart Mask: examples/greedy_mask_heart_colored.png")
    print(
        f"✓ Stats: placed {stats['placed_words']} / {stats['num_words']} words, success={stats['success']}"
    )

    # Square layout with colored mask
    twc_color_square = TrueWordCloud(
        values=values,
        method="square",
        seed=123,
        mask=color_mask_img,
        mask_shape_transparency=True,
        show_mask_outline=True,
        mask_outline_color="#F80202",
        mask_outline_width=2,
        use_mask_colors=True,
    )
    image_square, stats_square = twc_color_square.generate_with_stats()
    image_square.save("examples/square_mask_heart_colored.png")
    print("✓ Square with colored Heart Mask: examples/square_mask_heart_colored.png")
    print(
        f"✓ Stats: placed {stats_square['placed_words']} / {stats_square['num_words']} words, success={stats_square['success']}"
    )

    # Distance transform layout with colored mask
    twc_color_dist = TrueWordCloud(
        values=values,
        method="distance_transform",
        seed=123,
        mask=color_mask_img,
        mask_shape_transparency=True,
        show_mask_outline=True,
        mask_outline_color="#F80202",
        mask_outline_width=2,
        use_mask_colors=True,
    )
    image_dist, stats_dist = twc_color_dist.generate_with_stats()
    image_dist.save("examples/distance_transform_mask_heart_colored.png")
    print(
        "✓ Distance Transform with colored Heart Mask: examples/distance_transform_mask_heart_colored.png"
    )
    print(
        f"✓ Stats: placed {stats_dist['placed_words']} / {stats_dist['num_words']} words, success={stats_dist['success']}"
    )


def main():
    """Run all TrueWordCloud examples demonstrating different features and use cases."""
    print("\n" + "=" * 70)
    print("TrueWordCloud Examples")
    print("=" * 70)
    print("\nGenerating example word clouds...")
    print(
        "\nAll examples generate greedy, square, and distance transform layouts for comparison:"
    )
    print("  • Greedy: faster, deterministic, radial pattern")
    print("  • Square: compact, randomized, fills gaps")
    print(
        "  • Distance Transform: optimized for irrelegular shapes, slower, more organic"
    )

    print("\n1. Basic word frequencies (example_basic)")
    example_basic()
    print("\n2. Keyness scores/statistical significance (example_keyness)")
    example_keyness()
    print("\n3. Custom color function (example_custom_colors)")
    example_custom_colors()
    print("\n4. Categorical colors by word type (example_categorical_colors)")
    example_categorical_colors()
    print("\n5. Large dataset with many words (example_large_dataset)")
    example_large_dataset()
    print("\n6. Custom font usage (example_custom_font)")
    example_custom_font()
    print("\n7. Load data from CSV with stoplist (example_from_csv_with_stoplist)")
    example_from_csv_with_stoplist()
    print("\n8. Use a heart-shaped mask (example_with_mask)")
    example_with_mask()
    print(
        "\n9. Use a colored mask and mask-derived word colors (example_with_colored_mask)"
    )
    example_with_colored_mask()

    print("\n" + "=" * 70)
    print("✓ All examples generated successfully!")
    print("✓ Compare greedy vs square outputs to see layout differences")
    print("=" * 70)


if __name__ == "__main__":
    main()
