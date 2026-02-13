"""
TrueWordCloud Examples
======================

This file demonstrates various use cases for TrueWordCloud.
"""

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

    twc = TrueWordCloud(values=word_frequencies, method="greedy")
    image = twc.generate()
    image.save("examples/example_greedy_basic.png")
    print("✓ Saved: examples/example_greedy_basic.png")


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

    twc = TrueWordCloud(
        values=keyness_scores, method="square", base_font_size=80, margin=4
    )
    image = twc.generate()
    image.save("examples/example_square_keyness.png")
    print("✓ Saved: examples/example_square_keyness.png")


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

    twc = TrueWordCloud(
        values=values,
        method="greedy",
        color_func=red_blue_gradient,
        background_color=(255, 255, 255),
    )
    image = twc.generate()
    image.save("examples/example_greedy_colors.png")
    print("✓ Saved: examples/example_greedy_colors.png")


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

    twc = TrueWordCloud(values=values, method="square", color_func=category_color)
    image = twc.generate()
    image.save("examples/example_square_categorical.png")
    print("✓ Saved: examples/example_square_categorical.png")


def example_comparison():
    """Compare greedy vs square methods"""
    print("\n" + "=" * 70)
    print("Example 5: Comparing Methods")
    print("=" * 70)

    values = {
        "word": 100,
        "cloud": 90,
        "visualization": 80,
        "data": 70,
        "text": 60,
        "frequency": 50,
        "analysis": 40,
        "corpus": 30,
        "term": 20,
    }

    # Greedy method
    twc_greedy = TrueWordCloud(values=values, method="greedy")
    img_greedy, stats_greedy = twc_greedy.generate_with_stats()
    img_greedy.save("examples/example_greedy.png")
    print(
        f"✓ Greedy: {stats_greedy['canvas_size'][0]}×{stats_greedy['canvas_size'][1]}px"
    )

    # Square method
    twc_square = TrueWordCloud(values=values, method="square")
    img_square, stats_square = twc_square.generate_with_stats()
    img_square.save("examples/example_square.png")
    print(
        f"✓ Square: {stats_square['canvas_size'][0]}×{stats_square['canvas_size'][1]}px"
    )

    print("\nGreedy: faster, deterministic, radial pattern")
    print("Square: compact, randomized, fills gaps")


def example_large_dataset():
    """Large dataset with many words"""
    print("\n" + "=" * 70)
    print("Example 6: Large Dataset (50 words)")
    print("=" * 70)

    # Simulate word frequencies with decreasing values
    import random

    words = [f"word{i:02d}" for i in range(1, 51)]
    values = {word: int(100 * (0.95**i)) for i, word in enumerate(words)}

    twc = TrueWordCloud(
        values=values, method="square", base_font_size=80, min_font_size=8, margin=2
    )

    image, stats = twc.generate_with_stats()
    image.save("examples/example_square_large.png")
    print(f"✓ Generated cloud with {stats['num_words']} words")
    print(f"✓ Font sizes: {stats['size_range'][0]}pt - {stats['size_range'][1]}pt")
    print(f"✓ Canvas: {stats['canvas_size'][0]}×{stats['canvas_size'][1]}px")


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

    twc = TrueWordCloud(
        values=values, method="greedy", font_path=font_path, base_font_size=90
    )

    image = twc.generate()
    image.save("examples/example_greedy_custom_font.png")
    print(f"✓ Saved with {'custom' if font_path else 'default'} font")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("TrueWordCloud Examples")
    print("=" * 70)
    print("\nGenerating example word clouds...")

    example_basic()
    example_keyness()
    example_custom_colors()
    example_categorical_colors()
    example_comparison()
    example_large_dataset()
    example_custom_font()

    print("\n" + "=" * 70)
    print("✓ All examples generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
