# TrueWordCloud

**Value-Proportional Word Cloud Generator**

A word cloud generator that maintains TRUE proportional relationships between values. Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud ensures font sizes are ALWAYS proportional to the input values.

## Key Features

- ✅ **True Proportionality** - Font sizes strictly proportional to input values (no squeezing/normalization)
- 🎨 **Two Layout Algorithms** - Choose between 'greedy' (fast, deterministic) and 'square' (compact, randomized)
- 📐 **Dynamic Canvas** - Canvas size determined by content, not pre-fixed dimensions
- 🔢 **Any Numeric Values** - Works with frequencies, keyness scores, TF-IDF, probabilities, etc.
- 🎯 **No Overlaps** - Guaranteed non-overlapping word placement
- 🌈 **Custom Colors** - Flexible color function support

## Installation

```bash
pip install truewordcloud
```

Or install from source:

```bash
git clone https://github.com/laurenceanthony/truewordcloud.git
cd truewordcloud
pip install -e .
```

## Quick Start

```python
from truewordcloud import TrueWordCloud

# Simple usage
values = {'python': 100, 'data': 80, 'science': 75, 'visualization': 60}
twc = TrueWordCloud(values=values)
image = twc.generate()
image.save('wordcloud.png')
```

## Layout Algorithms

### Greedy Spiral (method='greedy')

**Best for: Speed, reproducibility, circular aesthetics**

- ⚡ Fast spiral placement from center outward
- 🔒 Deterministic (same input → same output)
- 🎯 Creates radial/circular patterns
- ✅ Ideal for scientific papers, reports, consistent branding

```python
twc = TrueWordCloud(values=values, method='greedy')
```

### Square Packing (method='square')

**Best for: Compact layouts, gap filling, visual variety**

- 📦 Center-outward square packing with intelligent gap filling
- 🎲 Randomized (varied layouts each run)
- 📐 Maintains roughly square aspect ratio (width ≈ height)
- ✅ Ideal for presentations, posters, artistic displays

```python
twc = TrueWordCloud(values=values, method='square')
```

## Advanced Usage

### Custom Colors

```python
def color_func(word, freq, norm_freq):
    # norm_freq is between 0 and 1
    if norm_freq > 0.7:
        return (255, 0, 0)  # Red for high frequency
    elif norm_freq > 0.4:
        return (0, 0, 255)  # Blue for medium
    else:
        return (128, 128, 128)  # Gray for low

twc = TrueWordCloud(values=values, color_func=color_func)
```

### All Parameters

```python
twc = TrueWordCloud(
    values={'word': 100, 'cloud': 50},  # Required: word -> value mapping
    method='greedy',                     # 'greedy' or 'square'
    base_font_size=100,                  # Font size for max value word
    font_path='/path/to/font.ttf',       # Custom font (auto-detected if None)
    min_font_size=10,                    # Minimum font size
    background_color=(255, 255, 255),    # RGB tuple
    margin=2,                            # Pixels between words
    color_func=None                      # Custom color function
)

# Generate with statistics
image, stats = twc.generate_with_stats()
print(stats)  # {'num_words': 2, 'size_range': (50, 100), 'canvas_size': (800, 600), 'method': 'greedy'}
```

## Comparison with Traditional Word Clouds

| Feature | TrueWordCloud | Traditional Word Clouds |
|---------|---------------|------------------------|
| Proportionality | ✅ Strict (font_size ∝ value) | ❌ Arbitrary resizing to fit |
| Canvas Size | Dynamic (fits content) | Fixed (pre-defined) |
| Reproducibility | ✅ Greedy method | Sometimes |
| Layout Options | 2 algorithms | Usually 1 |
| Value Types | Any numeric | Usually just frequencies |

## Why True Proportionality Matters

Traditional word clouds often **lie** about the data:
- A word with value 100 might be rendered at 80pt
- A word with value 50 might be rendered at 75pt
- Ratios like 2:1 become 1.07:1

**TrueWordCloud guarantees:**
- Value 100 → 100pt, Value 50 → 50pt
- Ratios are preserved: 2:1 stays 2:1
- Visual size accurately represents data magnitude

## Use Cases

- **Linguistic Analysis** - Word frequencies, keyness scores, TF-IDF
- **Survey Results** - Response counts, satisfaction scores
- **Scientific Papers** - Maintaining accurate proportional relationships
- **Marketing** - Brand mentions, sentiment scores
- **Education** - Concept importance, study time allocation

## Requirements

- Python 3.7+
- Pillow (PIL)
- numpy
- matplotlib (for font detection)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use TrueWordCloud in academic work, please cite:

```
@software{truewordcloud2026,
  title={TrueWordCloud: Value-Proportional Word Cloud Generator},
  author={Laurence Anthony},
  year={2026},
  url={https://github.com/laurenceanthony/truewordcloud}
}
```

## Examples

### Frequency Data
```python
word_frequencies = {
    'the': 1000, 'Python': 500, 'data': 400, 'analysis': 300,
    'machine': 250, 'learning': 250, 'algorithm': 200
}
twc = TrueWordCloud(values=word_frequencies, method='greedy')
twc.generate().save('frequencies.png')
```

### Keyness Scores
```python
keyness_scores = {
    'significant': 12.5, 'analysis': 8.3, 'corpus': 6.7,
    'frequency': 5.2, 'text': 4.1
}
twc = TrueWordCloud(values=keyness_scores, method='square', base_font_size=50)
twc.generate().save('keyness.png')
```

### With Custom Styling
```python
from PIL import ImageColor

def rainbow_color(word, freq, norm_freq):
    # Rainbow gradient based on frequency
    hue = int(norm_freq * 270)  # 0 (red) to 270 (blue)
    return ImageColor.getrgb(f'hsl({hue}, 100%, 50%)')

twc = TrueWordCloud(
    values=word_frequencies,
    method='square',
    color_func=rainbow_color,
    background_color=(0, 0, 0),  # Black background
    margin=5
)
twc.generate().save('rainbow.png')
```

## FAQ

**Q: Why are the layouts different sizes?**  
A: Canvas size is determined by content. More words or higher values = larger canvas. This maintains true proportions.

**Q: Can I fix the canvas size?**  
A: Not directly, as that would require resizing words (breaking true proportionality). Instead, adjust `base_font_size` to control overall scale.

**Q: Which method should I use?**  
A: Use `greedy` for speed and reproducibility. Use `square` for compact layouts and visual variety.

**Q: How do I make words fit in a specific area?**  
A: Reduce `base_font_size` until the generated canvas is the desired size.

---

**Made with ❤️ for accurate data visualization**
