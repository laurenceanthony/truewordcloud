# TrueWordCloud v1.0.0 Release Notes

## 🚀 Initial Release

## Outline
A word cloud generator that maintains TRUE proportional relationships between values. Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud ensures font sizes are ALWAYS proportional to the input values.


- **True Proportionality**: Font sizes strictly proportional to input values (no squeezing/normalization).
- **Greedy Spiral Layout**: Fast, deterministic spiral placement from center outward.
- **Square Packing Layout**: Compact, randomized square packing for visual variety.
- **Dynamic Canvas**: Canvas size determined by content, not pre-fixed dimensions.
- **Any Numeric Values**: Supports frequencies, keyness scores, TF-IDF, probabilities, etc.
- **No Overlaps**: Guaranteed non-overlapping word placement.
- **Custom Colors**: Flexible color function support.

## 🛠️ Improvements

- **Flexible Parameters**: Control font size, margin, background color, and more.
- **Custom Font Support**: Use any TTF font for word rendering.

## 📦 Requirements

- Python 3.7+
- Pillow (PIL)
- numpy
- matplotlib

## 📚 Documentation

- See README.md for feature details, usage examples, and FAQ.

---

**Made with ❤️ for accurate data visualization**
