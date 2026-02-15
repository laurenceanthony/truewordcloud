# TrueWordCloud v1.1.0 Release Notes

## 🚀 New Features

## Outline
A word cloud generator that maintains TRUE proportional relationships between values. Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud ensures font sizes are ALWAYS proportional to the input values.

- **Mask Support**: Constrain word placement using custom mask images (black=allowed, white=forbidden).
- **Color Masks**: Assign word colors from an image mask for visually rich word clouds.
- **Mask Outline**: Optionally overlay the mask outline on the generated word cloud for enhanced shape visibility.
- **Detailed Statistics**: Use `generate_with_stats()` to obtain placement and layout statistics.
- **Three Layout Algorithms**: Added 'distance_transform' for most compact, mask-constrained layouts. Default layout is now 'distance_transform'.

## 🛠️ Improvements

- **Font Fallback**: Improved font handling; falls back to PIL default if custom font not found.
- **Documentation**: Updated README.md with new features, usage examples, and FAQ.
- **Packaging**: Removed matplotlib dependency, added scipy for advanced layout algorithms.
- **Metadata Synchronization**: Provided script to autogenerate setup.py and pyproject.toml for consistent versioning and metadata.

## 🐛 Bug Fixes

- Fixed AttributeError in TrueWordCloud when using mask/color mask features.
- Improved mask outline rendering for better visual quality.
- Resolved versioning conflicts between setup.py and pyproject.toml.

## 📦 Requirements

- Python 3.7+
- Pillow (PIL)
- numpy
- scipy

## 📚 Documentation

- See updated README.md for feature details, usage examples, and FAQ.

---

**Made with ❤️ for accurate data visualization**
