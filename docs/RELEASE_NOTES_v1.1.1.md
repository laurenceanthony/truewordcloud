# TrueWordCloud v1.1.1 Release Notes

## Outline
A word cloud generator that maintains TRUE proportional relationships between values. Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud ensures font sizes are ALWAYS proportional to the input values.
This release focuses on improvements to mask handling and layout algorithms, resulting in more accurate and visually appealing word clouds.

## Changes

### Improved Mask Handling
- Adjusted mask processing to ensure better alignment between the mask outline and the generated word cloud.
- Fixed edge mapping issues so mask boundaries are now faithfully represented in the output image.

### Greedy Spiral Layout Enhancements
- Refined the greedy spiral placement algorithm to reduce empty space and produce more compact word clouds.
- Words are now packed more efficiently, resulting in denser and more visually balanced layouts.

## Bug Fixes
- Resolved minor issues related to mask outline clipping and coordinate mapping.

## Upgrade Notes
No breaking changes. Existing usage remains compatible.

---
For more details, see the documentation and previous release notes.
