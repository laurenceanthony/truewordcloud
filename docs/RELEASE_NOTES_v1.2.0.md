# TrueWordCloud v1.2.0 Release Notes

## Outline
A word cloud generator that maintains TRUE proportional relationships between values. Unlike traditional word clouds that arbitrarily resize words to fit a canvas, TrueWordCloud ensures font sizes are ALWAYS proportional to the input values.
This release introduces a major refactor to the TrueWordCloud codebase, focusing on clarity, consistency, and ease of use. The application now features a unified naming scheme and improved parameter management, making the API more intuitive and less confusing for users.

## Changes

### Refactoring and Naming Scheme
- The codebase was refactored to adopt a consistent naming scheme across all classes, methods, and parameters.
- All configuration parameters are now set in the `__init__` constructor, eliminating confusion between initialization and generation steps.

### Parameter Management Improvements
- Redundant and unused parameters were removed for a cleaner and more maintainable API.
- Mask-related and layout-related options are now clearly grouped and documented.

### Documentation Updates
- All documentation, including README and docstrings, was updated to reflect the new API and naming conventions.
- Examples and usage guides now match the refactored implementation.

## Upgrade Notes
- This release may require minor code changes for users who previously set parameters outside of the constructor or used deprecated options.
- Please review the updated documentation for migration guidance.

---
For more details, see the documentation and previous release notes.