# TrueWordCloud - GitHub Repository Setup Guide

## Files Created for GitHub Repository

The following files have been created for your TrueWordCloud package:

### Core Files
1. **truewordcloud.py** - Main library with both greedy and square algorithms
2. **README_TrueWordCloud.md** - Complete documentation (rename to README.md for GitHub)
3. **setup_truewordcloud.py** - setuptools configuration (rename to setup.py)
4. **LICENSE_TrueWordCloud.txt** - MIT License (rename to LICENSE)
5. **.gitignore_truewordcloud** - Python gitignore (rename to .gitignore)
6. **examples_truewordcloud.py** - Example usage demonstrations

## Setting Up GitHub Repository

### Step 1: Create Repository Structure

Create a new directory for the repository:
```bash
mkdir truewordcloud
cd truewordcloud
```

### Step 2: Copy and Rename Files

Copy the files and rename them:
```bash
# Core library
cp truewordcloud.py truewordcloud.py

# Documentation
cp README_TrueWordCloud.md README.md

# Package setup
cp setup_truewordcloud.py setup.py

# License
cp LICENSE_TrueWordCloud.txt LICENSE

# Git configuration
cp .gitignore_truewordcloud .gitignore

# Examples
cp examples_truewordcloud.py examples.py
```

### Step 3: Update Personal Information

Edit these files with your information:

**setup.py:**
```python
author="Your Name",
author_email="your.email@example.com",
url="https://github.com/yourusername/truewordcloud",
```

**README.md:**
- Update GitHub username in URLs
- Update citation author name

**LICENSE:**
- Replace `[Your Name]` with your actual name

### Step 4: Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: TrueWordCloud v1.0.0"
```

### Step 5: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `truewordcloud`
3. Description: "Value-proportional word cloud generator with true size relationships"
4. Public repository
5. Don't initialize with README (we already have one)

### Step 6: Push to GitHub

```bash
git remote add origin https://github.com/yourusername/truewordcloud.git
git branch -M main
git push -u origin main
```

## Testing Before Publishing

### Test the Package Locally

```bash
# Install in development mode
pip install -e .

# Run tests
python truewordcloud.py

# Run examples
python examples.py
```

### Test Installation from GitHub

```bash
pip install git+https://github.com/yourusername/truewordcloud.git
```

## Publishing to PyPI (Optional)

### Prerequisites
```bash
pip install build twine
```

### Build Distribution
```bash
python -m build
```

### Upload to PyPI
```bash
# Test PyPI first
twine upload --repository testpypi dist/*

# If successful, upload to real PyPI
twine upload dist/*
```

## Recommended GitHub Repository Structure

```
truewordcloud/
├── truewordcloud.py          # Main library
├── README.md                  # Documentation
├── LICENSE                    # MIT License
├── setup.py                   # Package configuration
├── .gitignore                 # Git ignore rules
├── examples.py                # Usage examples
├── requirements.txt           # Dependencies (optional)
└── tests/                     # Unit tests (future)
    └── test_truewordcloud.py
```

## Requirements.txt (Optional)

Create requirements.txt:
```
Pillow>=8.0.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## Future Enhancements

Consider adding:
- [ ] Unit tests (pytest)
- [ ] GitHub Actions for CI/CD
- [ ] More example images in README
- [ ] Performance benchmarks
- [ ] Jupyter notebook tutorials
- [ ] Documentation with Sphinx

## Using in AntConc

After the package is on GitHub and working:

1. Install from GitHub in AntConc environment:
   ```bash
   pip install git+https://github.com/yourusername/truewordcloud.git
   ```

2. Import in AntConc:
   ```python
   from truewordcloud import TrueWordCloud
   
   # In your word frequency export function:
   twc = TrueWordCloud(values=word_dict, method='greedy')
   image = twc.generate()
   ```

## Support and Documentation

Once published:
- Enable GitHub Issues for bug reports
- Enable GitHub Discussions for questions
- Add CONTRIBUTING.md for contributors
- Consider adding CODE_OF_CONDUCT.md

## Version Numbering

Follow semantic versioning (semver.org):
- v1.0.0 - Initial release
- v1.0.x - Bug fixes
- v1.x.0 - New features (backward compatible)
- v2.0.0 - Breaking changes

---

Generated: 2026-02-14
