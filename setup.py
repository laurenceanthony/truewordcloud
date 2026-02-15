"""
Setup configuration for TrueWordCloud
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="truewordcloud",
    version="1.1.0",
    author="Laurence Anthony",
    author_email="anthony@antlabsolutions.com",
    description="Value-proportional word cloud generator with true size relationships",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laurenceanthony/truewordcloud",
    py_modules=["truewordcloud"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    keywords="wordcloud visualization text-analysis frequency linguistics data-visualization",
    project_urls={
        "Bug Reports": "https://github.com/laurenceanthony/truewordcloud/issues",
        "Source": "https://github.com/laurenceanthony/truewordcloud",
    },
)
