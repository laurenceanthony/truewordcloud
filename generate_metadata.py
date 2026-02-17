import textwrap

metadata = {
    "name": "truewordcloud",
    "version": "1.2.0",
    "author": "Laurence Anthony",
    "author_email": "anthony@antlabsolutions.com",
    "description": "Value-proportional word cloud generator with true size relationships",
    "url": "https://github.com/laurenceanthony/truewordcloud",
    "python_requires": ">=3.7",
    "install_requires": [
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    "classifiers": [
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
    "keywords": "wordcloud visualization text-analysis frequency linguistics data-visualization",
    "project_urls": {
        "Bug Reports": "https://github.com/laurenceanthony/truewordcloud/issues",
        "Source": "https://github.com/laurenceanthony/truewordcloud",
    },
}

# Generate setup.py
setup_py = f"""
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{metadata['name']}",
    version="{metadata['version']}",
    author="{metadata['author']}",
    author_email="{metadata['author_email']}",
    description="{metadata['description']}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="{metadata['url']}",
    py_modules=["truewordcloud"],
    classifiers={metadata['classifiers']},
    python_requires="{metadata['python_requires']}",
    install_requires={metadata['install_requires']},
    keywords="{metadata['keywords']}",
    project_urls={metadata['project_urls']},
)
""".strip()

# Generate pyproject.toml (PEP 621 style)
pyproject_toml = textwrap.dedent(
    f"""
    [build-system]
    requires = ["setuptools>=61.0", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "{metadata['name']}"
    version = "{metadata['version']}"
    description = "{metadata['description']}"
    readme = "README.md"
    requires-python = "{metadata['python_requires']}"
    license = {{text = "MIT"}}
    authors = [{{name = "{metadata['author']}", email = "{metadata['author_email']}"}}]
    keywords = [{', '.join(f'"{kw.strip()}"' for kw in metadata['keywords'].split())}]
    dependencies = [
        {', '.join(f'"{dep}"' for dep in metadata['install_requires'])}
    ]
    classifiers = [
        {', '.join(f'"{c}"' for c in metadata['classifiers'])}
    ]

    [project.urls]
    BugReports = "{metadata['project_urls']['Bug Reports']}"
    Source = "{metadata['project_urls']['Source']}"
"""
).strip()

with open("setup.py", "w", encoding="utf-8") as f:
    f.write(setup_py + "\n")

with open("pyproject.toml", "w", encoding="utf-8") as f:
    f.write(pyproject_toml + "\n")

print("setup.py and pyproject.toml have been generated and are in sync.")
