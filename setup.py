from setuptools import setup, find_packages

setup(
    name="sheaf-transformers",
    version="0.2.0",
    author="Ayman Machhidan",
    description="Sheaf-theoretic diagnostics and layers for Transformer attention",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ayman-machhidan/sheaf-transformers",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        "hf": ["torch>=2.0", "transformers>=4.30"],
        "dev": ["pytest>=7.0"],
        "bench": ["datasets>=2.14", "tqdm>=4.65"],
        "viz": ["matplotlib>=3.5"],
        "all": ["torch>=2.0", "transformers>=4.30", "datasets>=2.14",
                "tqdm>=4.65", "matplotlib>=3.5", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "sheaf-transformers=sheaf_transformers.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
