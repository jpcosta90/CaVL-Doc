from setuptools import setup, find_packages

setup(
    name="cavl-doc",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "tqdm"
    ],
)