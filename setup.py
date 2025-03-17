from setuptools import setup, find_packages

setup(
    name="freqai_optimization",
    version="0.1.0",
    description="Advanced feature generation and risk management for FreqAI",
    author="Henry Tomlinson",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "ta>=0.10.0",
        "lightgbm>=3.3.0",
        "mlflow>=2.0.0",
        "optuna>=3.0.0",
        "ccxt>=4.0.0",
        "freqtrade>=2024.1"
    ],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
