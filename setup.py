from setuptools import setup, find_packages

setup(
    name='freqai_optimization',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'ccxt',
        'pandas',
        'scikit-learn',
        'lightgbm',
        'ta-lib',
        'mlflow',
        'optuna'
    ],
)
