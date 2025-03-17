from config.base_config import FreqAIConfig
import mlflow
import optuna

def test_imports():
    print("Config import successful")
    print("MLflow version:", mlflow.__version__)
    print("Optuna version:", optuna.__version__)

if __name__ == "__main__":
    test_imports()
