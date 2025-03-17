import matplotlib.pyplot as plt
import optuna

def plot_optimization_results(study):
    """
    Create visualizations for optimization results
    
    :param study: Optuna study object
    """
    # Hyperparameter Importance
    plt.figure(figsize=(10, 6))
    importance = optuna.visualization.plot_param_importances(study)
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    plt.savefig('hyperparameter_importance.png')
    plt.close()

    # Optimization History
    plt.figure(figsize=(10, 6))
    history = optuna.visualization.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig('optimization_history.png')
    plt.close()

    print("Visualization images saved: ")
    print("1. hyperparameter_importance.png")
    print("2. optimization_history.png")
