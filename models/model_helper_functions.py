import numpy as np
import matplotlib.pyplot as plt
import json

def save_history(history_obj, total_time, model_output_dir, MODEL_NAME):
    """
    Saves model history data as a JSON.

    Args:
        history_obj: The history object from the model.
        total_time: The total time spent fitting the model.
        model_output_dir: The model output directory
        MODEL_NAME: The name of the model
    
    Returns:
        A dictionary of the history data
    """

    history_dict = history_obj.history
    history_dict['total_time'] = total_time
    history_dict['lr'] = list(np.array(history_dict['lr']).astype(float))
    json.dump(history_dict, open(f"{model_output_dir}/{MODEL_NAME}/model_history.json", 'w'))

    return history_dict


def map_to_metrics(metrics_names, metric_tuple):
    """
    Maps a returned metric to its name.

    Args:
        metrics_names: The name of the metrics used for model fitting.
        metric_tuple: The tuple of model metrics
    
    Returns:
        A dictionary that maps a metric name to its value from the model.
    """

    return {key:value for key, value in zip(metrics_names, metric_tuple)}


def plot_metric(history:dict, metric_name:str, model_num=None):
    """
    Plots a model metric.

    Args:
        history: The model history dictionary containing the metrics
        metric_name: The name of the metric to plot.
        model_num: The number associated to this model.
    """
    
    label_map = {
        'loss':'Loss',
        'accuracy':'Accuracy',
        'auc':'AUC'
    }
    
    plt.figure(figsize=(7,5))
    plt.plot(history[metric_name], label=metric_name)
    plt.plot(history[f'val_{metric_name}'], label=f'val_{metric_name}')
    if model_num:
        plt.title(f"Model {model_num} Training {label_map[metric_name]} vs Validation {label_map[metric_name]}")
    else:
        plt.title(f"Training {label_map[metric_name]} vs Validation {label_map[metric_name]}")
    plt.xlabel('Epoch')
    plt.ylabel(label_map[metric_name])
    plt.xticks(range(len(history[metric_name])))
    plt.legend()
    plt.show()


def plot_learning_rate(history:dict, model_num=None):
    """
    Plots the learning rate.

    Args:
        history: The model history dictionary containing the metrics
        model_num: The number associated to this model.
    """

    plt.figure(figsize=(7,5))
    plt.plot(history['lr'])
    if model_num:
        plt.title(f'Model {model_num} Learning Rate vs Epoch')
    else:
        plt.title('Learning Rate vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.xticks(range(len(history['lr'])))
    plt.show()
