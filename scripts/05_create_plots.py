"""Create figures for the uncertainty estimation predictions."""  # pylint: disable=invalid-name

from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import (
    get_model_order_and_color,
    get_nx2_figure,
    get_performance_metrics,
    load_data,
    remove_ax_frame,
    test_set_composition2ax,
    test_set_nn_similarity2ax,
)
from sklearn.calibration import calibration_curve


def plot_test_set_composition(
    data_df: pd.DataFrame, save_path: Path, **kwargs: Any
) -> None:
    """Plot the composition of the test set.

    X axis: Number of negative compounds
    Y axis: Number of positive compounds

    Parameters
    ----------
    data_df : pd.DataFrame
        Data for the endpoint.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    if "figsize" not in kwargs:
        kwargs["figsize"] = (8, 6)
    _, ax = plt.subplots(figsize=kwargs["figsize"])
    handles, labels = test_set_composition2ax(data_df, ax)
    ax.legend(handles, labels, ncol=1)

    ax.set_xlabel("Number of negative compounds")
    ax.set_ylabel("Number of positive compounds")
    plt.savefig(save_path / "test_set_composition.png")


def plot_similarity_to_training(
    data_df: pd.DataFrame, save_path: Path, **kwargs: Any
) -> None:
    """Plot the similarity of the test set compounds to the training set.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data for the endpoint.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    _, ax = plt.subplots(figsize=kwargs["figsize"])
    for split_strategy, iter_df in data_df.groupby("Split strategy"):
        test_set_nn_similarity2ax(iter_df, ax, label=str(split_strategy))

    ax.legend()
    ax.set_xlabel("Similarity to training set")
    ax.set_ylabel("Count")
    plt.savefig(save_path / "similarity_to_training.png")


def plot_metrics(data_df: pd.DataFrame, save_path: Path, **kwargs: Any) -> None:
    """Plot the performance metrics for each model.

    Parameters
    ----------
    data_df : pd.DataFrame
        Predictions for the endpoint.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    model_order, color_dict = get_model_order_and_color()
    performance_df = get_performance_metrics(data_df)
    _, axs, ax_legend = get_nx2_figure(
        figsize=kwargs.get("figsize", None), nrows=1, share_y=False
    )
    for i, metric in enumerate(["Balanced accuracy", "Brier score"]):
        sns.boxplot(
            data=performance_df.loc[performance_df["metric"] == metric],
            x="split",
            hue="model",
            y="Performance",
            ax=axs[i],
            palette=color_dict,
            hue_order=model_order,
        )
        axs[i].set_title(metric)
    handles, labels = axs[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=3)
    axs[0].legend().remove()
    axs[1].legend().remove()
    axs[1].set_ylabel("")
    plt.savefig(save_path / "performance_metrics.png")


def plot_calibration_curves(
    data_df: pd.DataFrame, save_path: Path, **kwargs: Any
) -> None:
    """Plot the calibration curves for each model.

    Parameters
    ----------
    data_df : pd.DataFrame
        Predictions for the endpoint.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    model_color = get_model_order_and_color()[1]
    _, axs, ax_legend = get_nx2_figure(figsize=kwargs.get("figsize", None), nrows=1)
    name2ax_dict = {"Random": axs[0], "Agglomerative clustering": axs[1]}
    for model, color in model_color.items():
        model_df = data_df.loc[data_df["Model name"] == model]
        for split in data_df["Split strategy"].unique():
            split_df = model_df.loc[model_df["Split strategy"] == split]
            prob_true, prob_pred = calibration_curve(
                split_df["label"], split_df["proba"], n_bins=10, strategy="uniform"
            )
            ax = name2ax_dict[split]
            ax.plot(prob_pred, prob_true, label=f"{model}", marker=".", color=color)
    legend, handles = axs[0].get_legend_handles_labels()
    ax_legend.legend(legend, handles, loc="center", ncol=4)
    axs[0].plot((0, 1), (0, 1), ls="--", color="gray")
    axs[1].plot((0, 1), (0, 1), ls="--", color="gray")
    axs[0].set_xlabel("Mean predicted probability (Positive class: 1)")
    axs[1].set_xlabel("Mean predicted probability (Positive class: 1)")
    axs[0].set_ylabel("Fraction of positives (Positive class: 1)")
    axs[0].set_title("Calibration plots for random split")
    axs[1].set_title("Calibration plots for agglomerative clustering split")
    plt.savefig(save_path / "calibration_curves.png")


def plot_proba_rf(data_df: pd.DataFrame, save_path: Path, **kwargs: Any) -> None:
    """Plot the probability distribution for RF models with Morgan and Neural fingerprints.

    Parameters
    ----------
    data_df : pd.DataFrame
        Predictions for the endpoint.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 10)

    _, axs, ax_legend = get_nx2_figure(figsize=kwargs["figsize"], nrows=2)
    models = ["Chemprop", "Neural FP + RF"]
    splits = ["Random", "Agglomerative clustering"]
    row = 0
    for split_name, split_df in data_df.groupby("Split strategy"):
        if split_name not in splits:
            continue
        col = 0
        for model, model_df in split_df.groupby("Model name"):
            if model not in models:
                continue
            sns.histplot(
                data=model_df,
                x="proba",
                ax=axs[row, col],
                label=f"{model}",
                alpha=0.5,
                hue="label",
                bins=np.linspace(0, 1, 20),
                stat="density",
                common_norm=False,
            )
            axs[row, col].set_title(f"{model} - {split_name}")
            if row == 1:
                axs[row, col].set_xlabel("Predicted probability")
            else:
                axs[row, col].set_xlabel("")
            if col == 0:
                axs[row, col].set_ylabel("Density")
            else:
                axs[row, col].set_ylabel("")
            col += 1
        row += 1
    handles, _ = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend().remove()
    axs[0, 1].legend().remove()
    axs[1, 0].legend().remove()
    axs[1, 1].legend().remove()
    ax_legend.legend(handles, ["Active", "Inactive"], loc="center", ncol=4)
    plt.savefig(save_path / "proba_distribution_rf.png")


@click.command()
@click.option(
    "--endpoint", type=str, required=True, help="Endpoint to create figures for."
)
def create_figures(endpoint: str) -> None:
    """Create figures for the uncertainty estimation predictions.

    Parameters
    ----------
    endpoint : str
        Endpoint to create figures for.
    """
    base_path = Path(__file__).parents[1]
    prediction_folder = base_path / "data" / "intermediate_data" / "model_predictions"
    data_df = load_data(endpoint, prediction_folder)

    plot_kwargs = {"figsize": (6, 4)}

    save_path = base_path / "data" / "figures" / endpoint
    save_path.mkdir(parents=True, exist_ok=True)
    plot_test_set_composition(data_df, save_path, **plot_kwargs)
    plot_similarity_to_training(data_df, save_path, **plot_kwargs)
    plot_metrics(data_df, save_path)
    plot_calibration_curves(data_df, save_path)
    plot_proba_rf(data_df, save_path)


if __name__ == "__main__":
    create_figures()  # pylint: disable=no-value-for-parameter
