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
    get_nxm_figure,
    get_performance_metrics,
    load_data,
    test_set_composition2ax,
    test_set_nn_similarity2ax,
)
from sklearn.calibration import calibration_curve


def plot_data_report(data_df: pd.DataFrame, save_path: Path, **kwargs: Any) -> None:
    """Plot the data report for the endpoint.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data for the endpoint.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    fig, axs = plt.subplots(1, 2, figsize=kwargs.get("figsize", (12, 6)))

    handles, labels = test_set_composition2ax(data_df, axs[0])
    axs[0].legend(handles, labels, ncol=1)

    axs[0].set_xlabel("Number of negative compounds")
    axs[0].set_ylabel("Number of positive compounds")

    for split_strategy, iter_df in data_df.groupby("Split strategy"):
        test_set_nn_similarity2ax(iter_df, axs[1], label=str(split_strategy))
    axs[1].legend()
    axs[1].set_xlabel("Similarity to training set")
    axs[1].set_ylabel("Count")
    fig.tight_layout()
    plt.savefig(save_path / "data_report.png")


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
    _, axs, ax_legend = get_nxm_figure(
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
    ax_legend.legend(handles, labels, loc="center", ncol=4)
    axs[0].legend().remove()
    axs[1].legend().remove()
    axs[1].set_ylabel("")
    plt.savefig(save_path / "performance_metrics.png")


def plot_calibration_curves(  # pylint: disable=too-many-locals
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
    _, axs, ax_legend = get_nxm_figure(figsize=kwargs.get("figsize", None), nrows=1)
    name2ax_dict = {"Random": axs[1], "Agglomerative clustering": axs[0]}
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
    axs[0].set_title("Random split")
    axs[1].set_title("Agglomerative clustering split")
    plt.savefig(save_path / "calibration_curves.png")


def plot_proba_chemprop(data_df: pd.DataFrame, save_path: Path, **kwargs: Any) -> None:
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

    (_, subfigs), axs, ax_legend = get_nxm_figure(figsize=kwargs["figsize"], nrows=2, share_y=False)
    models = ["Chemprop", "Cal. Chemprop"]
    splits = ["Random", "Agglomerative clustering"]
    for row, split_name in enumerate(splits):
        split_df = data_df.loc[data_df["Split strategy"] == split_name]
        subfigs[row].suptitle(f"{split_name} split")
        col = 0
        for model in models:
            model_df = split_df.loc[split_df["Model name"] == model]
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
            axs[row, col].set_title(f"{model}")
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
    plt.savefig(save_path / "proba_distribution_chemprop.png")


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

    (_, subfigs), axs, ax_legend = get_nxm_figure(figsize=kwargs["figsize"], nrows=2, share_y=False)
    models = ["Morgan FP + RF", "Neural FP + RF"]
    splits = ["Random", "Agglomerative clustering"]
    row = 0
    for row, split_name in enumerate(splits):
        split_df = data_df.loc[data_df["Split strategy"] == split_name]
        subfigs[row].suptitle(f"{split_name} split")
        col = 0
        for model in models:
            model_df = split_df.loc[split_df["Model name"] == model]
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
            axs[row, col].set_title(f"{model}")
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

    plot_kwargs = {"figsize": (8, 3.5)}

    save_path = base_path / "data" / "figures" / endpoint
    save_path.mkdir(parents=True, exist_ok=True)
    plot_metrics(data_df, save_path)
    plot_calibration_curves(data_df, save_path)
    plot_proba_chemprop(data_df, save_path)
    plot_proba_rf(data_df, save_path)
    plot_data_report(data_df, save_path, **plot_kwargs)


if __name__ == "__main__":
    create_figures()  # pylint: disable=no-value-for-parameter
