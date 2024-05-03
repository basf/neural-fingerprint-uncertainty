"""Create figures for the uncertainty estimation predictions."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name


from itertools import product
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import (
    get_nx2_figure,
    get_performance_metrics,
    load_all_data,
    load_data,
    test_set_composition2ax,
    test_set_nn_similarity2ax,
)
from sklearn.calibration import calibration_curve


def plot_test_set_composition(
    data_df_list: list[pd.DataFrame],
    data_name_list: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot the composition of the test sets from both endpoints.

    X axis: Number of negative compounds
    Y axis: Number of positive compounds

    Parameters
    ----------
    data_df_list : list[pd.DataFrame]
        Dataframes with the predictions for the endpoints.
        Each item in the list is a dataframe for an endpoint.
    data_name_list : list[str] | None, optional (default=None)
        Names of the endpoints.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Figure size.
    """
    _, axs, ax_legend = get_nx2_figure(figsize=figsize, share_y=False)
    legend = None
    handles = None
    for i, data_df in enumerate(data_df_list):
        if data_name_list:
            axs[i].set_title(data_name_list[i])
        if i == 0:
            handles, legend = test_set_composition2ax(data_df, axs[i])
        else:
            test_set_composition2ax(data_df, axs[i])
        axs[i].set_xlabel("Number of negative compounds")
        axs[i].set_ylabel("")
    axs[0].set_ylabel("Number of positive compounds")
    if not handles or not legend:
        raise ValueError("No handles and labels found.")
    ax_legend.legend(handles, legend, loc="center", ncol=4)
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path / "test_set_composition.png")


def get_model_order_and_color() -> tuple[list[str], dict[str, str]]:
    """Get the model order and color mapping.

    Returns
    -------
    list[str]
        Model order.
    dict[str, str]
        Model color mapping.
    """
    model_order = [
        "Morgan FP + KNN",
        "Neural FP + KNN",
        "Morgan FP + RF",
        "Neural FP + RF",
        "Morgan FP + SVC",
        "Neural FP + SVC",
        "Chemprop",
    ]
    model_color = {}
    for i, model in enumerate(model_order):
        if model != "Chemprop":
            model_color[model] = sns.color_palette("Paired")[i]
        else:
            model_color[model] = sns.color_palette("Paired")[i + 1]
    return model_order, model_color


def plot_similarity_to_training(
    data_df_list: list[pd.DataFrame],
    save_path: Path,
    col_tiles: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot the similarity of the test set compounds to the training set.

    Parameters
    ----------
    data_df_list : list[pd.DataFrame]
        Dataframes with the predictions for the endpoints.
        Each item in the list is a dataframe for an endpoint.
    save_path : Path
        Path to save the figure.
    col_tiles : list[str] | None, optional (default=None)
        Column titles for the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Figure size.
    """
    _, axs, ax_legend = get_nx2_figure(figsize=figsize, nrows=1)
    handles = None
    labels = None
    for i, data_df in enumerate(data_df_list):
        for _, split in enumerate(["Agglomerative clustering", "Random"]):
            split_df = data_df.loc[data_df["Split strategy"] == split]
            test_set_nn_similarity2ax(split_df, axs[i], split)
        if col_tiles:
            axs[i].set_title(col_tiles[i])
        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()
    for ax in axs:
        ax.legend().remove()
        ax.set_xlabel("Similarity to training set")

    axs[0].set_ylabel("Count")
    axs[1].set_ylabel("")
    axs[1].tick_params(axis="y", which="both", labelleft=False)
    if not handles or not labels:
        raise ValueError("No handles and labels found.")
    ax_legend.legend(handles, labels, loc="center", ncol=4)

    plt.savefig(save_path / "similarity_to_training.png")


def plot_metrics(
    data_df_list: list[pd.DataFrame],
    data_name_list: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot the performance metrics for each model.

    Parameters
    ----------
    data_df_list : list[pd.DataFrame]
        Predictions for the endpoint.
    data_name_list : list[str] | None
        Names of the endpoints.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Figure size.
    """
    model_order, color_dict = get_model_order_and_color()
    (_, subfig_list), axs, ax_legend = get_nx2_figure(
        figsize=figsize, nrows=2, share_y=False
    )
    handles = None
    labels = None
    for i, data_df in enumerate(data_df_list):
        performance_df = get_performance_metrics(data_df)
        if data_name_list:
            subfig_list[i].suptitle(data_name_list[i])
        for j, metric in enumerate(["Balanced accuracy", "Brier score"]):
            ax = axs[i, j]
            sns.boxplot(
                data=performance_df.loc[performance_df["metric"] == metric],
                x="split",
                hue="model",
                y="Performance",
                ax=ax,
                palette=color_dict,
                hue_order=model_order,
            )
            if i == j == 0:
                handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel("")
            ax.legend().remove()
            ax.set_title(metric)
            if j == 0:
                ax.set_ylabel("Metric value")
            else:
                ax.set_ylabel("")
        axs[i, 0].set_ylim([0.45, 0.8])
        axs[i, 1].set_ylim([0.16, 0.32])
        axs[i, 1].set_yticks(np.arange(0.16, 0.35, 0.04))
    if not handles or not labels:
        raise ValueError("No handles and labels found.")
    ax_legend.legend(handles, labels, loc="center", ncol=4)
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path / "performance_metrics.png")


def plot_metrics_scatter(
    base_path: Path,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Create a scatter plot with all endpoints and models.

    The X axis is the balanced accuracy and the Y axis is the Brier score.
    The color represents the model.

    Parameters
    ----------
    base_path : Path
        Path to the data.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
    """
    final_performance_df = load_all_data(base_path)
    final_performance_df = final_performance_df.pivot_table(
        index=["endpoint", "metric", "model"], columns="split", values="Performance"
    ).reset_index()
    model_order, color_dict = get_model_order_and_color()
    _, axs, ax_legend = get_nx2_figure(figsize=figsize, nrows=1, share_y=False)
    for i, metric in enumerate(["Balanced accuracy", "Brier score",]):
        sns.scatterplot(
            data=final_performance_df.loc[final_performance_df["metric"] == metric],
            x="Random",
            y="Agglomerative clustering",
            hue="model",
            palette=color_dict,
            ax=axs[i],
            hue_order=model_order,
        )
        axs[i].set_title(metric)
        axs[i].set_xlabel("Random split")
        axs[i].plot([0, 1], [0, 1], ls="--", color="gray")

    axs[0].set_ylabel("Agglomerative clustering split")
    axs[1].set_ylabel("")
    axs[0].set_xlim([0.5, 1])
    axs[0].set_ylim([0.5, 1])
    axs[1].set_xlim([0, 0.5])
    axs[1].set_ylim([0, 0.5])
    handles, labels = axs[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=4)
    axs[0].legend().remove()
    axs[1].legend().remove()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path / "scatter_metrics.png")


def plot_metrics_all(
        base_path: Path,
        save_path: Path | str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> None:
    """Create a boxplot with metrics of all endpoints and models.

    Parameters
    ----------
    base_path : Path
        Path to the data.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Size of the figure.
    """
    all_endpoint_df = load_all_data(base_path)
    model_order, color_dict = get_model_order_and_color()
    _, axs, ax_legend = get_nx2_figure(figsize=figsize, nrows=1, share_y=False)

    for i, metric in enumerate(["Balanced accuracy", "Brier score"]):
        sns.boxplot(
            data=all_endpoint_df.loc[all_endpoint_df["metric"] == metric],
            x="split",
            hue="model",
            y="Performance",
            ax=axs[i],
            palette=color_dict,
            hue_order=model_order,
        )
        axs[i].set_title(metric)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("Metric value")
    handles, labels = axs[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=4)
    axs[0].legend().remove()
    axs[1].legend().remove()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path / "performance_metrics_all.png")




def plot_calibration_curves(
    data_df_list: list[pd.DataFrame],
    data_name_list: list[str] | None,
    save_path: Path,
    **kwargs: Any,
) -> None:
    """Plot the calibration curves for each model.

    Parameters
    ----------
    data_df_list : list[pd.DataFrame]
        Predictions for the endpoints.
    data_name_list : list[str] | None
        Names of the endpoints.
    save_path : Path
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    model_color = get_model_order_and_color()[1]
    (_, subfig_list), axs, ax_legend = get_nx2_figure(
        figsize=kwargs.get("figsize", None), nrows=2
    )
    name2ax_dict = {"Random": 0, "Agglomerative clustering": 1}
    legend = None
    handles = None
    for row, data_df in enumerate(data_df_list):
        if data_name_list:
            subfig_list[row].suptitle(data_name_list[row])
        for split, split_df in data_df.groupby("Split strategy"):
            col = name2ax_dict[split]
            ax = axs[row, col]
            for model, color in model_color.items():
                model_df = split_df.loc[split_df["Model name"] == model]
                prob_true, prob_pred = calibration_curve(
                    model_df["label"], model_df["proba"], n_bins=10, strategy="uniform"
                )

                ax.plot(prob_pred, prob_true, label=f"{model}", marker=".", color=color)
                if row == 0 and split == "Random":
                    legend, handles = ax.get_legend_handles_labels()
            ax.plot((0, 1), (0, 1), ls="--", color="gray")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(f"{split} split")
            if row == 1:
                ax.set_xlabel("Mean predicted probability (Positive class: 1)")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("Fraction of positives (Positive class: 1)")
            else:
                ax.set_ylabel("")

    if not legend or not handles:
        raise ValueError("No legend and handles found.")
    ax_legend.legend(legend, handles, loc="center", ncol=4)
    plt.savefig(save_path / "calibration_curves.png")


def plot_proba_rf(
    data_df_list: list[pd.DataFrame],
    save_path: Path,
    data_titles: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot the probability distribution for RF models with Morgan and Neural fingerprints.

    Parameters
    ----------
    data_df_list : list[pd.DataFrame]
        Predictions for the endpoint.
    save_path : Path
        Path to save the figure.
    data_titles : list[str] | None, optional (default=None)
        Data titles for the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Figure size.
    """

    (_, subfig_list), axs, ax_legend = get_nx2_figure(figsize=figsize, nrows=4)

    models = ["Chemprop", "Neural FP + RF"]
    splits = ["Random", "Agglomerative clustering"]
    if data_titles:
        row_titles = [
            f"{name} - {split} split" for name, split in product(data_titles, splits)
        ]
    else:
        row_titles = None
    handles = None
    for i, (data_df, split) in enumerate(product(data_df_list, splits)):
        if row_titles:
            subfig_list[i].suptitle(row_titles[i])
        for j, model in enumerate(models):
            split_df = data_df.loc[
                (data_df["Model name"] == model) & (data_df["Split strategy"] == split)
            ]
            sns.histplot(
                data=split_df,
                x="proba",
                ax=axs[i, j],
                label=f"{model}",
                alpha=0.5,
                hue="label",
                bins=np.linspace(0, 1, 20),
                stat="density",
                common_norm=False,
            )
            axs[i, j].set_title(f"{model}")
            if i == 1:
                axs[i, j].set_xlabel("Predicted probability")
            else:
                axs[i, j].set_xlabel("")
            if j == 0:
                axs[i, j].set_ylabel("Density")
            else:
                axs[i, j].set_ylabel("")
            axs[i, j].legend().remove()
            if i == 0 and j == 1:
                handles, _ = axs[i, j].get_legend_handles_labels()
    if not handles:
        raise ValueError("No handles and labels found.")
    ax_legend.legend(handles, ["Active", "Inactive"], loc="center", ncol=4)
    plt.savefig(save_path / "proba_distribution_rf.png")


@click.command()
@click.option(
    "--endpoint_a", type=str, required=True, help="Endpoint to create figures for."
)
@click.option(
    "--endpoint_b", type=str, required=True, help="Endpoint to create figures for."
)
def create_figures(endpoint_a: str, endpoint_b: str) -> None:
    """Create figures for the uncertainty estimation predictions.

    Notes
    -----
    The figures are analog to 05_create_figures.py, but contain 2 endpoints.

    Parameters
    ----------
    endpoint_a: str
        Endpoint to create figures for.
    endpoint_b: str
        Endpoint to create figures for.
    """
    base_path = Path(__file__).parents[1]
    prediction_folder = base_path / "data" / "intermediate_data" / "model_predictions"
    data_a_df = load_data(endpoint_a, prediction_folder)
    data_b_df = load_data(endpoint_b, prediction_folder)

    save_path = base_path / "data" / "figures" / "final_figures"
    save_path.mkdir(parents=True, exist_ok=True)
    plot_metrics_scatter(base_path, save_path=save_path, figsize=(8, 4))
    plot_metrics_all(base_path, save_path=save_path, figsize=(8, 4))
    plot_metrics(
        [data_a_df, data_b_df],
        data_name_list=[endpoint_a, endpoint_b],
        save_path=save_path,
    )
    plot_test_set_composition(
        [data_a_df, data_b_df],
        data_name_list=[endpoint_a, endpoint_b],
        save_path=save_path,
    )
    plot_similarity_to_training(
        [data_a_df, data_b_df],
        save_path,
        col_tiles=[endpoint_a, endpoint_b],
    )
    plot_calibration_curves(
        [data_a_df, data_b_df],
        data_name_list=[endpoint_a, endpoint_b],
        save_path=save_path,
    )
    plot_proba_rf(
        [data_a_df, data_b_df],
        data_titles=[endpoint_a, endpoint_b],
        save_path=save_path,
        figsize=(10, 15),
    )


if __name__ == "__main__":
    create_figures()  # pylint: disable=no-value-for-parameter
