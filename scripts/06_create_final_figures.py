"""Create figures for the uncertainty estimation predictions."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name


from itertools import product
from pathlib import Path
from typing import Any, Literal

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import (
    DEFAULT_IMAGE_FORMAT,
    get_endpoint_list,
    get_model_order_and_color,
    get_nxm_figure,
    get_performance_metrics,
    load_all_performances,
    test_set_composition2ax,
    test_set_nn_similarity2ax,
)
from scipy.stats import mannwhitneyu
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
    _, axs, ax_legend = get_nxm_figure(figsize=figsize, share_y=False)
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
        plt.savefig(save_path / f"test_set_composition.{DEFAULT_IMAGE_FORMAT}")


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
    _, axs, ax_legend = get_nxm_figure(figsize=figsize, nrows=1)
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

    plt.savefig(save_path / f"similarity_to_training.{DEFAULT_IMAGE_FORMAT}")


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
    (_, subfig_list), axs, ax_legend = get_nxm_figure(
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
        plt.savefig(save_path / f"performance_metrics.{DEFAULT_IMAGE_FORMAT}")


def plot_metrics_scatter(
    base_path: Path,
    comparison: Literal["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural"],
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
    final_performance_df = load_all_performances(base_path, comparison=comparison)
    final_performance_df = final_performance_df.pivot_table(
        index=["endpoint", "metric", "model"], columns="split", values="Performance"
    ).reset_index()
    model_order, color_dict = get_model_order_and_color(comparison=comparison)
    _, axs, ax_legend = get_nxm_figure(figsize=figsize, nrows=1, share_y=False)
    for i, metric in enumerate(
        [
            "Balanced accuracy",
            "Brier score",
        ]
    ):
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
    ax_legend.legend(handles, labels, loc="center", ncol=len(labels) // 2)
    axs[0].legend().remove()
    axs[1].legend().remove()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path / f"scatter_metrics_{comparison}.{DEFAULT_IMAGE_FORMAT}")


def plot_metrics_scatter_encoding(
    base_path: Path,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Create a scatter plot comparing the performance of models with different encodings.

    The X axis is the performance with the binary Morgan fingerprint and the Y axis is the
    performance with the counted Morgan fingerprint.

    Parameters
    ----------
    base_path : Path
        Path to the data.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Size of the figure.
    """
    full_performance_df = load_all_performances(
        base_path, comparison="morgan_vs_counted"
    )
    for metric in ["Balanced accuracy", "Brier score"]:
        metric_df = full_performance_df.loc[full_performance_df["metric"] == metric]
        metric_df = metric_df.pivot_table(
            index=["endpoint", "split", "base_model"],
            columns="encoding",
            values="Performance",
        ).reset_index()
        _, axs, ax_legend = get_nxm_figure(figsize=figsize, nrows=1, share_y=True)
        for i, split in enumerate(
            [
                "Agglomerative clustering",
                "Random",
            ]
        ):
            sns.scatterplot(
                data=metric_df.loc[metric_df["split"] == split],
                x="Binary Morgan FP",
                y="Counted Morgan FP",
                hue="base_model",
                ax=axs[i],
            )
            axs[i].set_title(split)
            axs[i].set_xlabel("Binary Morgan FP")
            axs[i].plot([0, 1], [0, 1], ls="--", color="gray")

        axs[0].set_ylabel("Counted Morgan FP")
        axs[1].set_ylabel("")
        if metric == "Balanced accuracy":
            lim = [0.5, 1]
        elif metric == "Brier score":
            lim = [0, 0.5]
        else:
            raise ValueError("Metric not recognized.")
        axs[0].set_xlim(lim)
        axs[0].set_ylim(lim)
        axs[1].set_xlim(lim)
        handles, labels = axs[0].get_legend_handles_labels()
        ax_legend.legend(handles, labels, loc="center", ncol=4)
        axs[0].legend().remove()
        axs[1].legend().remove()
        if save_path:
            save_path = Path(save_path)
            metric_str = metric.lower().replace(" ", "_")
            plt.savefig(
                save_path
                / f"{metric_str}_scatter_counted_binary_fp.{DEFAULT_IMAGE_FORMAT}"
            )


def plot_metrics_all(
    base_path: Path,
    comparison: Literal["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural"],
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
    all_endpoint_df = load_all_performances(base_path, comparison=comparison)
    model_order, color_dict = get_model_order_and_color(comparison=comparison)
    _, axs, ax_legend = get_nxm_figure(figsize=figsize, ncols=1, nrows=3, share_y=False)

    for i, metric in enumerate(["Balanced accuracy", "Brier score", "Log loss"]):
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
    ax_legend.legend(handles, labels, loc="center", ncol=len(labels) // 2)
    axs[0].legend().remove()
    axs[1].legend().remove()
    axs[2].legend().remove()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(
            save_path / f"performance_metrics_all_{comparison}.{DEFAULT_IMAGE_FORMAT}"
        )


def plot_precision_recall(
    base_path: Path,
    comparison: Literal["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural"],
    save_path: Path | str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Create a boxplot with precision and recall for all endpoints and models.

    Parameters
    ----------
    base_path : Path
        Path to the data.
    comparison : Literal["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural"]
        Comparison between models.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Size of the figure.
    """
    all_data_df = load_all_performances(base_path, comparison=comparison)
    model_order, color_dict = get_model_order_and_color(comparison=comparison)
    _, axs, ax_legend = get_nxm_figure(figsize=figsize, ncols=1, nrows=2, share_y=False)

    for i, metric in enumerate(["Precision", "Recall"]):
        sns.boxplot(
            data=all_data_df.loc[all_data_df["metric"] == metric],
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
    ax_legend.legend(handles, labels, loc="center", ncol=len(labels) // 2)
    axs[0].legend().remove()
    axs[1].legend().remove()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path / f"precision_recall_{comparison}.{DEFAULT_IMAGE_FORMAT}")


def compare_chemprop_calibration(
    base_path: Path,
    save_path: Path,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Compare the calibration of the Chemprop models.

    Parameters
    ----------
    base_path : Path
        Path to the data.
    save_path : Path
        Path to save the figure.
    """
    endpoint_list = get_endpoint_list(base_path)
    prediction_path = base_path / "data" / "intermediate_data" / "model_predictions"
    performance_df_list = []
    for endpoint in endpoint_list:
        endpoint_prediction_df_list = []
        for cal_method in ["isotonic", "sigmoid"]:
            prediction_df = pd.read_csv(
                prediction_path
                / f"neural_fingerprint_predictions_{cal_method}_{endpoint}.tsv.gz",
                sep="\t",
            )
            prediction_df = prediction_df.loc[
                prediction_df["model"] == "Calibrated Chemprop"
            ]
            prediction_df["encoding"] = "Neural FP"
            prediction_df["endpoint"] = endpoint
            prediction_df["Model name"] = f"Chemprop {cal_method} calibration"
            endpoint_prediction_df_list.append(prediction_df)
        endpoint_prediction_df = pd.concat(endpoint_prediction_df_list)
        performance_df = get_performance_metrics(endpoint_prediction_df)
        performance_df_list.append(performance_df)
    all_performance_df = pd.concat(performance_df_list)
    metric_list = ["Balanced accuracy", "Brier score", "Log loss"]
    _, axs, ax_legend = get_nxm_figure(
        figsize=figsize, ncols=1, nrows=len(metric_list), share_y=False
    )
    for i, metric in enumerate(metric_list):
        sns.boxplot(
            data=all_performance_df.loc[all_performance_df["metric"] == metric],
            x="split",
            hue="model",
            y="Performance",
            ax=axs[i],
        )
        axs[i].set_title(metric)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("Metric value")
    axs[0].set_ylabel("Balanced accuracy")
    axs[1].set_ylabel("Brier score")
    axs[2].set_ylabel("Log loss")
    handles, labels = axs[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=2)
    axs[0].legend().remove()
    axs[1].legend().remove()
    axs[2].legend().remove()

    plt.savefig(save_path / f"chemprop_calibration.{DEFAULT_IMAGE_FORMAT}")


def plot_significance_matrix(
    base_path: Path,
    comparison: Literal["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural"],
    save_path: Path | str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Compare if metrics are significantly different between models.

    Parameters
    ----------
    base_path : Path
        Path to the data.
    comparison : Literal["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural"]
        Comparison between models.
    save_path : Path | str | None, optional (default=None)
        Path to save the figure.
    figsize : tuple[int, int] | None, optional (default=None)
        Size of the figure.
    """
    all_endpoint_df = load_all_performances(base_path, comparison=comparison)
    model_order, _ = get_model_order_and_color(comparison=comparison)
    (_, subfigures), axs, ax_legend = get_nxm_figure(
        figsize=figsize, ncols=2, nrows=3, share_y=False
    )

    for i, metric in enumerate(["Balanced accuracy", "Brier score", "Log loss"]):
        metric_df = all_endpoint_df.loc[all_endpoint_df["metric"] == metric]
        if metric == "Balanced accuracy":
            better_site = "greater"
        else:
            better_site = "less"
        for j, split in enumerate(["Agglomerative clustering", "Random"]):
            split_df = metric_df.loc[metric_df["split"] == split]
            significance_matrix = np.ones((len(model_order), len(model_order)))
            for a, model_a in enumerate(model_order):
                model_a_df = split_df.loc[split_df["model"] == model_a]
                for b, model_b in enumerate(model_order):
                    model_b_df = split_df.loc[split_df["model"] == model_b]
                    p_value = mannwhitneyu(
                        model_a_df["Performance"],
                        model_b_df["Performance"],
                        alternative=better_site,
                    ).pvalue
                    significance_matrix[a, b] = p_value

            color_mating = np.zeros((len(model_order), len(model_order), 3))
            for a, _ in enumerate(model_order):
                for b, _ in enumerate(model_order):
                    p_value = significance_matrix[a, b]
                    reverse_p_value = significance_matrix[b, a]
                    if p_value < 0.05 and reverse_p_value < 0.05:
                        raise ValueError("Both p-values are significant.")
                    if p_value < 0.01:
                        # Add tuple with dark green rgb color
                        color_mating[a, b, :] = (0, 121 / 255, 58 / 255)
                    elif p_value < 0.05:
                        # Add tuple with light green rgb color
                        color_mating[a, b, :] = (166 / 255, 208 / 255, 186 / 255)
                    elif reverse_p_value < 0.01:
                        # Add tuple with dark red rgb color
                        color_mating[a, b, :] = (197 / 255, 0, 34 / 255)
                    elif reverse_p_value < 0.05:
                        # Add tuple with light red rgb color
                        color_mating[a, b, :] = (235 / 255, 166 / 255, 178 / 255)
                    else:
                        # Add tuple with white rgb color
                        color_mating[a, b, :] = (1, 1, 1)
            axs[i, j].imshow(
                color_mating,
                extent=[0, len(model_order), len(model_order), 0],
                origin="upper",
            )
            if j == 0:
                axs[i, j].set_yticks(np.arange(len(model_order)) + 0.5)
                y_labels = [f"{model} ({m})" for m, model in enumerate(model_order)]
                axs[i, j].set_yticklabels(y_labels)
            else:
                axs[i, j].set_ylabel("")
                axs[i, j].set_yticks([])
            axs[i, j].set_xticks(np.arange(len(model_order)) + 0.5)
            x_ticks = [f"({m})" for m, model in enumerate(model_order)]
            axs[i, j].set_xticklabels(x_ticks)
            axs[i, j].set_title(split)
        subfigures[i].suptitle(metric)
    row_strong_significant_better = mpatches.Patch(
        color=(0, 121 / 255, 58 / 255),
        label="Row model significantly better (p < 0.01)",
    )
    row_significant_better = mpatches.Patch(
        color=(166 / 255, 208 / 255, 186 / 255),
        label="Row model significantly better (p < 0.05)",
    )
    not_significant = mpatches.Patch(color=(1, 1, 1), label="Not significant")
    row_significant_worse = mpatches.Patch(
        color=(235 / 255, 166 / 255, 178 / 255),
        label="Row model significantly worse (p < 0.05)",
    )
    row_strong_significant_worse = mpatches.Patch(
        color=(197 / 255, 0, 34 / 255), label="Row model significantly worse (p < 0.01)"
    )
    ax_legend.legend(
        [
            row_strong_significant_better,
            row_significant_better,
            not_significant,
            row_significant_worse,
            row_strong_significant_worse,
        ],
        [
            "Row model significantly better (p < 0.01)",
            "Row model significantly better (p < 0.05)",
            "Not significant",
            "Row model significantly worse (p < 0.05)",
            "Row model significantly worse (p < 0.01)",
        ],
        loc="center",
        ncol=1,
    )
    if save_path:
        save_path = Path(save_path)
        plt.savefig(
            save_path / f"significance_plot_{comparison}.{DEFAULT_IMAGE_FORMAT}"
        )


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
    (_, subfig_list), axs, ax_legend = get_nxm_figure(
        figsize=kwargs.get("figsize", None), nrows=2
    )
    name2ax_dict = {"Agglomerative clustering": 0, "Random": 1}
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
    plt.savefig(save_path / f"calibration_curves.{DEFAULT_IMAGE_FORMAT}")


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

    (_, subfig_list), axs, ax_legend = get_nxm_figure(figsize=figsize, nrows=4)

    models = ["Chemprop", "Neural FP + RF"]
    splits = ["Agglomerative clustering", "Random"]
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
    plt.savefig(save_path / f"proba_distribution_rf.{DEFAULT_IMAGE_FORMAT}")


@click.command()
@click.option(
    "--comparison",
    type=click.Choice(
        ["morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural", "other"]
    ),
    required=True,
    help="Comparison to create figures for.",
)
def create_figures(
    comparison: Literal[
        "morgan_vs_neural", "morgan_vs_counted", "counted_vs_neural", "other"
    ]
) -> None:
    """Create figures for the uncertainty estimation predictions."""
    base_path = Path(__file__).parents[1]

    save_path = base_path / "data" / "figures" / "final_figures"
    save_path.mkdir(parents=True, exist_ok=True)
    if comparison == "other":
        compare_chemprop_calibration(base_path, save_path)
        plot_metrics_scatter_encoding(base_path, save_path=save_path, figsize=(8, 4))
    else:
        plot_significance_matrix(
            base_path, save_path=save_path, figsize=(8, 12), comparison=comparison
        )
        plot_metrics_scatter(
            base_path, save_path=save_path, figsize=(8, 4), comparison=comparison
        )
        plot_metrics_all(base_path, save_path=save_path, comparison=comparison)
        plot_precision_recall(base_path, save_path=save_path, comparison=comparison)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    create_figures()
