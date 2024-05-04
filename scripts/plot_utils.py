"""Utility functions for plotting the results of the benchmarking."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.figure import SubFigure
from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators.similarity_transformation import TanimotoToTraining
from molpipeline.mol2any import MolToMorganFP
from sklearn.metrics import balanced_accuracy_score, brier_score_loss


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
        "Calibrated Chemprop",
        "Chemprop",
    ]
    model_color = {}
    for i, model in enumerate(model_order):
        if model != "Chemprop":
            model_color[model] = sns.color_palette("Paired")[i]
        else:
            model_color[model] = sns.color_palette("Paired")[i + 1]
    return model_order, model_color


def get_sim_pipeline() -> Pipeline:
    """Get the similarity pipeline."""
    sim_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            ("mol2morgan", MolToMorganFP()),
            ("sim", TanimotoToTraining()),
        ],
        n_jobs=1,
    )
    return sim_pipeline


def remove_ax_frame(ax: plt.Axes) -> None:
    """Remove the frame of the axis.

    Inplace operation.

    Parameters
    ----------
    ax : plt.Axes
        Axis to remove the frame
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def get_nx2_figure(  # pylint: disable=too-many-variables
    figsize: tuple[int, int] | None = None, nrows: int = 1, share_y: bool = True
) -> tuple[tuple[plt.Figure, list[SubFigure]], npt.NDArray[plt.Axes], plt.Axes]:  # type: ignore
    """Get a figure with n rows, 2 cols, and a legend axis.

    Parameters
    ----------
    figsize : tuple[int, int], optional (default=None)
        Figure size. If None, the default size is (10, 5).
    nrows : int, optional (default=1)
        Number of rows.
    share_y : bool, optional (default=True)
        Whether to share the y-axis between the two columns.

    Returns
    -------
    plt.Figure
        The figure containing the axes.
    npt.NDArray[[plt.Axes]
        Array of axes.
    plt.Axes
        The legend axis.
    """
    if figsize is None:
        figsize = (8, 4 * nrows)
    fig = plt.figure(layout="constrained", figsize=figsize)
    row_height = 9 * nrows
    if nrows > 1:
        row_height += 1
    gs = plt.GridSpec(row_height, 1, figure=fig)
    ax_list = []
    sub_fig_list = []
    for i in range(nrows):
        start_row = 9 * i
        end_row = 9 * (i + 1) - 1
        sub_fig = fig.add_subfigure(gs[start_row:end_row, :])
        sub_fig_list.append(sub_fig)
        sub_spec = sub_fig.add_gridspec(1, 2)
        ax_0 = sub_fig.add_subplot(sub_spec[0, 0])
        if share_y:
            ax_1 = sub_fig.add_subplot(sub_spec[0, 1], sharey=ax_0)
        else:
            ax_1 = sub_fig.add_subplot(sub_spec[0, 1])
        ax_list.append([ax_0, ax_1])
    ax_legend = fig.add_subplot(gs[-1:, :])
    remove_ax_frame(ax_legend)
    return (fig, sub_fig_list), np.array(ax_list).squeeze(), ax_legend


def assert_same_smiles_set(data_df1: pd.DataFrame, data_df2: pd.DataFrame) -> None:
    """Check if the dataframes have the same smiles set.

    This is to ensure that the predictions are for the same compounds.

    Parameters
    ----------
    data_df1 : pd.DataFrame
        First dataframe.
    data_df2 : pd.DataFrame
        Second dataframe.

    Raises
    ------

    """
    smiles_set1 = set(data_df1["smiles"].tolist())
    smiles_set2 = set(data_df2["smiles"].tolist())

    if smiles_set1 != smiles_set2:
        endpoint_name = data_df1["endpoint"].iloc[0]
        model = data_df1["model"].iloc[0]
        split_method = data_df1["Split strategy"].iloc[0]
        trial = data_df1["trial"].iloc[0]
        raise ValueError(
            f"The smiles sets for the neural and morgan predictions are not the same"
            f" for {endpoint_name}, {model}, {split_method}, {trial}."
        )


def load_data(endpoint: str, prediction_folder: Path) -> pd.DataFrame:
    """Load the data for the endpoint.

    Parameters
    ----------
    endpoint : str
        Endpoint to load data for.
    prediction_folder : Path
        Folder with the predictions.

    Returns
    -------
    pd.DataFrame
        Data for the endpoint.
    """
    nnfp_prediction_df = pd.read_csv(
        prediction_folder / f"neural_fingerprint_predictions_{endpoint}.tsv.gz",
        sep="\t",
    )
    nnfp_prediction_df["encoding"] = "Neural FP"
    morganfp_prediction_df = pd.read_csv(
        prediction_folder / f"morgan_fingerprint_predictions_{endpoint}.tsv.gz",
        sep="\t",
    )
    morganfp_prediction_df["encoding"] = "Morgan FP"
    split_columns = ["endpoint", "model", "Split strategy", "trial"]
    for idx_list, iter_df in nnfp_prediction_df.groupby(split_columns):
        morgan_iter_df = morganfp_prediction_df
        for col_name, col_value in zip(split_columns, idx_list):  # type: ignore
            morgan_iter_df = morgan_iter_df.loc[morgan_iter_df[col_name] == col_value]

        if "Chemprop" not in idx_list[1]:  # Chemprop has no corresponding Morgan model
            assert_same_smiles_set(iter_df, morgan_iter_df)
    endpoint_df = pd.concat([nnfp_prediction_df, morganfp_prediction_df])
    endpoint_df["Model name"] = endpoint_df[["encoding", "model"]].apply(
        " + ".join, axis=1
    )
    endpoint_df.loc[endpoint_df["model"] == "Chemprop", "Model name"] = "Chemprop"
    endpoint_df.loc[endpoint_df["model"] == "Calibrated Chemprop", "Model name"] = (
        "Calibrated Chemprop"
    )
    return endpoint_df


def load_all_data(base_path: Path) -> pd.DataFrame:
    """Load all the data for all endpoints.

    Parameters
    ----------
    base_path : Path
        Base path of the project.

    Returns
    -------
    pd.DataFrame
        Data for all endpoints.
    """
    prediction_folder = base_path / "data" / "intermediate_data" / "model_predictions"
    config_path = base_path / "config" / "endpoints.yaml"
    # Load endpoints from yaml
    with open(config_path, "r", encoding="UTF-8") as file:
        endpoint_list = yaml.safe_load(file)["endpoint"]
    performance_df_list = []
    for endpoint in endpoint_list:
        data_df = load_data(endpoint, prediction_folder)
        performance_df = get_performance_metrics(data_df)
        performance_df["endpoint"] = endpoint
        performance_df_list.append(performance_df)
    final_performance_df = pd.concat(performance_df_list)
    return final_performance_df


def get_test_set_compounds(data_df: pd.DataFrame) -> pd.DataFrame:
    """Get the compounds used as test set.

    Gets

    Parameters
    ----------
    data_df : pd.DataFrame
        Data for the endpoint.

    Returns
    -------
    pd.DataFrame
        Dataframe with the test set compounds.
    """
    test_set_df = data_df[
        ["smiles", "label", "trial", "Split strategy"]
    ].drop_duplicates()
    for split_stragegy, split_df in test_set_df.groupby("Split strategy"):
        for trial, test_df in split_df.groupby("trial"):
            training_df = split_df.loc[split_df["trial"] != trial]
            training_smiles = training_df["smiles"].tolist()
            test_smiles = test_df["smiles"].tolist()
            intersection = set(training_smiles).intersection(set(test_smiles))
            if intersection:
                raise ValueError(
                    f"Test set for {split_stragegy}, {trial} contains compounds "
                    f"also in the training set."
                )
    return test_set_df


def test_set_composition2ax(
    data_df: pd.DataFrame, ax: plt.Axes
) -> tuple[list[plt.Artist], list[Any]]:
    """Plot the composition of the test set.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data for the endpoint.
    ax : plt.Axes
        Axes to plot the composition.
    """
    test_set_cpds = get_test_set_compounds(data_df)
    count_df = test_set_cpds.pivot_table(
        index=["trial", "Split strategy"],
        values="smiles",
        columns="label",
        aggfunc="nunique",
    )
    count_df.reset_index(inplace=True)
    pos_total, neg_total = count_df.query("`Split strategy` == 'Random'")[[1, 0]].sum(
        axis=0
    )
    n_trials = test_set_cpds.trial.nunique()
    sns.scatterplot(
        data=count_df,
        x=0,
        y=1,
        hue="Split strategy",
        ax=ax,
        hue_order=["Agglomerative clustering", "Random"],
    )
    ax.scatter(
        [neg_total / n_trials],
        [pos_total / n_trials],
        label="Perfect split",
        marker="x",
        color="k",
    )
    radius_10_cpds = plt.Circle(
        (neg_total / n_trials, pos_total / n_trials), 10, fill=False
    )

    ax.axis("equal")
    ax.add_artist(radius_10_cpds)
    handles, legend = ax.get_legend_handles_labels()
    ax.legend().remove()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.plot([0, 5000], [0, 5000], color="grey", ls="--")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    return handles, legend


def test_set_nn_similarity2ax(
    data_df: pd.DataFrame, ax: plt.Axes, label: str | None = None
) -> tuple[list[plt.Artist], list[Any]]:
    """Plot the similarity of the test set compounds to the training set.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data for the endpoint.
    ax : plt.Axes
        Axes to plot the similarity.
    label : str | None, optional (default=None)
        Label for the plot.
    """
    sim_list = []
    for trial in data_df.trial.unique():
        training_smis = data_df.loc[data_df.trial != trial, "smiles"].tolist()
        test_smis = data_df.loc[data_df.trial == trial, "smiles"].tolist()
        sim_pl = get_sim_pipeline()
        sim_pl.fit(training_smis)
        max_sim = sim_pl.transform(test_smis).max(axis=1)
        if len(max_sim) != len(test_smis):
            raise AssertionError()
        sim_list.extend(max_sim.tolist())

    sns.histplot(
        sim_list,
        ax=ax,
        label=label,
        bins=np.linspace(0, 1, 20),
        alpha=0.5,
    )
    ax.set_xlabel("Similarity to training set")
    ax.set_ylabel("Count")

    return ax.get_legend_handles_labels()


def get_performance_metrics(data_df: pd.DataFrame) -> pd.DataFrame:
    """Get the performance metrics for each model.

    Parameters
    ----------
    data_df : pd.DataFrame
        Predictions for the endpoint.

    Returns
    -------
    pd.DataFrame
        Performance metrics for each model and split.
    """
    performance_df_list = []
    for (model, trial, split), iter_df in data_df.groupby(
        ["Model name", "trial", "Split strategy"]
    ):
        iter_dict = {
            "model": model,
            "trial": trial,
            "split": split,
        }
        ba_dict = {
            "metric": "Balanced accuracy",
            "Performance": balanced_accuracy_score(
                iter_df["label"], iter_df["prediction"]
            ),
        }
        brier_dict = {
            "metric": "Brier score",
            "Performance": brier_score_loss(iter_df["label"], iter_df["proba"]),
        }
        for perf_dict in [ba_dict, brier_dict]:
            perf_dict.update(iter_dict)
            performance_df_list.append(perf_dict)
    return pd.DataFrame(performance_df_list)
