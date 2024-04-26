"""Create figures for the uncertainty estimation predictions."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name


from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators.similarity_transformation import TanimotoToTraining
from molpipeline.mol2any import MolToMorganFP
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score, brier_score_loss


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
    endpoint_neural_prediction_df = pd.read_csv(
        prediction_folder / f"neural_fingerprint_predictions_{endpoint}.tsv.gz",
        sep="\t",
    )
    endpoint_neural_prediction_df["encoding"] = "Neural FP"
    endpoint_morgan_prediction_df = pd.read_csv(
        prediction_folder / f"morgan_fingerprint_predictions_{endpoint}.tsv.gz",
        sep="\t",
    )
    endpoint_morgan_prediction_df["encoding"] = "Morgan FP"
    split_columns = ["endpoint", "model", "Split strategy", "trial"]
    for idx_list, iter_df in endpoint_neural_prediction_df.groupby(split_columns):
        endpoint_name, model, split_method, trial = idx_list
        smiles_set = set(iter_df["smiles"].tolist())
        morgan_iter_df = endpoint_morgan_prediction_df
        for col_name, col_value in zip(split_columns, idx_list):  # type: ignore
            morgan_iter_df = morgan_iter_df.loc[morgan_iter_df[col_name] == col_value]

        morgan_smiles_set = set(morgan_iter_df["smiles"].tolist())
        if model != "Chemprop" and smiles_set != morgan_smiles_set:
            raise ValueError(
                f"The smiles sets for the neural and morgan predictions are not the same "
                f"for {endpoint_name}, {model}, {split_method}, {trial}."
            )
    endpoint_df = pd.concat(
        [endpoint_neural_prediction_df, endpoint_morgan_prediction_df]
    )
    endpoint_df["Model name"] = endpoint_df[["encoding", "model"]].apply(
        " + ".join, axis=1
    )
    endpoint_df.loc[endpoint_df["model"] == "Chemprop", "Model name"] = "Chemprop"
    return endpoint_df


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
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
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
    sns.scatterplot(data=count_df, x=0, y=1, hue="Split strategy", ax=ax)
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
    ax.legend()

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.plot([0, 5000], [0, 5000], color="grey", ls="--")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("Number of negative compounds")
    ax.set_ylabel("Number of positive compounds")
    plt.savefig(save_path / "test_set_composition.png")


def get_sim_pipeline() -> Pipeline:
    """Get the similarity pipeline."""
    sim_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            ("mol2morgan", MolToMorganFP()),
            ("sim", TanimotoToTraining()),
        ]
    )
    return sim_pipeline


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
    sim_dict = {}
    test_df = get_test_set_compounds(data_df)
    for split_strategy, iter_df in test_df.groupby("Split strategy"):
        sim_list = []
        for trial in iter_df.trial.unique():
            training_smis = iter_df.loc[iter_df.trial != trial, "smiles"].tolist()
            test_smis = iter_df.loc[iter_df.trial == trial, "smiles"].tolist()
            sim_pl = get_sim_pipeline()
            sim_pl.fit(training_smis)
            max_sim = sim_pl.transform(test_smis).max(axis=1)
            if len(max_sim) != len(test_smis):
                raise AssertionError()
            sim_list.extend(max_sim.tolist())
        sim_dict[split_strategy] = sim_list
    if "figsize" not in kwargs:
        kwargs["figsize"] = (8, 6)
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    for split, sim_list in sim_dict.items():
        sns.histplot(
            sim_list, ax=ax, label=split, bins=np.linspace(0, 1, 20), alpha=0.5
        )

    ax.legend()
    ax.set_xlabel("Similarity to training set")
    ax.set_ylabel("Count")
    plt.savefig(save_path / "similarity_to_training.png")


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
            "metric": "BA",
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
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 5)
    fig = plt.figure(layout="constrained", figsize=kwargs["figsize"])
    gs = plt.GridSpec(10, 2, figure=fig)
    ax_random = fig.add_subplot(gs[0:8, 0])
    ax_agg = fig.add_subplot(gs[0:8, 1], sharey=ax_random)
    ax_legend = fig.add_subplot(gs[8:, :])
    sns.boxplot(
        data=performance_df.query("split == 'Random'"),
        x="metric",
        hue="model",
        y="Performance",
        ax=ax_random,
        palette=color_dict,
        hue_order=model_order,
    )
    sns.boxplot(
        data=performance_df.query("split == 'Agglomerative clustering'"),
        x="metric",
        hue="model",
        y="Performance",
        ax=ax_agg,
        palette=color_dict,
        hue_order=model_order,
    )
    remove_ax_frame(ax_legend)
    handles, labels = ax_random.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=4)
    ax_random.legend().remove()
    ax_agg.legend().remove()
    ax_random.set_title("Random split")
    ax_agg.set_title("Agglomerative clustering split")
    ax_random.set_ylabel("Metric value")
    ax_agg.set_ylabel("")
    plt.savefig(save_path / "performance_metrics.png")


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
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 5)
    fig = plt.figure(layout="constrained", figsize=kwargs["figsize"])
    gs = plt.GridSpec(10, 2, figure=fig)
    ax_random = fig.add_subplot(gs[0:8, 0])
    ax_agglomerative = fig.add_subplot(gs[0:8, 1], sharey=ax_random)
    ax_legend = fig.add_subplot(gs[8:, :])
    remove_ax_frame(ax_legend)
    name2ax_dict = {"Random": ax_random, "Agglomerative clustering": ax_agglomerative}
    for model, color in model_color.items():
        model_df = data_df.loc[data_df["Model name"] == model]
        for split in data_df["Split strategy"].unique():
            split_df = model_df.loc[model_df["Split strategy"] == split]
            prob_true, prob_pred = calibration_curve(
                split_df["label"], split_df["proba"], n_bins=10, strategy="uniform"
            )
            bs = brier_score_loss(split_df["label"], split_df["proba"])
            ax = name2ax_dict[split]
            ax.plot(prob_pred, prob_true, label=f"{model}", marker=".", color=color)
    legend, handles = ax_random.get_legend_handles_labels()
    ax_legend.legend(legend, handles, loc="center", ncol=4)
    ax_random.plot((0, 1), (0, 1), ls="--", color="gray")
    ax_agglomerative.plot((0, 1), (0, 1), ls="--", color="gray")
    ax_random.set_xlabel("Mean predicted probability (Positive class: 1)")
    ax_agglomerative.set_xlabel("Mean predicted probability (Positive class: 1)")
    ax_random.set_ylabel("Fraction of positives (Positive class: 1)")
    ax_random.set_title("Calibration plots for random split")
    ax_agglomerative.set_title("Calibration plots for agglomerative clustering split")
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
    model_color = get_model_order_and_color()[1]
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 10)

    fig = plt.figure(layout="constrained", figsize=kwargs["figsize"])
    gs = plt.GridSpec(11, 2, figure=fig)

    top_left = fig.add_subplot(gs[0:5, 0])
    top_right = fig.add_subplot(gs[5:10, 0])
    bottom_left = fig.add_subplot(gs[0:5, 1], sharex=top_left)
    bottom_right = fig.add_subplot(gs[5:10, 1], sharex=top_right)
    axs = np.array(
        [
            [top_left, bottom_left],
            [top_right, bottom_right],
        ]
    )
    ax_legend = fig.add_subplot(gs[10:, :])
    models = ["Chemprop", "Neural FP + RF"]
    splits = ["Random", "Agglomerative clustering"]
    for i, split in enumerate(splits):
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
            axs[i, j].set_title(f"{model} - {split}")
            if i == 1:
                axs[i, j].set_xlabel("Predicted probability")
            else:
                axs[i, j].set_xlabel("")
            if j == 0:
                axs[i, j].set_ylabel("Density")
            else:
                axs[i, j].set_ylabel("")
    handles, labels = top_left.get_legend_handles_labels()
    top_left.legend().remove()
    bottom_left.legend().remove()
    top_right.legend().remove()
    bottom_right.legend().remove()
    ax_legend.legend(handles, ["Active", "Inactive"], loc="center", ncol=4)
    remove_ax_frame(ax_legend)
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
