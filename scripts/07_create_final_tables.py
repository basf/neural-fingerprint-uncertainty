"""Srcipt to create the final tables for the paper."""  # pylint: disable=invalid-name

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from loguru import logger
from plot_utils import get_performance_metrics, load_all_performances, load_data, get_model_order_and_color


def agg_mean_std(values: npt.NDArray[np.float_]) -> str:
    """Aggregate a numpy array to a string with mean and std.

    Parameters
    ----------
    values : npt.NDArray[np.float_]
        The values.

    Returns
    -------
    str
        The aggregated string.
    """
    x_mean = np.mean(values)
    x_std = np.std(values)
    return f"{x_mean:.2f} ± {x_std:.2f}"


def create_dataset_size_table(base_path: Path) -> None:
    """Create the performance table."""
    prediction_folder = base_path / "data" / "intermediate_data" / "model_predictions"
    config_path = base_path / "config" / "endpoints.yaml"
    # Load endpoints from yaml
    with open(config_path, "r", encoding="UTF-8") as file:
        endpoints = yaml.safe_load(file)["endpoint"]
    dataset_size_list = []
    for endpoint in endpoints:
        data_df = load_data(endpoint, prediction_folder)
        label_count_df = data_df.pivot_table(
            index="endpoint", values="smiles", columns="label", aggfunc="nunique"
        )
        label_count_df.rename(columns={0: "#Inactives", 1: "#Actives"}, inplace=True)
        dataset_size_list.append(label_count_df)
    dataset_size_df = pd.concat(dataset_size_list)
    dataset_size_df.index = dataset_size_df.index.str.replace("_", r"\_")
    logger.info("Creating performance table with dataset sizes.")
    logger.info(f"\n{dataset_size_df.to_latex()}")


def create_performance_table(base_path: Path) -> None:
    """Create the performance table."""
    all_data_df = load_all_performances(base_path)
    all_data_df["endpoint"] = all_data_df["endpoint"].str.replace("_", r"\_")
    final_performance_df = all_data_df.query("split == 'Random'")
    final_performance_df = final_performance_df.loc[
        final_performance_df["metric"].isin(["Balanced accuracy", "Brier score"])
    ]
    final_performance_df = final_performance_df.loc[
        final_performance_df["model"].isin(
            ["Chemprop", "Morgan FP + RF", "Neural FP + RF"]
        )
    ]
    agg_performance_df = final_performance_df.pivot_table(
        index=["endpoint", "model"],
        columns=["metric"],
        values="Performance",
        aggfunc=agg_mean_std,
    )
    agg_performance_df.reset_index(inplace=True)
    logger.info("Creating performance table for overview")
    logger.info(f"\n{agg_performance_df.to_latex(index=False)}")


def create_table_rf_calibrated_rf(base_path: Path) -> None:
    """Create the performance table for the RF and Calibrated RF models."""
    all_data_df = load_all_performances(base_path)
    all_data_df["endpoint"] = all_data_df["endpoint"].str.replace("_", r"\_")
    all_data_df["encoding"] = "Neural FP"
    all_data_df.loc[all_data_df["model"].str.contains("Morgan FP"), "encoding"] = (
        "Morgan FP"
    )
    all_data_df["base_model"] = all_data_df["model"]
    all_data_df["base_model"] = all_data_df["base_model"].str.replace(
        "Morgan FP + ", ""
    )
    all_data_df["base_model"] = all_data_df["base_model"].str.replace(
        "Neural FP + ", ""
    )
    all_data_df = all_data_df.loc[
        all_data_df["metric"].isin(["Balanced accuracy", "Brier score", "Log loss"])
    ]
    all_data_df.loc[all_data_df["metric"] == "Balanced accuracy", "metric"] = "BA"
    all_data_df = all_data_df.loc[
        all_data_df["base_model"].isin(["RF", "Calibrated RF"])
    ]
    all_data_df = all_data_df.query("encoding == 'Neural FP'")
    for split in ["Random", "Agglomerative clustering"]:
        split_df = all_data_df.loc[all_data_df["split"] == split]
        agg_performance_df = split_df.pivot_table(
            index=["endpoint", "metric"],
            columns=["encoding", "base_model"],
            values="Performance",
            aggfunc=agg_mean_std,
        )
        agg_performance_df.reset_index(inplace=True)
        logger.info(f"Creating performance table for {split}")
        logger.info(f"\n{agg_performance_df.to_latex(index=False)}")


def create_table_precision_recall(base_path: Path) -> None:
    """Create the performance table for the selectivity and recall."""
    all_data_df = load_all_performances(base_path)
    all_data_df["endpoint"] = all_data_df["endpoint"].str.replace("_", r"\_")
    all_data_df.loc[all_data_df["model"] == "Morgan FP + Calibrated RF", "model"] = (
        "Morgan FP + Cal. RF"
    )
    all_data_df.loc[all_data_df["model"] == "Neural FP + Calibrated RF", "model"] = (
        "Neural FP + Cal. RF"
    )
    all_data_df = all_data_df.loc[all_data_df["metric"].isin(["Precision", "Recall"])]
    agg_performance_df = all_data_df.pivot_table(
        index=["endpoint", "model"],
        columns=["split", "metric"],
        values="Performance",
        aggfunc=agg_mean_std,
    )
    agg_performance_df.reset_index(inplace=True)
    logger.info("Creating table for selectivity and recall")
    logger.info(f"\n{agg_performance_df.to_latex(index=False)}")
    agg_performance_df2 = all_data_df.pivot_table(
        index="model",
        columns=["split", "metric"],
        values="Performance",
        aggfunc=agg_mean_std,
    )
    model_order = get_model_order_and_color()[0]
    agg_performance_df2 = agg_performance_df2.loc[model_order]
    agg_performance_df2.reset_index(inplace=True)
    logger.info("Creating table for selectivity and recall")
    logger.info(f"\n{agg_performance_df2.to_latex(index=False)}")


def main() -> None:
    """Main function."""
    base_path = Path(__file__).parents[1]
    logger.add(base_path / "logs/07_create_final_tables.log")
    create_dataset_size_table(base_path)
    create_performance_table(base_path)
    create_table_rf_calibrated_rf(base_path)
    create_table_precision_recall(base_path)


if __name__ == "__main__":
    main()
