"""Srcipt to create the final tables for the paper."""  # pylint: disable=invalid-name

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from loguru import logger
from plot_utils import get_performance_metrics, load_data


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


def create_overview_table(base_path: Path) -> None:
    """Create the performance table."""
    prediction_folder = base_path / "data" / "intermediate_data" / "model_predictions"
    config_path = base_path / "config" / "endpoints.yaml"
    # Load endpoints from yaml
    with open(config_path, "r", encoding="UTF-8") as file:
        endpoints = yaml.safe_load(file)["endpoint"]
    performance_df_list = []
    dataset_size_list = []
    for endpoint in endpoints:
        data_df = load_data(endpoint, prediction_folder)
        label_count_df = data_df.pivot_table(
            index="endpoint", values="smiles", columns="label", aggfunc="nunique"
        )
        label_count_df.rename(columns={0: "#Inactives", 1: "#Actives"}, inplace=True)
        dataset_size_list.append(label_count_df)
        performance_df = get_performance_metrics(data_df)
        performance_df["endpoint"] = endpoint
        performance_df_list.append(performance_df)
    final_performance_df = pd.concat(performance_df_list)
    dataset_size_df = pd.concat(dataset_size_list)
    final_performance_df = final_performance_df.query("metric == 'Balanced accuracy'")
    final_performance_df = final_performance_df.query("split == 'Random'")
    show_model_list = ["Chemprop", "Morgan FP + RF", "Neural FP + RF"]
    final_performance_df = final_performance_df.loc[
        final_performance_df["model"].isin(show_model_list)
    ]

    mean_performance_df = final_performance_df.pivot_table(
        index="endpoint", columns="model", values="Performance", aggfunc=agg_mean_std
    )
    mean_performance_df = dataset_size_df.merge(mean_performance_df, on="endpoint")
    mean_performance_df.index = mean_performance_df.index.str.replace("_", r"\_")
    logger.info("Creating performance table for overview")
    logger.info(f"\n{mean_performance_df.to_latex()}")


def create_performance_table(base_path: Path) -> None:
    """Create the performance table."""
    prediction_folder = base_path / "data" / "intermediate_data" / "model_predictions"
    config_path = base_path / "config" / "endpoints.yaml"
    # Load endpoints from yaml
    with open(config_path, "r", encoding="UTF-8") as file:
        endpoints = yaml.safe_load(file)["endpoint"]
    performance_df_list = []
    for endpoint in endpoints:
        data_df = load_data(endpoint, prediction_folder)
        performance_df = get_performance_metrics(data_df)
        performance_df["endpoint"] = endpoint.replace("_", r"\_")
        performance_df_list.append(performance_df)
    final_performance_df = pd.concat(performance_df_list)
    final_performance_df = final_performance_df.query("split == 'Random'")
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


def main() -> None:
    """Main function."""
    base_path = Path(__file__).parents[1]
    logger.add(base_path / "logs/07_create_final_tables.log")
    create_overview_table(base_path)
    create_performance_table(base_path)


if __name__ == "__main__":
    main()
