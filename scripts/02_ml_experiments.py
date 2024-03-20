"""Run ML experiments on the Tox21 dataset."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any import MolToFoldedMorganFingerprint
from molpipeline.sklearn_estimators.similarity_transformation import TanimotoToTraining
from molpipeline.utils.kernel import tanimoto_similarity_sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeavePGroupsOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of jobs to use for training.",
    )
    argument_parser.add_argument(
        "--endpoint",
        type=str,
        help="Endpoint to train on.",
    )
    args = argument_parser.parse_args()
    return args


def define_models(n_jobs: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Define the models to train.

    Parameters
    ----------
    n_jobs : int
        Number of jobs to use for training.

    Returns
    -------
    dict[str, tuple[Pipeline, dict[str, list[Any]]]]
        Dictionary of model names and tuples of the model pipeline and the
        hyperparameter grid.
    """
    knn_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMolPipelineElement()),
            ("mol2morgan", MolToFoldedMorganFingerprint(output_datatype="sparse")),
            ("precomputed_kernel", TanimotoToTraining()),
            (
                "k_nearest_neighbors",
                KNeighborsClassifier(metric="precomputed", n_jobs=n_jobs),
            ),
        ],
        n_jobs=n_jobs,
        memory=joblib.Memory(),
    )
    knn_hyperparams = {
        "k_nearest_neighbors__n_neighbors": [9],
    }

    svc_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMolPipelineElement()),
            ("mol2morgan", MolToFoldedMorganFingerprint(output_datatype="sparse")),
            ("svc", SVC(kernel=tanimoto_similarity_sparse, probability=True)),
        ],
        n_jobs=n_jobs,
        memory=joblib.Memory(),
    )
    svc_hyperparams = {
        "svc__C": np.power(5.0, np.arange(-4, 4)).tolist(),
        "svc__class_weight": [None, "balanced"],
    }

    random_forest_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMolPipelineElement()),
            ("mol2morgan", MolToFoldedMorganFingerprint(output_datatype="sparse")),
            (
                "balanced_random_forest",
                RandomForestClassifier(
                    n_estimators=1024,
                    bootstrap=True,
                    n_jobs=n_jobs,
                ),
            ),
        ],
        n_jobs=n_jobs,
        memory=joblib.Memory(),
    )
    rf_hyperparams = {
        "balanced_random_forest__max_depth": [4, 16, None],
    }

    model_dict = {
        "KNN": (knn_pipeline, knn_hyperparams),
        "SVC": (svc_pipeline, svc_hyperparams),
        "RF": (random_forest_pipeline, rf_hyperparams),
    }
    return model_dict


def main() -> None:
    """Run ML experiments on the Tox21 dataset."""
    data_path = Path(__file__).parents[1] / "data"
    tox21_presplit_df = pd.read_csv(
        data_path / "intermediate_data" / "tox21_presplit.tsv", sep="\t"
    )

    args = parse_args()

    endpoint_df = tox21_presplit_df.query(f"endpoint == '{args.endpoint}'")
    prediction_path = data_path / "intermediate_data" / "model_predictions"
    prediction_path.mkdir(parents=True, exist_ok=True)
    save_path = prediction_path / f"test_set_predictions_{args.endpoint}.tsv.gz"

    split_strategy_list = [
        "Random",
        "Agglomerative clustering",
        #  "Murcko scaffold",
        #  "Generic scaffold",
    ]

    model_dict = define_models(args.n_jobs)
    splitter = LeavePGroupsOut(1)
    prediction_df_list = []

    n_splits: int = 0
    for split_strategy in split_strategy_list:
        n_splits = n_splits + splitter.get_n_splits(
            groups=endpoint_df[split_strategy].tolist()
        )
    pbar = tqdm(total=n_splits * len(model_dict), leave=False)

    for split_strategy in split_strategy_list:
        for trial, (train_idx, test_idx) in enumerate(
            splitter.split(
                endpoint_df.smiles.tolist(),
                groups=endpoint_df[split_strategy].tolist(),
            )
        ):
            for model_name, (model, hyperparams) in model_dict.items():
                model = GridSearchCV(
                    model,
                    hyperparams,
                    cv=LeaveOneGroupOut(),
                    scoring="balanced_accuracy",
                )
                train_df = endpoint_df.iloc[train_idx]
                test_df = endpoint_df.iloc[test_idx].copy()
                # TODO: Balance training data
                model.fit(
                    train_df.smiles.tolist(),
                    train_df.label.tolist(),
                    groups=train_df[split_strategy].tolist(),
                )
                test_df["proba"] = model.predict_proba(test_df.smiles.tolist())[:, 1]
                test_df["prediction"] = model.predict(test_df.smiles.tolist())
                test_df["endpoint"] = args.endpoint
                test_df["trial"] = trial
                test_df["Split strategy"] = split_strategy
                test_df["model"] = model_name
                prediction_df_list.append(test_df)
                pbar.update(1)
    pbar.close()
    prediction_df = pd.concat(prediction_df_list)
    prediction_df.to_csv(save_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
