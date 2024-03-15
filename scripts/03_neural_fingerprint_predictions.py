"""Run ML experiments on the Tox21 dataset."""

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightning import pytorch as pl
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any import MolToFoldedMorganFingerprint
from molpipeline.pipeline_elements.mol2any.mol2chemprop import MolToChemprop
from molpipeline.pipeline_elements.post_prediction import PostPredictionWrapper
from molpipeline.sklearn_estimators.chemprop.models import ChempropClassifier
from molpipeline.sklearn_estimators.chemprop.neural_fingerprint import ChempropNeuralFP
from molpipeline.sklearn_estimators.similarity_transformation import TanimotoToTraining
from molpipeline.utils.kernel import tanimoto_similarity_sparse
from sklearn.base import BaseEstimator
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


def define_chemprop_pipeline(n_jobs: int) -> Pipeline:
    """Define the Chemprop pipeline.

    Parameters
    ----------
    n_jobs : int
        Number of jobs to use for training.

    Returns
    -------
    Pipeline
        Chemprop pipeline.
    """
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        max_epochs=200,
        enable_model_summary=False,
        callbacks=[],
        enable_progress_bar=False,
        val_check_interval=0.0,
    )
    return Pipeline(
        [
            ("smi2mol", SmilesToMolPipelineElement()),
            ("error_filter", ErrorFilter(filter_everything=True)),
            ("mol2graph", MolToChemprop()),
            (
                "chemprop",
                ChempropClassifier(
                    n_jobs=4,
                    lightning_trainer=trainer,
                    model__message_passing__dropout=0.2,
                ),
            ),
        ],
        n_jobs=1,
        memory=joblib.Memory(),
    )


def define_models(n_jobs) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
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
    knn_model = KNeighborsClassifier(n_neighbors=9, n_jobs=n_jobs)
    knn_hyperparams = {}

    svc_model = SVC(probability=True)
    svc_hyperparams = {
        "C": np.power(5.0, np.arange(-4, 4)),
    }

    rf_model = RandomForestClassifier(
        n_estimators=1024,
        n_jobs=n_jobs,
    )
    rf_hyperparams = {
        "max_depth": [4, 16, None],
    }

    model_dict = {
        "KNN": (knn_model, knn_hyperparams),
        "SVC": (svc_model, svc_hyperparams),
        "RF": (rf_model, rf_hyperparams),
    }
    return model_dict


def compile_pipeline(
    model: BaseEstimator, neural_encoder: ChempropNeuralFP
) -> Pipeline:
    """Compile the model and the neural encoder into a pipeline.

    Parameters
    ----------
    model : BaseEstimator
        The model to use.
    neural_encoder : ChempropNeuralFP
        The neural encoder to use.

    Returns
    -------
    Pipeline
        The compiled pipeline.
    """
    error_filter = ErrorFilter(filter_everything=True)
    error_replacer = ErrorReplacer.from_error_filter(error_filter, np.nan)
    return Pipeline(
        [
            ("smi2mol", SmilesToMolPipelineElement()),
            ("error_filter", error_filter),
            ("mol2graph", MolToChemprop()),
            ("neural_encoder", neural_encoder),
            ("model", model),
            ("error_replacer", PostPredictionWrapper(error_replacer)),
        ],
        n_jobs=1,
        memory=joblib.Memory(),
    )


def main() -> None:
    """Run ML experiments on the Tox21 dataset."""

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

    data_path = Path(__file__).parents[1] / "data"
    presplit_ml_data_df = pd.read_csv(
        data_path / "intermediate_data" / "tox21_presplit.tsv", sep="\t"
    )

    args = parse_args()

    endpoint_df = presplit_ml_data_df.query(f"endpoint == '{args.endpoint}'")

    split_strategy_list = [
        "Random",
        "Agglomerative clustering",
        #  "Murcko scaffold",
        #  "Generic scaffold",
    ]

    model_dict = define_models(args.n_jobs)
    splitter = LeavePGroupsOut(1)
    prediction_df_list = []

    for split_strategy in tqdm(split_strategy_list, desc="Split strategy"):
        iter_splits = splitter.split(
            endpoint_df["smiles"].tolist(),
            groups=endpoint_df[split_strategy].tolist(),
        )
        for trial, (train_idx, test_idx) in tqdm(
            enumerate(iter_splits), desc="Split", leave=False
        ):
            chemprop_model = define_chemprop_pipeline(args.n_jobs)
            chemprop_model.fit(
                endpoint_df.iloc[train_idx]["smiles"].tolist(),
                endpoint_df.label.to_numpy()[train_idx],
            )
            test_df = endpoint_df.iloc[test_idx].copy()
            test_df["proba"] = chemprop_model.predict_proba(
                test_df.smiles.tolist()
            )[:, 1]
            test_df["prediction"] = chemprop_model.predict(test_df.smiles.tolist())
            test_df["endpoint"] = args.endpoint
            test_df["trial"] = trial
            test_df["Split strategy"] = split_strategy
            test_df["model"] = "Chemprop"
            prediction_df_list.append(test_df)

            chemprop_encoder = chemprop_model["chemprop"].to_encoder()
            for model_name, (model, hyperparams) in tqdm(
                model_dict.items(), desc="Model", leave=False
            ):
                model = compile_pipeline(model, chemprop_encoder)
                hparams = {"model__" + k: v for k, v in hyperparams.items()}
                model = GridSearchCV(
                    model,
                    hparams,
                    cv=LeaveOneGroupOut(),
                    scoring="balanced_accuracy",
                    n_jobs=1,
                )
                train_df = endpoint_df.iloc[train_idx]
                test_df = endpoint_df.iloc[test_idx].copy()
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
    prediction_df = pd.concat(prediction_df_list)
    save_path = (
        data_path
        / "intermediate_data"
        / "model_predictions"
        / f"neural_fingerprint_predictions_{args.endpoint}.tsv.gz"
    )
    prediction_df.to_csv(save_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
