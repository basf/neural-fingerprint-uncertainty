"""Assign groups to the Tox21 dataset for cross-validation."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any import (
    MolToFoldedMorganFingerprint,
    MolToSmilesPipelineElement,
)
from molpipeline.pipeline_elements.mol2mol import (
    MakeScaffoldGenericPipelineElement,
    MurckoScaffoldPipelineElement,
)
from molpipeline.utils.kernel import self_tanimoto_distance
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from tqdm.auto import tqdm

N_GROUPS = 5
clustering_pipeline = Pipeline(
    [
        ("smi2mol", SmilesToMolPipelineElement()),
        ("mol2morgan", MolToFoldedMorganFingerprint(sparse_output=False)),
        (
            "agg_clustering",
            AgglomerativeClustering(
                distance_threshold=0.8,
                linkage="average",
                metric=self_tanimoto_distance,
                n_clusters=None,
            ),
        ),
    ],
    n_jobs=16,
)

murcko_scaffold = MurckoScaffoldPipelineElement()
none_filter = ErrorFilter.from_element_list([murcko_scaffold])
none_filler = ErrorReplacer.from_error_filter(none_filter, "")
scaffold_pipeline = Pipeline(
    [
        ("smi2mol", SmilesToMolPipelineElement()),
        ("murcko_scaffold", murcko_scaffold),
        ("mol2smi", MolToSmilesPipelineElement()),
        ("none_filter", none_filter),
        ("none_filler", none_filler),
        ("reshape2d", FunctionTransformer(func=np.atleast_2d)),
        ("transpose", FunctionTransformer(func=np.transpose)),
        ("scaffold_encoder", OrdinalEncoder()),
        ("reshape1d", FunctionTransformer(func=np.ravel)),
    ],
    n_jobs=16,
)

murcko_scaffold2 = MurckoScaffoldPipelineElement()
none_filter2 = ErrorFilter.from_element_list([murcko_scaffold2])
none_filler2 = ErrorReplacer.from_error_filter(none_filter2, "")
generic_scaffold_pipeline = Pipeline(
    [
        ("smi2mol", SmilesToMolPipelineElement()),
        ("murcko_scaffold", murcko_scaffold2),
        ("generic_scaffold", MakeScaffoldGenericPipelineElement()),
        ("mol2smi", MolToSmilesPipelineElement()),
        ("none_filter", none_filter2),
        ("none_filler", none_filler2),
        ("reshape2d", FunctionTransformer(func=np.atleast_2d)),
        ("transpose", FunctionTransformer(func=np.transpose)),
        ("scaffold_encoder", OrdinalEncoder()),
        ("reshape1d", FunctionTransformer(func=np.ravel)),
    ],
    n_jobs=16,
)
cluster_dict = {
    "Agglomerative clustering": clustering_pipeline,
    "Murcko scaffold": scaffold_pipeline,
    "Generic scaffold": generic_scaffold_pipeline,
}

data_path = Path(__file__).parents[1] / "data"
tox21_df = pd.read_csv(data_path / "imported_data" / "ml_ready_data.tsv", sep="\t")
endpoint_df_list = []
skf = StratifiedKFold(N_GROUPS, shuffle=True)
sgkf = StratifiedGroupKFold(N_GROUPS)
for endpoint, endpoint_df in tqdm(tox21_df.groupby("endpoint")):
    group_array = -np.ones_like(endpoint_df.label.to_numpy())
    for split, (train_idx, test_idx) in enumerate(
        skf.split(endpoint_df.smiles.tolist(), endpoint_df.label.to_numpy())
    ):
        group_array[test_idx] = split
    endpoint_df["Random"] = group_array
    for grouping_name, grouping_pipeline in cluster_dict.items():
        grouping_pipeline_copy = clone(grouping_pipeline)
        if hasattr(grouping_pipeline_copy, "fit_predict"):
            cluster_label = grouping_pipeline_copy.fit_predict(
                endpoint_df.smiles.tolist(), endpoint_df.label.to_numpy()
            )
        else:
            cluster_label = grouping_pipeline_copy.fit_transform(
                endpoint_df.smiles.tolist(), endpoint_df.label.to_numpy()
            )
        group_array = -np.ones_like(endpoint_df.label.to_numpy())
        for split, (train_idx, test_idx) in enumerate(
            sgkf.split(
                endpoint_df.smiles.tolist(),
                endpoint_df.label.to_numpy(),
                groups=cluster_label,
            )
        ):
            group_array[test_idx] = split
        endpoint_df[grouping_name] = group_array
    # Aggolomerative clustering with group K fold
    endpoint_df_list.append(endpoint_df)
tox21_presplit = pd.concat(endpoint_df_list)


if not os.path.isdir(data_path / "intermediate_data"):
    os.mkdir(data_path / "intermediate_data")
tox21_presplit.to_csv(
    data_path / "intermediate_data" / "tox21_presplit.tsv", sep="\t", index=False
)