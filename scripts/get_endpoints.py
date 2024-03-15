from pathlib import Path

import pandas as pd

data_path = Path(__file__).parents[1] / "data"
tox21_presplit_df = pd.read_csv(
    data_path / "intermediate_data" / "tox21_presplit.tsv", sep="\t"
)
print(tox21_presplit_df["endpoint"].unique())
