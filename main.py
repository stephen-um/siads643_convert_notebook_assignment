import pandas as pd
import numpy as np
import yaml
import os
from load_data import load_data_from_csv, load_data_from_db
from clean_transform_data import rename_columns, clean_transform_data

def main():
    df = load_data_from_csv(config['input_data_path'])
    df = rename_columns(df)
    clean_df = clean_transform_data(df)
    pd.DataFrame(clean_df).to_csv(config['output_data_path'])
    return pd.DataFrame(clean_df)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)

    main()