"""
Create amplitude DataFrame from complex DataFrame.

It loads the complex DataFrame from the combined folder and computes the amplitude
of the CSI values for each subcarrier in the packet in each different scenario.

The DataFrame is saved as a pickle file in the combined folder.
"""

import logging
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd

DATA_PATH = "../data/acquisition/"
OUTPUT_PATH = "../data/combined/"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    start_time = time.time()
    complex_df_filename = "complex_csi_dataframe.pkl"
    output_filename = "amplitude_csi_dataframe.pkl"
    output_file = OUTPUT_PATH + output_filename
    non_complex_columns = ["person", "position"]

    complex_df: pd.DataFrame = pd.read_pickle(OUTPUT_PATH + complex_df_filename)
    non_complex_df = complex_df[non_complex_columns].astype(
        {"person": "uint8", "position": "uint8"}
    )
    complex_columns_df = complex_df.drop(columns=non_complex_columns)

    dask_df: dd.DataFrame = dd.from_pandas(complex_columns_df, npartitions=16)
    amplitude_dask_df: dd.DataFrame = dask_df.map_partitions(
        lambda df: df.apply(np.abs).astype(np.float32)
    )
    amplitude_df = amplitude_dask_df.compute()

    result_df = pd.concat([non_complex_df, amplitude_df], axis=1)
    result_df.to_pickle(output_file)

    execution_time = time.time() - start_time
    logging.info(f"Execution time: {execution_time:.2f} seconds")

    print(f"Complex DataFrame shape: {result_df.shape}")
    print(result_df.iloc[0, 0], result_df.iloc[0, 1], result_df.iloc[0, 2])
    print(
        type(result_df.iloc[0, 0]),
        type(result_df.iloc[0, 1]),
        type(result_df.iloc[0, 2]),
    )
    print(result_df.head())
