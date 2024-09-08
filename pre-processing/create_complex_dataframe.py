"""
Creates DataFrame from the CSI data in the acquisition folder.

It saves the CSI values as the complex numbers for each subcarrier in the packet
in each different scenario.

The DataFrame is saved as a pickle file in the combined folder.
"""

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# Add the top-level directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import interleaved as decoder

DATA_PATH = "../data/acquisition"
OUTPUT_PATH = "../data/combined"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_packet_df(person: str, file: str) -> pd.DataFrame:
    pcap_path = os.path.join(DATA_PATH, person, file)
    position = file.split("_")[0]
    samples = decoder.read_pcap(pcap_path)
    data = samples.get_pd_csi()

    data["person"] = person
    data["position"] = position

    return data


if __name__ == "__main__":
    start_time = time.time()
    df_list = []
    filename = "complex_csi_dataframe.pkl"
    output_file = os.path.join(OUTPUT_PATH, filename)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(get_packet_df, person, file)
            for person in os.listdir(DATA_PATH)
            for file in os.listdir(os.path.join(DATA_PATH, person))
        ]

        total = len(futures)
        for i, future in enumerate(as_completed(futures)):
            df_list.append(future.result())
            print(f"Progress: {i+1}/{total}", end="\r")
        logging.info(f"{i+1}/{total} All tasks completed")

    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined DataFrame shape: {combined_df.shape}")

    combined_df.to_pickle(output_file)
    logging.info(f"DataFrame saved as {filename}")
    execution_time = time.time() - start_time
    logging.info(f"Execution time: {execution_time:.2f} seconds")
