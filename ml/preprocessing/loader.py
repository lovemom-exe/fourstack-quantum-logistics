# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: This module provides utility functions for handling file paths and directories
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os

from datasets import DatasetDict, load_dataset

import utils.path as util_path

# ==========================================================================
# PARAMETERS
# ==========================================================================
# Data Path
save_data_path = util_path.RAW_DATA_PATH

# Load Dataset: FreshRetailNet-50K
ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")

# Load Dataset: Warehouse & Inventory Management Dataset of health commodity
# ds_1 = load_dataset(
#     "electricsheepafrica/warehouse-inventory-management", "regional_warehouse"
# )
# ds_2 = load_dataset(
#     "electricsheepafrica/warehouse-inventory-management", "district_store"
# )
# ds_3 = load_dataset(
#     "electricsheepafrica/warehouse-inventory-management",
#     "national_central_medical_store",
# )


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def huggingface_dataset(ds: DatasetDict, name: str):
    # Check if the save_data_path exists
    if not os.path.exists(save_data_path):
        print(f"directory: {save_data_path} Is Not Exist")
        return
    if ds is None:
        return

    # Save each split of the dataset to a CSV file
    for split_name in ds.keys():
        output_filename = f"{name}_{split_name}_data.csv"
        output_file_path = os.path.join(save_data_path, output_filename)
        ds[split_name].to_csv(output_file_path, index=False)
        print(f"Saved {split_name} split to {output_filename}")


def main():
    huggingface_dataset(ds, "fresh_retail")


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
if __name__ == "__main__":
    main()
