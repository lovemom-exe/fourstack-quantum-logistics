# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: Use Root Path, Data Path
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os

# ==========================================================================
# PARAMETERS
# ==========================================================================

# Root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data path
DATA_PATH = os.path.join(ROOT_PATH, "data")

# Benchmark path
BENCHMARK = os.path.join(ROOT_PATH, "evaluation/benchmark_data")

# Training/evaluation_result
TRAINING_EVA_RESULT = os.path.join(ROOT_PATH, "training/evaluation_result")

# Raw data path
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
# Processed data path
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
# feature engineering data path
FEATURE_ENGINEERING_DATA_PATH = os.path.join(DATA_PATH, "features")


# ==========================================================================
# RAW
# ==========================================================================
# Fresh retail data path
FRESH_RETAIL_EVAL_PATH = os.path.join(RAW_DATA_PATH, "fresh_retail_eval_data.csv")
FRESH_RETAIL_TRAIN_PATH = os.path.join(RAW_DATA_PATH, "fresh_retail_train_data.csv")

# Warehouse & Inventory Management Dataset of health commodity
DISTRICT_STORE_PATH = os.path.join(RAW_DATA_PATH, "district_store.csv")
NATIONAL_CENTRAL_MEDICAL_STORE_PATH = os.path.join(
    RAW_DATA_PATH, "national_central_medical_store.csv"
)
REGIONAL_WAREHOUSE_PATH = os.path.join(RAW_DATA_PATH, "regional_warehouse.csv")

#
PERISHABLE_GOODS_DATA = os.path.join(RAW_DATA_PATH, "perishable_goods_management.csv")
# ==========================================================================
# PROCESSED
# ==========================================================================
# Fresh retail data path
FRESH_RETAIL_EVAL_PATH_PROCESS = os.path.join(
    PROCESSED_DATA_PATH, "fresh_retail_eval_data.csv"
)
FRESH_RETAIL_TRAIN_PATH_PROCESS = os.path.join(
    PROCESSED_DATA_PATH, "fresh_retail_train_data.csv"
)

# Warehouse & Inventory Management Dataset of health commodity
DISTRICT_STORE_PATH_PROCESS = os.path.join(PROCESSED_DATA_PATH, "district_store.csv")
NATIONAL_CENTRAL_MEDICAL_STORE_PATH_PROCESS = os.path.join(
    PROCESSED_DATA_PATH, "national_central_medical_store.csv"
)
REGIONAL_WAREHOUSE_PATH_PROCESS = os.path.join(
    PROCESSED_DATA_PATH, "regional_warehouse.csv"
)
# ==========================================================================
# FEAATURES
# ==========================================================================
# Fresh retail data path
FRESH_RETAIL_EVAL_PATH_FEATURE = os.path.join(
    FEATURE_ENGINEERING_DATA_PATH, "fresh_retail_eval_data.csv"
)
FRESH_RETAIL_TRAIN_PATH_FEATURE = os.path.join(
    FEATURE_ENGINEERING_DATA_PATH, "fresh_retail_train_data.csv"
)

# Warehouse & Inventory Management Dataset of health commodity
DISTRICT_STORE_PATH_FEATURE = os.path.join(
    FEATURE_ENGINEERING_DATA_PATH, "district_store.csv"
)
NATIONAL_CENTRAL_MEDICAL_STORE_PATH_FEATURE = os.path.join(
    FEATURE_ENGINEERING_DATA_PATH, "national_central_medical_store.csv"
)
REGIONAL_WAREHOUSE_PATH_FEATURE = os.path.join(
    FEATURE_ENGINEERING_DATA_PATH, "regional_warehouse.csv"
)

# ==========================================================================
# Benchmark Data Path | Optimize result path
# ==========================================================================
BENCHMARK_FOOD = os.path.join(BENCHMARK, "food_model")
VQR_TRAIN_RESULT = os.path.join(TRAINING_EVA_RESULT, "vqr.csv")


# ==========================================================================
# Benchmark Data Path | Optimize result path
# ==========================================================================

# ==========================================================================
# REFRESH ACCESS KEY
# ==========================================================================
ACCESS_TOKEN_PATH = os.path.join(ROOT_PATH, "access_token.csv")

# ==========================================================================
# Test: DOES THE DATAPATH IS EXIST
# ==========================================================================

TEST_PATH = ACCESS_TOKEN_PATH


def main():
    if os.path.exists(TEST_PATH):
        print(TEST_PATH)
    else:
        return


if __name__ == "__main__":
    main()
