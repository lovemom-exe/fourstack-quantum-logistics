# ==========================================================================
# Author: Hoang Anh Quan
# Version: 0.1.0
# Purpose: Load and manage forecast data from the database
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os

# ==========================================================================
# PARAMETERS
# ==========================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(ROOT_DIR, "fake_test_data/forecast.json")

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
with open(TEST_DATA_PATH, "r") as file:
    forecast_data = json.load(file)
