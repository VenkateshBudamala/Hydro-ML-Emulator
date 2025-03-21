# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:29:44 2024

@author: Dr. Venkatesh Budamala, IISc Bangalore

Description:
This script includes basic inputs like parameter configuration, threshold setup, 
and handling of observed flow data.
"""

##### Inputs to the Framework #####

# Physical Model Selection
model_selection = {
    1: "SWAT Model",
    2: "RAVEN Model",
    3: "SUMMA Model",
    4: "HEC-RAS Model"
}

selected_physical_model = 1  # Change this to 2, 3, or 4 for other models


# Path to the physical model directory
physical_model_loc = ''

## Parameters (for SWAT model; can be edited for other models) ##
# Parameter ranges for calibration 
param_ranges = [
    (35, 98),         # CN2: SCS curve number
    (0, 1),           # ALPHA_BF: Baseflow alpha factor
    (0, 500),         # GW_DELAY: Groundwater delay
    (0, 5000),        # GWQMN: Threshold depth of water in the shallow aquifer
    (0, 1),           # ESCO: Soil evaporation compensation factor
    (0, 1),           # SOL_AWC: Available water capacity of the soil layer
    (0.02, 0.2),      # GW_REVAP: Groundwater revap coefficient
    (0, 1),           # RCHRG_DP: Deep aquifer percolation fraction
    (0, 500)          # REVAPMN: Minimum depth for revap
]

# Corresponding parameter configuration files
param_config = [
    'v__CN2.mgt',
    'v__ALPHA_BF.gw',
    'v__GW_DELAY.gw',
    'v__GWQMN.gw',
    'v__ESCO.bsn',
    'v__SOL_AWC().sol',
    'v__GW_REVAP.gw',
    'v__RCHRG_DP.gw',
    'v__REVAPMN.gw'
]

# Reach number for observed flow extraction
reach_no = 1

##### Thresholds and Sampling #####

# Sampling settings
total_samples = 100 * len(param_ranges)
initial_samples = 20 * len(param_ranges)
extra_samples = 10 * len(param_ranges)

# Correlation coefficient threshold
cc_threshold = 0.97

# Target column for observed data (zero-based indexing)
target_col = -1

##### Observed Data #####

# Load calibration observed flow data
obs_file = r'D:\Obs_Basanthpur.xlsx'
obs = pd.read_excel(obs_file)  # Assuming default sheet
obs_flow = np.array(obs.iloc[:, reach_no])

# Load validation observed flow data from a different sheet
obs_validation_sheet = 'Sheet2'
obs_validation = pd.read_excel(obs_file, sheet_name=obs_validation_sheet)
obs_flow_v = np.array(obs_validation.iloc[:, reach_no])


