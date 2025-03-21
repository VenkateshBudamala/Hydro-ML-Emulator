# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:08:33 2024

@author: Dr. Venkatesh Budamala, IISc Bangalore
"""

# List to store execution time for each step
timing_list = []

## Setting the Working Directory to Script Location ##
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
os.chdir(current_dir)  # Change working directory

# Executing required input and function files
exec(open('Input_File.py').read())  # Execute input file
exec(open('Functions.py').read())  # Execute functions file

# Backup the default calibration range for the physical model
txt_backup(Physical_Model_Loc)

## Step 1: Generating the Initial Design ##
start_time = time.time()
int_samples_lhs = INT_DESG(int_samples, param_ranges)  # Generate initial samples
timing_list.append(time.time() - start_time)  # Record execution time

## Step 2: Generating the Objective Function ##
start_time = time.time()
par_set = Gen_PS(int_samples, int_samples_lhs, par_conf, reach_no, obs_flow, swat_loc)  # Generate parameter set
timing_list.append(time.time() - start_time)

## Step 3: Fit the Emulator Model ##
start_time = time.time()
model, conv_crt_train, conv_crt_test = Fit_Emulator(par_set, target_col)  # Fit emulator model
timing_list.append(time.time() - start_time)

## Step 4: Adaptive Sampling ##
start_time = time.time()
if conv_crt_train < cc_threshold and conv_crt_test < cc_threshold:
    # Perform adaptive sampling
    adpt_samples_lv = lola_voronoi_adaptive_sampling(int_samples_lhs, model, extra_samples, param_ranges)
    
    # Update model with adaptive design
    model, par_set, conv_crt_train, conv_crt_test = ADP_DESG(
        tot_samples_lhs, int_samples, extra_samples, Total_samples, target_col,
        par_set, param_ranges, par_conf, reach_no, obs_flow, swat_loc,
        cc_threshold, conv_crt_test, conv_crt_train
    )

timing_list.append(time.time() - start_time)

## Step 5: Optimization ##
start_time = time.time()
if conv_crt_train > cc_threshold and conv_crt_test > cc_threshold:
    # Perform optimization using emulator
    Optimal_PS, Perf_OPT, Perf_EMU = Emulator_Optimization_GA(
        model, param_ranges, par_conf, reach_no, obs_flow, swat_loc
    )

timing_list.append(time.time() - start_time)

## Step 6: Validation of Optimal Parameter Set in the Physical Model ##
txt_backup(Physical_Model_Loc)  # Backup model settings
Perf1, Perf2, Cali_Pflow, Vali_Pflow = Vali_PM_OPS(
    Optimal_PS, par_conf, reach_no, obs_flow, obs_flow_v, swat_loc
)

## Step 7: Saving Results ##
save_variables(model,par_set,Cali_Pflow,Vali_Pflow,Optimal_PS,timing_list)


