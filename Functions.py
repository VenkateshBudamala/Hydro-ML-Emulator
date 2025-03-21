# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:26:12 2024

@author: User
"""
# Clearing IPython environment
from IPython import get_ipython
get_ipython().magic('clear')  # Clears the console
get_ipython().magic('reset -f')  # Resets all variables

##### Importing Required Packages #####
import time  # For measuring execution time
import numpy as np  # Numerical computations
import pandas as pd  # Data handling
import os  # File and directory operations
import shutil  # File handling operations
import matplotlib.pyplot as plt  # Plotting library
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb
import hydroeval as he
from geneticalgorithm import geneticalgorithm as ga
import win32com.client
import numpy as np
import pandas as pd
import time
from pydsstools.heclib.dss import HecDss
from hydroeval import kge
import xarray as xr
import pysumma as ps
import time
import hydroeval as he
import calib_trialParams_UM
import pickle

###################### For Initial and Total Design ######################
def INT_DESG(num_i_samples,param_ranges):   
    print('******************* Generation of Initial Design ******************* ')

    # Generate Latin Hypercube Sampling
    int_samples = np.zeros((num_i_samples, len(param_ranges)))
    
    from pyDOE import lhs
    for i, (min_val, max_val) in enumerate(param_ranges):
        arr1 = lhs(1, samples=num_i_samples, criterion='maximin') * (max_val - min_val) + min_val
        int_samples[:, i] = arr1.reshape(num_i_samples, )
        
            
    return int_samples

##################### Physical model simulators ###############################
## For SWAT Simulator ##
def SWAT_SIMULATOR(par_set,par_conf,reach_no,obs_flow,swat_loc):   
    os.chdir(swat_loc)   
    with open('model.in', 'w') as file:
        
        for paras in range(0,len(par_set)):
            file.write(par_conf[paras]+'    '+ str(par_set[paras]) + '\n')
        
    
    import subprocess
    subprocess.call('Swat_edit.exe')
    subprocess.call("swat.exe")
    
    output_rch = np.genfromtxt('output.rch', skip_header=9)

    result = output_rch[:,6]
    pred_flow = np.array(result)
        
    from sklearn.metrics import mean_squared_error as mse
    of = np.sqrt(mse(obs_flow, pred_flow))
    import hydroeval as he
    of1 = he.evaluator(he.kge, pred_flow, obs_flow)
    of2 = of1[0]
    
    return of, of2, pred_flow

## For RAVEN Simulator ##
def RAVEN_SIMULATOR(par_set,par_conf,reach_no,obs_flow,Raven_loc):

    Parameters = dict(zip(par_conf, par_set))
    
    working_directory = Raven_loc
    folder_to_delete = os.path.join(working_directory, "output")

    try:
        if os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)
            #print(f"{folder_to_delete} has been deleted.")
        #else:
            #print(f"{folder_to_delete} does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting {folder_to_delete}: {e}")

    tpl_file_path = Model+'.rvp.tpl'
    rvp_file_path = Model + '.rvp'

    with open(tpl_file_path, 'r') as tpl_file:
        tpl_content = tpl_file.read()

        for old_value, new_value in Parameters.items():
            tpl_content = tpl_content.replace(old_value, str(new_value))

    with open(rvp_file_path, 'w') as rvp_file:
        rvp_file.write(tpl_content)

    #print(f'File "{rvp_file_path}" has been created from the modified .tpl file.')
    cpp_executable = os.path.join(working_directory, "Raven.exe")
    input_file = os.path.join(working_directory, Model)
    output_directory = os.path.join(working_directory, "output")

    os.makedirs(output_directory, exist_ok=True)

    command = [cpp_executable, input_file, '-o', output_directory]
    
    import subprocess
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        #print(f"C++ executable executed successfully for {input_file}. Output saved to {output_directory}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing C++ executable for {input_file}: {e}")

    performance_path = os.path.join(output_directory, "Diagnostics.csv")
    diagnostics = pd.read_csv(performance_path)
    #print(diagnostics)
    KGE_calib = diagnostics['DIAG_KLING_GUPTA'].iloc[1]  
    NSE_calib = diagnostics['DIAG_NASH_SUTCLIFFE'].iloc[1] 
    RMSE_calib = diagnostics['DIAG_RMSE'].iloc[1]
    
    KGE_valid = diagnostics['DIAG_KLING_GUPTA'].iloc[0]  
    NSE_valid = diagnostics['DIAG_NASH_SUTCLIFFE'].iloc[0] 
    RMSE_valid = diagnostics['DIAG_RMSE'].iloc[0]

    return KGE_calib, NSE_calib, RMSE_calib, KGE_valid, NSE_valid, RMSE_valid

## For SUMMA Model
def SUMMA_SIMULATOR(param_calib_gru, param_calib_hru,samp,sdir,summadir, sim_StartTime, sim_EndTime, run_suffix,obs):
 

  calib_trialParams_UM.filepreparation(sdir,param_calib_gru,param_calib_hru,samp)
  # Lumped
  executable    = summadir + 'summa.exe'
  file_manager  = sdir + 'fileManager.txt'
  s             = ps.Simulation(executable,file_manager)
  s.manager['simStartTime'] = sim_StartTime
  s.manager['simEndTime']   = sim_EndTime

  #print(s.manager)
  #print(s.decisions)

  time_start  = time.time()
  s.run('local', run_suffix = run_suffix)
  time_end    = time.time()
  #print(s.stdout)
  print('Status:',s.status)
  t_s         = round(time_end - time_start)
  print('SUMMA run took ' + str(round(time_end - time_start)) + ' sec. for lumped case')
  out = xr.open_dataset(sdir + 'output_'+ run_suffix + '_day.nc')
  sim = out['averageRoutedRunoff']
  print(sim)
  print(obs)
  metric_kge  = he.evaluator(he.kge,sim, obs)
  metric_rmse = he.evaluator(he.rmse, sim, obs)
  kge  = metric_kge[0,0]
  rmse = metric_rmse[0]
  print('KGE:',kge)
  print('RMSE:',rmse)
  return(kge,rmse)


## For HEC-RAS Simulator ##
def HEC_RAS_SIMULATOR(hec_loc, RIVER_ID, REACH_ID, samples, dss_file_path, dss_path_name, obs, start_date, end_date):
    hec = win32com.client.Dispatch("RAS641.HECRASController")
    hec.ShowRas()  # Show HEC-RAS window

    # Open HEC-RAS project
    RASProject = hec_loc
    hec.Project_Open(RASProject)  # Open HEC-RAS project

    RivName = "1"
    RchName = "1"

    # Reading project nodes: cross-sections, bridges, culverts, etc.
    NNod, TabRS, TabNTyp = None, None, None
    v1, v2, NNod, TabRS, TabNTyp = hec.Geometry_GetNodes(RIVER_ID, REACH_ID, NNod, TabRS, TabNTyp)

    reshaped_params = samples.reshape(3, NNod)
    
    start_time = time.time()
    
    for i in range(NNod):
        nLOB, nCh, nROB = reshaped_params[:,i]
        ErrMsg = None
        v0, v1, v2, v3, v4, v5, v0, ErrMsg = hec.Geometry_SetMann_LChR(RIVER_ID, REACH_ID, TabRS[i], float(nLOB), float(nCh), float(nROB), ErrMsg)
    
    NMsg, ListMsg, block = None, None, True
    v1, NMsg, ListMsg, v2 = hec.Compute_CurrentPlan(NMsg, ListMsg, block)
    hec.Project_Save()
    hec.Compute_HideComputationWindow()

    # Read flow data from DSS file for the current cross-section
    pathname = dss_path_name
    with HecDss.Open(dss_file_path) as fid:
        ts = fid.read_ts(pathname, window=(start_date, end_date), trim_missing=True)
        stage_data = pd.DataFrame({"Date": ts.pytimes, "Stage Elevation (m)": ts.values})

    stage_data['Date'] = pd.to_datetime(stage_data['Date'])
    stage_data.set_index('Date', inplace=True)
    stage_data["Stage Elevation (m)"]
    # Resample both observed and simulated data to daily frequency to ensure consistency
    observed_stage_data_daily = obs.resample('D').mean().interpolate()
    simulated_stage_data_daily = stage_data.resample('D').mean().interpolate()

    # Align observed and simulated data to the same time period
    observed_stage_data_daily, simulated_stage_data_daily = observed_stage_data_daily.align(simulated_stage_data_daily, join='inner')

    # Calculate KGE for the current cross-section
    kge_value = kge(simulated_stage_data_daily["Stage Elevation (m)"].values, observed_stage_data_daily.values)

    hec.QuitRas()  # Close HEC-RAS project
    del hec
    
    kge_value1 = kge_value[0]
    print("KGE =", kge_value1)
    
    End_time =time.time()
    
    print('Total time taken for one model run in seconds:', (End_time -start_time))
    
    return kge_value1


###################### Define the KGE function ######################
def kge_score(y_true, y_pred):
    """Calculate the Kling-Gupta Efficiency (KGE)"""
    import hydroeval as he
    of1 = he.evaluator(he.kge, y_pred, y_true)
    of2 = of1[0]
    
    return of2

################ Generating the Objective Function for 'n' sets ###############
def Gen_PS(no_samples,des_samples,par_conf,reach_no,obs_flow,Physical_Model_loc):
    obf = np.array([])
    for iter1 in range(no_samples):
        # print('Parameter Set:',iter1+1)
        samp = des_samples[iter1,:]
        
        if selected_physical_model == 1:
             obf_rmse,obf_kge,pf = SWAT_SIMULATOR(samp,par_conf,reach_no,obs_flow,Physical_Model_loc)
        
        elif selected_physical_model == 2:
            obf_kge = RAVEN_SIMULATOR(par_set,par_conf,reach_no,obs_flow,Physical_Model_loc)
        
        elif selected_physical_model == 3:
             obf_kge, = SUMMA_SIMULATOR(param_calib_gru, param_calib_hru,samp,sdir,summadir, sim_StartTime, sim_EndTime, run_suffix,obs)
        
        elif selected_physical_model == 4:
             obf_kge = HEC_RAS_SIMULATOR(hec_loc, RIVER_ID, REACH_ID, samples, dss_file_path, dss_path_name, obs, start_date, end_date)
        
        obf_kge = obf_kge.reshape(1,)
        obf = np.hstack((obf,obf_kge))
    
    obf = obf.reshape(len(obf),1)
    
    par_set = np.column_stack((des_samples,obf))
    
    return par_set

####################### Fit the adaptive emulator model #########################
def Fit_Emulator(par_set, target_col):
    # Extract features and target
    X, y = par_set[:, 0:target_col], par_set[:, target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 200, 300],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.5, 1]
    }

    # Initialize the model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0)

    # Set up GridSearchCV
    kge_scorer = make_scorer(kge_score, greater_is_better=True)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=kge_scorer,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=0
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    # Train the model with the best parameters
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        **best_params
    )
    best_model.fit(X_train, y_train)

    # Evaluate the model
    kge_scorer = make_scorer(kge_score, greater_is_better=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring=kge_scorer)
    conv_crt_train = cv_score.mean()
    
    print('CV_SCORE:', conv_crt_train)
    
    # Check and handle NaNs in predictions
    test_predictions = best_model.predict(X_test)
 
    of1 = he.evaluator(he.kge, test_predictions, y_test)
    conv_crt_test = of1[0]
    
    print("Convergence Criteria: %f" % (conv_crt_test))
    
    return best_model, conv_crt_train, conv_crt_test

###########################  For Adaptive Design #############################
def ADP_DESG(int_samples,extra_samples,Total_samples,target_col,par_set,param_ranges,par_conf,reach_no,obs_flow,swat_loc,cc_threshold,conv_crt_test,conv_crt_train):
    extra_samples_index = 0
    loop1 = 0
    sample = int_samples
    tot_samp = TOT_DESG(Total_samples, param_ranges)
    while (conv_crt_train < cc_threshold or conv_crt_test < cc_threshold):
        loop1 = loop1+1
        sample = sample + extra_samples 
        if sample > Total_samples:
            print("-------------------- Error: Exceeded Total Number of Samples -------------------- ")
            break
    
        print("****************** Generating the Adaptive Sample Size: ",sample,"*******************" )
        
        
        extra_samples_index = extra_samples_index + extra_samples
        par_set1 = par_set[:,0:target_col]
        
        from sobol_seq import i4_sobol_generate
        sobol_set_total = i4_sobol_generate(len(param_ranges), Total_samples - int_samples)
        sobol_set = sobol_set_total[extra_samples_index-extra_samples:extra_samples_index,:]
        
        extra_set = np.zeros_like(sobol_set)
        for i in range(len(param_ranges)):
            extra_set[:, i] = sobol_set[:, i] * (param_ranges[i][1] - param_ranges[i][0]) + param_ranges[i][0]
        
        ## Generation of Objective Function for Adaptive Sampling ##
        adapt_set = Gen_PS(extra_samples,extra_set,par_conf,reach_no,obs_flow,swat_loc)
        par_set = np.vstack((par_set, adapt_set))
               
        ## Fit the emulator model ##
        model,conv_crt_train,conv_crt_test = Fit_Emulator(par_set,-1)
        
    
    return model,par_set,conv_crt_train,conv_crt_test



####################### Emulator based Optimization ###########################
def Emulator_Optimization_GA(model,param_ranges,par_conf, reach_no, obs_flow,swat_loc):
    from geneticalgorithm import geneticalgorithm as ga
    
    def SWAT_Emulator(X1):
        return -1*model.predict([X1])
    
    # Initialize lists to store iteration data
    iterations = []
    objective_functions = []

    def callback(algorithm):
        # This callback function is called at the end of each iteration
        iterations.append(algorithm.current_iteration)
        objective_functions.append(algorithm.best_function)

    
    algorithm_param = {'max_num_iteration': 10000,\
                       'population_size':100,\
                       'mutation_probability':0.1,\
                       'elit_ratio': 0.01,\
                       'crossover_probability': 0.5,\
                       'parents_portion': 0.25,\
                       'crossover_type':'uniform',\
                       'max_iteration_without_improv':None}
    
    GA_model=ga(function=SWAT_Emulator,\
                dimension=len(param_ranges),\
                variable_type='real',\
                variable_boundaries=np.array(param_ranges),\
                algorithm_parameters=algorithm_param)
    
    GA_model.run()
    
    Optimal_parameter_set = GA_model.best_variable
    Perf_Optimization = -1*GA_model.best_function
    Perf_Emulator = -1*SWAT_Emulator(Optimal_parameter_set)

    print("Optimal Parameter Set:",Optimal_parameter_set)
    print("Optimized objective function through GA: %f"%(Perf_Optimization))
    print("Validation of optimized hydrological model through Emulator: %f"%(Perf_Emulator))
    
    # Save iteration data to CSV
    iteration_df = pd.DataFrame({
        'Iteration': iterations,
        'Objective_Function': objective_functions
    })
    

    return Optimal_parameter_set, Perf_Optimization, Perf_Emulator, GA_model, iteration_df



################# Validation of Physical model after optimization #############
def change_values_in_file(file_path, line_col_values):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for (line, col), value in line_col_values.items():
        # Subtracting 1 from line and col to convert to zero-based indexing
        lines[line - 1] = lines[line - 1][:col - 1] + str(value) + lines[line - 1][col:]

    with open(file_path, 'w') as file:
        file.writelines(lines)
        
def Vali_PM_OPS(Optimal_parameter_set, par_conf, reach_no, obs_flow, obs_flow_v,swat_loc):
    
    
    Perf1,Perf2, pred_flow_c = SWAT_SIMULATOR(Optimal_parameter_set, par_conf, reach_no, obs_flow,swat_loc)
    # print("Validation of optimized hydrological model through SWAT for Calibration period (RMSE): %f"%(Perf_SWAT))
    print("Validation of optimized hydrological model through SWAT for Calibration period (KGE): %f"%(Perf2))
    
    line_col_values = {
        (8, 15): " ",
        (8, 16): "5",
        (9, 13): "2",
        (9, 14): "0",
        (9, 15): "1",
        (9, 16): "1",
        (60, 16): "0"
        }

    change_values_in_file('file.cio', line_col_values)
    
     
    Perf1,Perf2, pred_flow_v = SWAT_SIMULATOR(Optimal_parameter_set, par_conf, reach_no, obs_flow_v,swat_loc)
    # print("Validation of optimized hydrological model through SWAT for Validation period (RMSE): %f"%(Perf_SWAT))
    print("Validation of optimized hydrological model through SWAT for Validation period (KGE): %f"%(Perf2))
    
    return Perf1,Perf2, pred_flow_c, pred_flow_v


################# Implement LOLA-Voronoi adaptive sampling ####################
from scipy.spatial import Voronoi
def lola_voronoi_adaptive_sampling(samples, model, n_new_samples, bounds):
    def compute_lola(samples, model):
        gradients = []
        for sample in samples:
            gradient = np.array([model.predict(sample.reshape(1, -1), return_std=True)[1]])
            gradients.append(gradient)
        return np.array(gradients).reshape(len(samples), -1)

    def sample_voronoi(samples, gradients, n_samples, bounds):
        vor = Voronoi(samples)
        new_samples = []
        
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                vertices = np.array([vor.vertices[i] for i in region])
                centroid = np.mean(vertices, axis=0)
                
                for dim, (lower, upper) in enumerate(bounds):
                    centroid[dim] = np.clip(centroid[dim], lower, upper)
                
                closest_sample_idx = np.argmin(np.linalg.norm(samples - centroid, axis=1))
                adjusted_centroid = centroid + gradients[closest_sample_idx]
                
                for dim, (lower, upper) in enumerate(bounds):
                    adjusted_centroid[dim] = np.clip(adjusted_centroid[dim], lower, upper)
                
                new_samples.append(adjusted_centroid)
                if len(new_samples) >= n_samples:
                    break

        return np.array(new_samples)

    lola_gradients = compute_lola(samples, model)
    new_samples = sample_voronoi(samples, lola_gradients, n_new_samples, bounds)
    return new_samples


################# Validation of Optimized set ####################
def Emulator_Optimization_GA_Def(param_ranges, par_conf, reach_no, obs_flow, swat_loc):
    
    def SWAT_SIM(X1):
        par_set = X1.tolist()
        of1, of2, of3 = SWAT_SIMULATOR(par_set, par_conf, reach_no, obs_flow, swat_loc)
        return -of2
    
    # Initialize lists to store iteration data
    iterations = []
    objective_functions = []

    def callback(algorithm):
        # This callback function is called at the end of each iteration
        iterations.append(algorithm.current_iteration)
        objective_functions.append(algorithm.best_function)
    
    algorithm_param = {
        'max_num_iteration': 100,
        'population_size': 10,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.25,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }

    GA_model = ga(
        function=SWAT_SIM,
        dimension=len(param_ranges),
        variable_type='real',
        variable_boundaries=np.array(param_ranges),
        algorithm_parameters=algorithm_param
    )

    GA_model.run()

    Optimal_parameter_set = GA_model.best_variable
    Perf_Optimization = -1 * GA_model.best_function
    Perf_Emulator, p1, p2 = SWAT_SIMULATOR(Optimal_parameter_set, par_conf, reach_no, obs_flow, swat_loc)

    print("Optimal Parameter Set:", Optimal_parameter_set)
    print("Optimized objective function through GA: %f" % Perf_Optimization)
    print("Validation of optimized hydrological model through Physical Model: %f" % p1)

    # Save iteration data to CSV
    iteration_df = pd.DataFrame({
        'Iteration': iterations,
        'Objective_Function': objective_functions
    })
    iteration_df.to_csv('GA_Iterations.csv', index=False)

    return Optimal_parameter_set, Perf_Optimization, Perf_Emulator, GA_model, iteration_df


############################### SWAT Pre-Process ##############################
def copy_files_with_replacement(source_folder, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)
        
        # If it's a file, copy it to the destination
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        # If it's a directory, use shutil.copytree
        elif os.path.isdir(source_path):
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            shutil.copytree(source_path, destination_path)


def txt_backup(Physical_Model_loc):

    source_folder = Physical_Model_loc+'\Backup'
    destination_folder = Physical_Model_loc
    
    copy_files_with_replacement(source_folder, destination_folder)
    
    os.chdir(Physical_Model_loc)
    line_col_values = {
        (8, 15): "1",
        (8, 16): "3",
        (9, 13): "1",
        (9, 14): "9",
        (9, 15): "9",
        (9, 16): "8",
        (60, 16): "3"
        }

    change_values_in_file('file.cio', line_col_values)

############################# Saving the results ##############################   
def save_variables(model,par_set,Cali_Pflow,Vali_Pflow,Optimal_PS,timing_list):  
    model.save_model('Fitted_model.xgb')  # Save the trained emulator model
    
    # Save parameter set
    df_par_set = pd.DataFrame(par_set)
    df_par_set.to_csv("Par_set.csv", index=False)
    
    # Save calibration and validation flow data
    df_Cali_Pflow = pd.DataFrame(Cali_Pflow)
    df_Vali_Pflow = pd.DataFrame(Vali_Pflow)
    df_Optimal_PS = pd.DataFrame()
    
    df_Cali_Pflow.to_csv('Calibration_Flow.csv', index=False)
    df_Vali_Pflow.to_csv('Validation_Flow.csv', index=False)
    df_Optimal_PS.to_csv('Optimal_set.csv', index=False)
    
    def save_all_variables(filename):
        global_vars = globals()
        # Filter out non-pickleable objects (like modules)
        pickleable_vars = {k: v for k, v in global_vars.items() if not isinstance(v, type(__import__('builtins')))}
        
        with open(filename, 'wb') as f:
            pickle.dump(pickleable_vars, f)
          
            
    def load_all_variables(filename):
        with open(filename, 'rb') as f:
            global_vars = pickle.load(f)
        globals().update(global_vars)
        
    # Save execution time for each step
    step_labels = [f"Step {i+1}" for i in range(len(timing_list))]
    timing_df = pd.DataFrame({'Step': step_labels, 'Time (seconds)': timing_list})
    timing_df.to_csv('Timing_results.csv', index=False)

    # Print execution times
    print("Time taken for each step (in seconds):", timing_list)