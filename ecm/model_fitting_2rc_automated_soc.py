__author__ = "Lukas Merkle"
__copyright__ = "Copyright 2020, 31.07.20"
__email__ = 'lukas.merkle@tum.de'

import numpy as np
from ocv_soc_rel import OCV_SOC_REL
import time
from scipy.optimize import least_squares
import json
import pickle
import os
import matplotlib.pyplot as plt


def load_data():
    '''
    PLACEHOLDER

    If you want to simulate your own data NOT using the docker interface, you can implement data loading here.
    '''

    raw_data = pd.DataFrame()
    file_name = ""
    dt = 0.2
    label_current="XXXX"

    return raw_data, file_name, dt, label_current

# load config data
settings = json.load(open("ocv_soc_settings.json","r"))["eGolf"]

aging_factor_capacity = 1
PARALLEL_CELLS = settings["parallel_cells_per_block"]
PACK_CAPACITY = settings["system_capacity"] * aging_factor_capacity # Ah eGolf Rated Pack Energy: 24.2 kWh Rated Pack Capacity: 75 Ah
CELL_CAPACITY = PACK_CAPACITY / (PARALLEL_CELLS) # Ah eGolf Rated Pack Energy: 24.2 kWh Rated Pack Capacity: 75 Ah / 3 parallel cells

# OCV relations
ocv_soc_rel_discharge   = OCV_SOC_REL(direction="mean", vehicle="eGolf", aging_factor_capacity=aging_factor_capacity)
ocv_soc_rel_charge      = OCV_SOC_REL(direction="mean", vehicle="eGolf", aging_factor_capacity=aging_factor_capacity)
ocv_soc_rel_mean        = OCV_SOC_REL(direction="mean", vehicle="eGolf", aging_factor_capacity=aging_factor_capacity)


def get_start_parameter():
    '''
    Definition to get start parameters
    '''
    
    r0_min = 0.0008
    R0 = 0.002
    r0_max = 0.008
    r1_min = 0.00001
    R1 = 0.001
    r1_max = 0.01
    c1_min = 10
    C1 = 90
    c1_max = 80000
    r2_min = 0.0001
    R2 = 0.005
    r2_max = 0.01
    c2_min = 1000
    C2 = 10000
    c2_max = 50000
    start_soc_min= 0.01
    start_soc= 0.95
    start_soc_max=1
    CELL_CAPACITY_min=19
    CELL_CAPACITY=23
    CELL_CAPACITY_max=28

    p0 = [R0, R1, C1, R2, C2, start_soc, CELL_CAPACITY]
    bounds_min = [r0_min, r1_min, c1_min, r2_min, c2_min, start_soc_min, CELL_CAPACITY_min]
    bounds_max = [r0_max, r1_max, c1_max, r2_max, c2_max, start_soc_max, CELL_CAPACITY_max]

    return p0, bounds_min, bounds_max

blend_threshold=10
blendfactor_discharge = lambda x: 1 if x>blend_threshold else 0.5+0.5*(x/blend_threshold)

# Definition to simulate U_OCV
cc=0
def simulate(i, R0, R1, C1, R2, C2, start_soc, CELL_CAPACITY, dt, **kwargs):
    global cc
    
    if "verbose" in kwargs:
        verbose=True
    else:
        verbose= False

    # if "START_SOC" in kwargs:
    #     START_SOC=kwargs["START_SOC"]
    # else:
    #     print("**ERROR**: No start_soc given!")

    START_SOC = start_soc

    n = len(i)
    U_RC1 = np.zeros(n, )
    U_RC2 = np.zeros(n, )
    U_OCV = np.zeros(n, )
    U_R0 = np.zeros(n, )
    U_OCV[0] = 3.7
    C_Ah = np.zeros(n, )
    C_Ah[0] = START_SOC * (CELL_CAPACITY)
    U_K = np.zeros(n, )
    U_K[0] = 3.7

    A = np.matrix([[np.exp(-dt / (R1 * C1)), 0],
                        [0, np.exp(-dt / (R2 * C2))]])

    B = np.matrix([dt / C1, dt / C2])
    C = np.array([-1, -1])

    x = np.zeros((2, n))
    x = np.matrix([U_RC1, U_RC2])
    u = i

    for t in range(1,n):
        x[:, t] = A * x[:, t-1] + B.T * u[t]

        # Ampere counting discharge using ocv rel
        if u[t] >= 0:  # discharging
            C_Ah[t] = C_Ah[t - 1] - i[t] * (dt / 3600)
            # SOC[t] = OCV_SOC_REL(direction="discharge").get_soc_from_capacity_system(C_Ah[t])
            try:
                bf_d = blendfactor_discharge(u[t])
                bf_c = 1-bf_d
                # U_K[t] = ocv_soc_rel_discharge.get_ocv_from_capacity_cell(C_Ah[t])*bf_d + ocv_soc_rel_charge.get_ocv_from_capacity_cell(C_Ah[t])*bf_c
                U_K[t] = ocv_soc_rel_discharge.get_ocv_from_soc_cell((C_Ah[t]/CELL_CAPACITY))*bf_d + ocv_soc_rel_charge.get_ocv_from_soc_cell((C_Ah[t]/CELL_CAPACITY))*bf_c
            except:
                print("**WARNING: Skipped U_K-calculation and took old value")
                U_K[t] = U_K[t-1]

        if u[t] < 0:  # charging
            C_Ah[t] = C_Ah[t - 1] - i[t] * (dt / 3600)
            # SOC[t] = OCV_SOC_REL(direction="charge").get_soc_from_capacity_system(C_Ah[t])
            try:
                bf_d = blendfactor_discharge(u[t])
                bf_c = 1 - bf_d
                # U_K[t] = ocv_soc_rel_charge.get_ocv_from_capacity_cell(C_Ah[t])*bf_c +  ocv_soc_rel_discharge.get_ocv_from_capacity_cell(C_Ah[t])*bf_d
                U_K[t] = ocv_soc_rel_charge.get_ocv_from_soc_cell((C_Ah[t]/CELL_CAPACITY))*bf_c +  ocv_soc_rel_discharge.get_ocv_from_soc_cell((C_Ah[t]/CELL_CAPACITY))*bf_d
            except:
                print("**WARNING: Skipped U_K-calculation and took old value")
                U_K[t] = U_K[t-1]

        U_OCV[t] = C * x[:, t] + (-R0) * u[t] + U_K[t]

        U_RC1[t] = -x[0, t]
        U_RC2[t] = -x[1, t]
        U_R0[t] = (-R0) * u[t]

    if verbose:
        print("Capacity start: {} Ah. SOC start: {}".format(C_Ah[0], C_Ah[0]/(CELL_CAPACITY* aging_factor_capacity)))
        print("Capacity end: {} Ah. SOC end: {}".format(C_Ah[-1], C_Ah[-1]/(CELL_CAPACITY* aging_factor_capacity)))
        print(f"Delta Capacity: {C_Ah[0] - C_Ah[-1]} Ah. Delta SOC: {C_Ah[0]/(CELL_CAPACITY* aging_factor_capacity) - C_Ah[-1]/(CELL_CAPACITY* aging_factor_capacity)} %p.")
        # f, axs = plt.subplots(3,1, figsize=(20,20))
        # axs[0].plot(U_R0, label="U_R0")
        # axs[0].plot(U_RC1, label="U_RC1")
        # axs[0].plot(U_RC2, label="U_RC2")
        # axs[1].plot(U_K, label="U_K")
        # axs[1].plot(U_OCV, label="U_OCV")
        # axs[2].plot(u, label="current")
        # axs[0].legend()
        # axs[1].legend()
        # axs[2].legend()
        # f.savefig(f'results/fittingplots/fitting_{cc}_singleVs.png')
        # # plt.show()
        # plt.close()

    return U_OCV

# Definition to simulate the residuals
def simulate_residuals(x, i, v, dt, START_SOC):
    erg = simulate(i, x[0], x[1], x[2], x[3], x[4], x[5],x[6], dt, START_SOC=START_SOC)
    return v - erg

# Definition to find the start soc
def find_start_soc(current, voltage, dt):
    THRESHOLD_I_ZERO = 2 # A
    SECONDS_ZERO_A = 3 # seconds
    current = abs(current)
    start=False
    statisfied = False
    for idx, i in enumerate(current):
        if i < THRESHOLD_I_ZERO and start == False:
            # start trajectory
            start=True
            idx_start = idx

        if i > THRESHOLD_I_ZERO and start == True:
            idx_end = idx-1

            if idx_end - idx_start > SECONDS_ZERO_A / dt and np.std(current[idx_start:idx_end]) < 1:
                # Statisfied!!
                statisfied = True
                start==False

                i_range = idx_end - idx_start

                # w, w2 is plotting window
                if idx_start>10:
                    w = 10
                    w2 = 10
                else:
                    w = 0
                    w2 = 10
                return ocv_soc_rel_mean.get_soc_from_ocv_cell(np.mean(voltage[idx_end]))
            else:
                # NOT STATISFIED. STD too big or snipped to short
                start=False

    else:
        return -1

# Definition to get the defined input current
def get_defined_input_current(length_input, dt):
    n = 500

    # Simulate Step
    i_step = np.zeros((int(length_input), 1))
    delta_ampere = 50  # 50A step
    i_step[10:10 + int(10 / dt)] = delta_ampere

    return i_step, delta_ampere

# Definition for model fitting
def fit(raw_data, dt, label_current):
    #################################################################
    # Setup Parameters and Input Data
    #################################################################
    p0, bounds_min, bounds_max = get_start_parameter()

    #################################################################
    # Looping the cells in the rawdata
    ################################################################
    # Datasource 1: files on harddrive
    if "label" in raw_data:
        flag_save = True
        # for idx in range(1, len(raw_data)):
        c = raw_data["label"]

        df_measure = raw_data["data"]
        # Real input current
        i = df_measure[label_current] / (- PARALLEL_CELLS)

        # smooth i
        # i_smoother= lambda x: 0 if x<=2 and x>=-2 else x
        # i = i.apply(i_smoother)

        idx_slice = df_measure[c].notna()
        voltage_original = df_measure[c].loc[idx_slice].values / 1000
        i_in = i.loc[idx_slice].values
        param_name = c
        
    # Datasource 2: in docker model from Datalake
    # Dataframe with this signature: {"current": cycle.Cycle.current.value, "voltage": cycle.Cycle.voltage.value, "time": cycle.Cycle.timestamp.value}
    else:
        flag_save = False
        i = raw_data["current"] / (- PARALLEL_CELLS)
        idx_slice = raw_data["voltage"].notna()
        voltage_original = raw_data["voltage"].loc[idx_slice].values / 1000
        i_in = i.loc[idx_slice].values
        c = "data_from_datalake"
        param_name=c

    # We have to estimate the SOC from ocv-soc before fitting
    START_SOC = find_start_soc(i_in, voltage_original, dt) / 100
    # if START_SOC == -1/100:
    #     print("** INFO **: No legal Start SOC found")
    #     return -1
    print("Cellname: {}".format(c))
    print("Got start SOC: {}".format(START_SOC))

    #################################################################
    # Fitting using scipy optimize least_squares
    #################################################################
    # res_1 = least_squares(simulate_residuals, p0, args=(i_in, voltage_original, dt, START_SOC), method="trf", ftol=1e-10, xtol=9e-7, gtol=1e-8, verbose=2, bounds=(bounds_min, bounds_max), max_nfev=600)
    res_1 = least_squares(simulate_residuals, p0, args=(i_in, voltage_original, dt, START_SOC), method="trf", ftol=1e-8, xtol=9e-8, gtol=1e-8, verbose=2, bounds=(bounds_min, bounds_max), max_nfev=200, x_scale=[100, 100, 0.1, 1000, 0.001, 1, 0.1])


    print("solution found")
#    if flag_save:
#        pickle.dump(res_1, open("params/cells/2rc_model_params_{}".format(param_name),"wb"))

    # print results
    [print(par, res_1.x[i]) for i, par in enumerate(["R0", "R1", "C1", "R2", "C2"])]
    print("Status: ", res_1.status, res_1.message)
    print("Tau1:", res_1.x[1]*res_1.x[2], "Tau2:", res_1.x[3]*res_1.x[4])

    # simulate with erg
    erg_fit = simulate(i_in, *res_1.x, dt, START_SOC=START_SOC, verbose=True)  # fitted parameters
    erg_orig = simulate(i_in, *p0, dt, START_SOC=START_SOC)  # original parameters

    # simulate Step
    # i_step, delta_ampere = get_defined_input_current(len(i_in), dt)
    standard_length = 30 / dt
    i_step, delta_ampere = get_defined_input_current(standard_length, dt)
    erg_step = simulate(i_step, *res_1.x, dt, START_SOC=START_SOC)
    r_0_10s = (erg_step[2] - erg_step[8 + int(10 / dt)]) / delta_ampere
    print("R_0_10s: {} Ohm".format(r_0_10s))

    # print RMSE
    rmse_startparameters = np.sqrt(np.mean((voltage_original[1:] - erg_orig[1:]) ** 2))
    rmse = np.sqrt(np.mean((voltage_original[1:] - erg_fit[1:]) ** 2))
    print("RMSE of signal using startparameters vs. original voltage: {}".format(rmse_startparameters))
    print("RMSE of fitted signal vs. original voltage: {}".format(rmse))

    # save results in results-object
    if flag_save:
        results = {}
        results["cell"] = c
        results["start_soc"] = START_SOC
        results["r0"] = res_1.x[0]
        results["r_0_10s"] = r_0_10s
        results["fitted_params"] = res_1
        results["rmse_fit"] = rmse
        results["pre"] = raw_data["pre"]
        results["data"] = raw_data["data"]
        results["post"] = raw_data["post"]

    return res_1, START_SOC, rmse

# Definition inference Ri
def inference_Ri(fitted_params, i=None, dt=0.2, start_soc=0.5, **kwargs):


    print("** Inference started...")
    # If no i given, use the predefined 10s i_step
    if i==None:
        standard_length = 30/dt  # standard length should be 30s.
        i_step, delta_ampere = get_defined_input_current(standard_length, dt)
        i=i_step

    print("** Test_params: {}, dt: {}, start_soc: {}".format(fitted_params, dt, start_soc))
    erg_step = simulate(i, *fitted_params, dt=dt, verbose=True)

    r_0 = (erg_step[2] - erg_step[8+int(10/dt)]) / delta_ampere

    print("** Result: R_0_10s: {} Ohm".format(r_0))
    return r_0

if __name__ == "__main__":
    start_time = time.time()
    
    raw_datas, file_name, dt, label_current = load_data()

    results_vector = [] 
    for idx, raw_data in enumerate(raw_datas):
        cc=idx
        r = fit(raw_data, dt, label_current)
        if r==-1:
            continue
        print("##################")
        print(r[0].x)
        print(r[1])
        print(f"* capacity fitted: {r[0].x[6]} Ah.")

        test_params = r[0].x
        START_SOC = r[1]

        # print(inference_Ri(test_params, i=None, dt=dt, start_soc=START_SOC))
        
        print("##################")
        results_vector.append(r[2])

        # test parameterfitting
        #######################
        df_measure = raw_data["data"]
        c = raw_data["label"]
        idx_slice = df_measure[c].notna()
        my_i = df_measure[label_current] / (- PARALLEL_CELLS)

        my_u = df_measure[c].loc[idx_slice].values / 1000
        Spannung_Modell = simulate(my_i, *r[0].x, dt = dt, verbose=True)

        # f, axs = plt.subplots(2,1, figsize=(25,25))
        # f.suptitle(f"rmse: {r[2]['rmse_fit']}")
        # axs[0].plot(Spannung_Modell, label="modell")
        # axs[0].plot(my_u, label="in U")
        # axs[1].plot(my_i, label="in I")
        # axs[0].legend()
        # axs[1].legend()
        # f.savefig(f'results/fittingplots/fitting_{idx}_start_soc_blend_opt.png')
        # # plt.show()
        # plt.close()
        #######################
               
    # results_file_name = "results_correct_current_{}".format(file_name)
    # pickle.dump(results_vector,open("results/cells/"+results_file_name, "wb"))
    
    # print duration time
    duration = time.time() - start_time
    print("\nExecution time: {}s.".format(duration))