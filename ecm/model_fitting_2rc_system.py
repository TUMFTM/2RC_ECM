__author__ = "Lukas Merkle"
__copyright__ = "Copyright 2020, 31.07.20"
__email__ = 'lukas.merkle@tum.de'

import numpy as np
from ocv_soc_rel import OCV_SOC_REL
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import json
import pickle
import os
import copy
import sys
import multiprocessing
from multiprocessing import Pool
import glob

VERBOSE = 0
flag_save = False

def load_data():
    '''
    PLACEHOLDER

    If you want to simulate your own data NOT using the docker interface, you can implement data loading here.
    '''

    df_measures = pd.DataFrame()
    file_name = ""
    dt = 0.2

    return df_measures, file_name, dt



##################################################################
# Parameters
##################################################################
# Load Config data
settings = json.load(open("ocv_soc_settings.json","r"))["eGolf"]

aging_factor_capacity   = 1
PARALLEL_CELLS = settings["parallel_cells_per_block"]
PACK_CAPACITY = settings["system_capacity"] * aging_factor_capacity                 # Ah eGolf Rated Pack Energy: 24.2 kWh Rated Pack Capacity: 75 Ah
CELL_CAPACITY = PACK_CAPACITY / (PARALLEL_CELLS)                                    # Ah eGolf Rated Pack Energy: 24.2 kWh Rated Pack Capacity: 75 Ah / 3 parallel cells

## OCV relations
ocv_soc_rel_discharge   = OCV_SOC_REL(direction="discharge", vehicle="eGolf", aging_factor_capacity=aging_factor_capacity)
ocv_soc_rel_charge      = OCV_SOC_REL(direction="charge", vehicle="eGolf", aging_factor_capacity=aging_factor_capacity)
ocv_soc_rel_mean        = OCV_SOC_REL(direction="mean", vehicle="eGolf", aging_factor_capacity=aging_factor_capacity)


def get_start_parameter():
    '''
    Parameters
    - Startparameterset for mostly running into xtol
    '''
    f = 1
    r0_min = 0.01
    R0 = 0.2
    r0_max = 0.3

    r1_min = f * 0.001
    R1 = f * 0.01
    r1_max = 1

    c1_min = 50
    C1 = 120
    c1_max = 1000

    r2_min = f * 0.01
    R2 = f * 0.01
    r2_max = 10

    c2_min = 500
    C2 = 5000
    c2_max = 120000

    start_soc_min = 0.01
    start_soc = 0.5
    start_soc_max = 1

    capacity_min = 50
    capacity = 71
    capacity_max = 78

    p0 = [R0, R1, C1, R2, C2, start_soc, capacity]
    bounds_min = [r0_min, r1_min, c1_min, r2_min, c2_min, start_soc_min, capacity_min]
    bounds_max = [r0_max, r1_max, c1_max, r2_max, c2_max, start_soc_max, capacity_max]

    return p0, bounds_min, bounds_max

def get_defined_input_current(length_input, dt):
    '''
    :return: defined input step
    '''
    n = 500
    # Simulate Step
    i_step = np.zeros((int(length_input), 1))
    delta_ampere = 200  # 200A step
    i_step[10:10 + int(10 / dt)] = delta_ampere

    return i_step, delta_ampere


blend_threshold=10
blendfactor_discharge = lambda x: 1 if x>blend_threshold else 0.5+0.5*(x/blend_threshold)
##################################################################
# Model Equations
##################################################################
def simulate(i, R0, R1, C1, R2, C2, start_soc, capacity, dt, **kwargs):
    # R0 = 0.002
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
    U_R0 = np.zeros(n, )
    U_OCV = np.zeros(n, )
    U_OCV[0] = 330
    C_Ah = np.zeros(n, )
    C_Ah[0] = START_SOC * (capacity)
    U_K = np.zeros(n, )
    U_K[0] = 330

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
                # U_K[t] = ocv_soc_rel_discharge.get_ocv_from_capacity_system(C_Ah[t])

                bf_d = blendfactor_discharge(u[t])
                bf_c = 1 - bf_d
                U_K[t] = ocv_soc_rel_discharge.get_ocv_from_soc_system((C_Ah[t] / capacity)) * bf_d + ocv_soc_rel_charge.get_ocv_from_soc_system((C_Ah[t] / capacity)) * bf_c
            except:
                print("**WARNING: Skipped U_K-calculation and took old value")
                U_K[t] = U_K[t-1]

        if u[t] < 0:  # charging
            C_Ah[t] = C_Ah[t - 1] - i[t] * (dt / 3600)
            # SOC[t] = OCV_SOC_REL(direction="charge").get_soc_from_capacity_system(C_Ah[t])
            try:
                # U_K[t] = ocv_soc_rel_charge.get_ocv_from_capacity_system(C_Ah[t])

                bf_d = blendfactor_discharge(u[t])
                bf_c = 1 - bf_d
                # U_K[t] = ocv_soc_rel_charge.get_ocv_from_capacity_cell(C_Ah[t])*bf_c +  ocv_soc_rel_discharge.get_ocv_from_capacity_cell(C_Ah[t])*bf_d
                U_K[t] = ocv_soc_rel_charge.get_ocv_from_soc_system((C_Ah[t] / capacity)) * bf_c + ocv_soc_rel_discharge.get_ocv_from_soc_system((C_Ah[t] / capacity)) * bf_d

            except:
                print("**WARNING: Skipped U_K-calculation and took old value")
                U_K[t] = U_K[t-1]

        U_OCV[t] = C * x[:, t] + (-R0) * u[t] + U_K[t]

        U_RC1[t] = -x[0, t]
        U_RC2[t] = -x[1, t]
        U_R0[t] = (-R0) * u[t]

    if verbose:
        print("Capacity start: {} Ah. SOC start: {}".format(C_Ah[0], C_Ah[0]/(capacity* aging_factor_capacity)))
        print("Capacity end: {} Ah. SOC end: {}".format(C_Ah[-1], C_Ah[-1]/(capacity* aging_factor_capacity)))
        print(f"Delta Capacity: {C_Ah[0] - C_Ah[-1]} Ah. Delta SOC: {C_Ah[0] / (capacity * aging_factor_capacity) - C_Ah[-1] / (capacity * aging_factor_capacity)} %p.")

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
        # # f.savefig(f'results/fittingplots/fitting_{cc}_singleVs.png')
        # plt.show()
        # plt.close()

    return U_OCV

def simulate_residuals(x, i, v, dt):

    erg = simulate(i, x[0], x[1], x[2], x[3], x[4], x[5], x[6], dt)
    return v - erg

##################################################################
# Helper Functions
##################################################################
def find_start_soc_examine(current, voltage):
    THRESHOLD_I_ZERO = 5 # A
    SECONDS_ZERO_A = 4
    current = abs(current)
    start=False
    statisfied = False
    found_counter = 0
    for idx, i in enumerate(current):
        if i < THRESHOLD_I_ZERO and start == False:
            # start trajectory
            start=True
            idx_start = idx

        if i > THRESHOLD_I_ZERO and start == True:
            idx_end = idx-1

            if idx_end - idx_start > SECONDS_ZERO_A / dt and np.std(current[idx_start:idx_end]) < 0.4:
                # Statisfied!!
                statisfied = True
                start==False

                i_range = idx_end - idx_start
                plt.figure()
                plt.title("SOC-finder. Cell: {}. idx_start: {}".format(c,idx_start))

                # w, w2 is plotting window
                if idx_start>20:
                    w = 20
                    w2 = 20
                else:
                    w = 0
                    w2 = 20
                plt.plot(range(0,len(voltage[idx_start-w:idx_end+w2])), voltage[idx_start-w:idx_end+w2])
                plt.plot(range(0,len(current[idx_start-w:idx_end+w2])), current[idx_start-w:idx_end+w2])

                plt.plot(range(w, w+len(voltage[idx_start:idx_end])), voltage[idx_start:idx_end])
                plt.plot(range(w, w+len(current[idx_start:idx_end])), current[idx_start:idx_end])
                # plt.show()
                soc = ocv_soc_rel_mean.get_soc_from_ocv_system(np.mean(voltage[idx_end]))
                found_counter = found_counter+1
                print("Possible soc no. {} ----------------------".format(found_counter))
                print("Mean: {}".format(np.mean(current[idx_start:idx_end])))
                print("Std: {}".format(np.std(current[idx_start:idx_end])))
                print("soc: {} %".format(soc))
                print("start_idx: {}".format(idx_start))
                print("Length: {} s".format((idx_end - idx_start)*dt))
            else:
                start=False

def find_start_soc(current, voltage, dt):
    THRESHOLD_I_ZERO = 2 * PARALLEL_CELLS # A
    THRESHOLD_I_STD = 1.1
    SECONDS_ZERO_A = 3
    current = abs(current)
    start=False
    statisfied = False
    idx_start = 0
    idx_end = 0
    for idx, i in enumerate(current):
        if i < THRESHOLD_I_ZERO and start == False:
            # start trajectory
            start=True
            idx_start = idx

        if i > THRESHOLD_I_ZERO and start == True:
            idx_end = idx-1

            if idx_end - idx_start > SECONDS_ZERO_A / dt and np.nanstd(current[idx_start:idx_end]) < THRESHOLD_I_STD:
                # Statisfied!!
                statisfied = True
                start==False

                i_range = idx_end - idx_start

                # w, w2 is plotting window
                if idx_start>20:
                    w = 20
                    w2 = 20
                else:
                    w = 0
                    w2 = 20

                if VERBOSE > 0:
                    plt.figure()
                    # plt.title("SOC-finder. Cell: {}. idx_start: {}".format(c, idx_start))
                    plt.plot(range(0,len(voltage[idx_start-w:idx_end+w2])), voltage[idx_start-w:idx_end+w2])
                    plt.plot(range(0,len(current[idx_start-w:idx_end+w2])), current[idx_start-w:idx_end+w2])

                    plt.plot(range(w, w+len(voltage[idx_start:idx_end])), voltage[idx_start:idx_end])
                    plt.plot(range(w, w+len(current[idx_start:idx_end])), current[idx_start:idx_end])

                    plt.figure()
                    plt.title("current")
                    plt.plot(current)
                return ocv_soc_rel_mean.get_soc_from_ocv_system(np.mean(voltage[idx_end-3:idx_end]))
            else:
                # NOT STATISFIED. STD too big or snipped to short
                start=False

    else:
        print("** Info** : std: {}. idxstart: {}, idxend: {}".format(np.std(current[idx_start:idx_end]), idx_start, idx_end))
        if VERBOSE > 0:
            plt.figure()
            plt.title("current from soc finder. no startsoc found")
            plt.plot(current)
            plt.show()
        return -1


##################################################################
# fit Function to do all the work
##################################################################
def fit(df_measure, file_name, dt):
    fit_start_t = time.time()
    # idx = df_measure["idx"][0]
    idx =df_measure.index[0]
    #################################################################
    # Setup Parameters and Input Data
    #################################################################
    # p0 = [R0 * 1, R1 * 1, C1 * 1, R2 * 1, C2 * 1]
    p0, bounds_min, bounds_max = get_start_parameter()

    #################################################################
    # Looping the cells in the rawdata
    ################################################################
    results_vector=[]

    # Results name
    results_file_name = "results_{}".format(file_name)

    c = "eGolf_HV_System"

    # Real input current
    i = df_measure["current"].values
    i_in = -i

    # Input Voltage
    voltage_original = df_measure["voltage"].values

    param_name = c

    #################################################################
    # Fitting using scipy optimize least_squares
    #################################################################
    res_1 = least_squares(simulate_residuals, p0, args=(i_in, voltage_original, dt), method="trf", ftol=1e-8, xtol=9e-8, gtol=1e-8, verbose=2, bounds=(bounds_min, bounds_max), max_nfev=200,  x_scale=[100, 100, 0.1, 1000, 0.001, 1, 0.1])
    # res_1 = least_squares(simulate_residuals, p0, args=(i_in, voltage_original), method="trf", ftol=1e-10, xtol=9e-7, gtol=1e-8, verbose=0, max_nfev=2000)
    if flag_save:
        pickle.dump(res_1, open("params/system/" + results_file_name + "_{}_{}.p".format(param_name, idx),"wb"))

    # Print results
    # bounds_min = [r0_min, r1_min, c1_min, r2_min, c2_min]
    # bounds_max = [r0_max, r1_max, c1_max, r2_max, c2_max]
    [print(par, res_1.x[i], "({}, {})".format(bounds_min[i], bounds_max[i])) for i, par in enumerate(["R0", "R1", "C1", "R2", "C2", "soc", "capacity"])]
    print("Status: ", res_1.status, res_1.message)
    print("Tau1:", res_1.x[1]*res_1.x[2], "Tau2:", res_1.x[3]*res_1.x[4])
    print(f"* capacity fitted: {res_1.x[6]} Ah.")
    # Simulate with erg
    erg_fit = simulate(i_in, *res_1.x, dt, verbose=True)  # fitted parameters
    erg_orig = simulate(i_in, *p0, dt)  # original parameters

    # Simulate Step
    # i_step = np.zeros((len(i_in),1))
    # Ampere_step = 200 # 200A step
    # i_step[10:10+int(10/dt)] = Ampere_step
    # i_step, delta_ampere = get_defined_input_current(len(i_in), dt)
    standard_length = 30 / dt
    i_step, delta_ampere = get_defined_input_current(standard_length, dt)
    erg_step= simulate(i_step, *res_1.x, dt)
    r_0_10s = (erg_step[2] - erg_step[8+int(10/dt)]) / delta_ampere
    print("R_0_10s: {} Ohm".format(r_0_10s))

    # Print RMSE
    rmse_startparameters = np.sqrt(np.mean((voltage_original[10:] - erg_orig[10:]) ** 2))
    rmse = np.sqrt(np.mean((voltage_original[10:] - erg_fit[10:]) ** 2))
    # rmse = np.sqrt(np.mean((voltage_original[60:] - erg_fit[60:]) ** 2))
    print("RMSE of signal using startparameters vs. original voltage: {}".format(rmse_startparameters))
    print("RMSE of fitted signal vs. original voltage: {}".format(rmse))

    # Save results in results-object
    results = {}
    results["cell"] = c
    results["start_soc"] = res_1.x[5]
    results["r0"] = res_1.x[0]
    results["fitted_params"] = res_1
    results["rmse_fit"] = rmse
    results["r_0_10s"] = r_0_10s
    # results["pre"] = raw_data[idx]["pre"]
    # results["post"] = raw_data[idx]["post"]
    results_vector.append(results)
    results_file_name = "results_{}".format(file_name)
    print("")
    if flag_save:
        pickle.dump(results_vector,open("results/system/"+results_file_name + "_{}.p".format(idx), "wb"))

    if VERBOSE > 0:
        # Plot
        plt.figure()
        plt.title("cell: {}, rmse: {}".format(c, rmse))
        plt.plot(voltage_original, marker="", label="Original")
        plt.plot(erg_fit, label="ocv_model_fitted")
        # plt.plot(erg_orig, label="param_orig")
        plt.legend()

        plt.figure()
        plt.title("Step cell: {}, rmse: {}".format(c, rmse))
        plt.plot(erg_step, label="ocv_model_fitted step")
        plt.plot(i_step, label="i_step")
        plt.legend()



    # #################################################################
    # # Nyquist of Model, calculating Z directly
    # #################################################################
    # R0 = 1
    # R1 = 1
    # C1 = 0.01
    # R2 = 0.003
    # C2 = 10000
    log_space = np.logspace(-6, 3, 500)
    re = lambda R0, R1, C1, R2, C2, f: R0 +  ((R1) / (1+(2*np.pi*f*R1*C1)**2)) + ((R2) / (1+(2*np.pi*f*R2*C2)**2))
    im = lambda R0, R1, C1, R2, C2, f:(2*np.pi*f*C1*(R1**2)) / ((1+(2*np.pi*f*R1*C1)**2)) + (2*np.pi*f*C2*(R2**2)) / ((1+(2*np.pi*f*R2*C2)**2))

    if VERBOSE > 0:
        plt.show()
    print("Duration single fit: {}s".format(time.time() - fit_start_t))

    return res_1, rmse


def inference_Ri(fitted_params, i=None, dt=0.2, start_soc=0.5, **kwargs):

    print("** Inference started...")
    # If no i given, use the predefined 10s i_step
    if i==None:
        standard_length = 30/dt  # standard length should be 30s.
        i_step, delta_ampere = get_defined_input_current(standard_length, dt)
        i=i_step

    erg_step = simulate(i, *fitted_params, dt)

    r_0 = (erg_step[2] - erg_step[8+int(10/dt)]) / delta_ampere
    print("** Result: R_0: {} Ohm".format(r_0))

    return r_0


if __name__ == "__main__":

    '''
    Only needed, if you want to run against local data without the docker container
    '''

    # for idx, df_measure in enumerate(df_measures):
    df_measures, file_name, dt= load_data()
    ts = time.time()

    t1 = fit(df_measure=df_measures[0], file_name="test", dt=0.2)

    # multi process to fit and infer df_measures parallely
    pool = Pool(10)
    results = [pool.apply_async(fit, args=(df_measure,file_name, dt)) for df_measure in df_measures]
    pool.close()
    pool.join()

    test_params = results[0]._value[0].x
    inference_Ri(test_params, i=None, dt=dt, start_soc=0.5)

    print("Duration: {} s".format(time.time() - ts))
    print("Ende")

