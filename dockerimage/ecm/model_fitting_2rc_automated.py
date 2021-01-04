import numpy as np
from ocv_soc_rel import OCV_SOC_REL
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import json
import pickle
import os
import multiprocessing
from multiprocessing import Pool

##################################################################
# Load Data from cell_rotator --> Automatically recorded
##################################################################
def load_data():
    path = "../../IoT/eGolf/egolf_cell_rotator/data/cycles_30s_hv_total_all_cells_2.p"
    # path = "../egolf_cell_rotator/data/cycles_30s_hv_total_2.p"
    raw_data = pickle.load(open(path, "rb"))
    file_name = os.path.split(path)[1]
    label_current = 7741
    label_voltages = [7744,
                        7745,
                        7746,
                        7747,
                        7748,
                        7749,
                        7750,
                        7751,
                        7752,
                        7753,
                        7754,
                        7755,
                        7756,
                        7757,
                        7758,
                        7759,
                        7760,
                        7761,
                        7762,
                        7763,
                        7764,
                        7765,
                        7766,
                        7767,
                        7768,
                        7769,
                        7770,
                        7771,
                        7772,
                        7773,
                        7774,
                        7775,
                        7776,
                        7777,
                        7778,
                        7779,
                        7780,
                        7781,
                        7782,
                        7783,
                        7784,
                        7785,
                        7786,
                        7787,
                        7788,
                        7789,
                        7790,
                        7791,
                        7792,
                        7793,
                        7794,
                        7795,
                        7796,
                        7797,
                        7798,
                        7799,
                        7800,
                        7801,
                        7802,
                        7803,
                        7804,
                        7805,
                        7806,
                        7807,
                        7808,
                        7809,
                        7810,
                        7811,
                        7812,
                        7813,
                        7814,
                        7815,
                        7816,
                        7817,
                        7818,
                        7819,
                        7820,
                        7821,
                        7822,
                        7823,
                        7824,
                        7825,
                        7826,
                        7827,
                        7828,
                        7829,
                        7830,
                        7831]

    # Resample data to a common dt
    raw_data = [{"data": x["data"].resample("500ms").mean(), "label": x["label"], "pre": x["pre"], "post":x["post"]} for x in raw_data]
    raw_data = [x for x in raw_data if x["label"] in label_voltages]
    dt = 0.5

    return raw_data, file_name, dt, label_current

# plt.figure()
# plt.title("cell rollator")
# raw_data[1]["data"][7741].plot(label="roll curr")
# raw_data[1]["data"][7745].plot(label="roll volt")
# plt.show()

##################################################################
# Parameters
##################################################################
# Load Config data
settings = json.load(open("ocv_soc_settings.json","r"))["aCar_eGolf"]

aging_factor_capacity   = 1
PARALLEL_CELLS = settings["parallel_cells_per_block"]
PACK_CAPACITY = settings["system_capacity"] * aging_factor_capacity                 # Ah eGolf Rated Pack Energy: 24.2 kWh Rated Pack Capacity: 75 Ah
CELL_CAPACITY = PACK_CAPACITY / (PARALLEL_CELLS)                                    # Ah eGolf Rated Pack Energy: 24.2 kWh Rated Pack Capacity: 75 Ah / 3 parallel cells
# START_SOC = 0.64  # 0.64

## OCV relations
ocv_soc_rel_discharge   = OCV_SOC_REL(direction="discharge", vehicle="aCar_eGolf", aging_factor_capacity=aging_factor_capacity)
ocv_soc_rel_charge      = OCV_SOC_REL(direction="charge", vehicle="aCar_eGolf", aging_factor_capacity=aging_factor_capacity)
ocv_soc_rel_mean        = OCV_SOC_REL(direction="mean", vehicle="aCar_eGolf", aging_factor_capacity=aging_factor_capacity)

# Parameters
def get_start_parameter():
    # Startparameters for running into all exits xtol, ftol, gtol
    # r0_min = 0.001
    # R0 = 0.001
    # r0_max = 0.012
    # r1_min = 0.0001
    # R1 = 0.001
    # r1_max = 0.03
    # c1_min = 0
    # C1 = 100
    # c1_max = 180
    # r2_min = 0.0001
    # R2 = 0.001
    # r2_max = 0.03
    # c2_min = 0
    # C2 = 10000
    # c2_max = 50000

    # Startparameterset for mostly running into xtol
    r0_min = 0.001
    R0 = 0.001
    r0_max = 0.008
    r1_min = 0.0001
    R1 = 0.001
    r1_max = 0.01
    c1_min = 90
    C1 = 100
    c1_max = 150
    r2_min = 0.0001
    R2 = 0.001
    r2_max = 0.01
    c2_min = 1000
    C2 = 1000
    c2_max = 5000

    p0 = [R0, R1, C1, R2, C2]
    bounds_min = [r0_min, r1_min, c1_min, r2_min, c2_min]
    bounds_max = [r0_max, r1_max, c1_max, r2_max, c2_max]

    return p0, bounds_min, bounds_max



# n = 500
#
# Fake EIS Current input
# fake_index = pd.date_range(start="1.1.2018 00:00:00", end="1.1.2018 01:00:00", freq="0.5S")
# excitation= np.zeros(fake_index.size, )
# f=0.1  # 1/s
# excitation = np.sin([x * 2 * np.pi * f for x in list(range(0, len(fake_index)))])
# i_fake = pd.Series(excitation, index=fake_index)



##################################################################
# Model Equations
##################################################################
def simulate(i, R0, R1, C1, R2, C2, dt, **kwargs):
    # R0 = 0.002
    if "verbose" in kwargs:
        verbose=True
    else:
        verbose= False

    if "START_SOC" in kwargs:
        START_SOC=kwargs["START_SOC"]
    else:
        print("**ERROR**: No start_soc given!")

    n = len(i)
    U_RC1 = np.zeros(n, )
    U_RC2 = np.zeros(n, )
    U_OCV = np.zeros(n, )
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
                U_K[t] = ocv_soc_rel_discharge.get_ocv_from_capacity_cell(C_Ah[t])
            except:
                print("**WARNING: Skipped U_K-calculation and took old value")
                U_K[t] = U_K[t-1]

        if u[t] < 0:  # charging
            C_Ah[t] = C_Ah[t - 1] - i[t] * (dt / 3600)
            # SOC[t] = OCV_SOC_REL(direction="charge").get_soc_from_capacity_system(C_Ah[t])
            try:
                U_K[t] = ocv_soc_rel_charge.get_ocv_from_capacity_cell(C_Ah[t])
            except:
                print("**WARNING: Skipped U_K-calculation and took old value")
                U_K[t] = U_K[t-1]

        U_OCV[t] = C * x[:, t] + (R0) * u[t] + U_K[t]

    if verbose:
        print("Capacity start: {} Ah. SOC start: {}".format(C_Ah[0], C_Ah[0]/(CELL_CAPACITY* aging_factor_capacity)))
        print("Capacity end: {} Ah. SOC end: {}".format(C_Ah[-1], C_Ah[-1]/(CELL_CAPACITY* aging_factor_capacity)))
    return U_OCV

def simulate_residuals(x, i, v, dt, START_SOC):

    # x_scale = np.array([1 / 2.12022750e-03, 1 / 0.0001, 1 / 100, 1 / 0.0001, 1 / 1.00008383e+04])
    # x = np.array(x) / x_scale

    erg = simulate(i, x[0], x[1], x[2], x[3], x[4], dt, START_SOC=START_SOC)
    return v - erg

##################################################################
# Helper Functions
##################################################################
def find_start_soc(current, voltage, dt):
    THRESHOLD_I_ZERO = 2 # A
    SECONDS_ZERO_A = 2
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
                # plt.figure()
                # plt.title("SOC-finder. Cell: {}. idx_start: {}".format(c,idx_start))

                # w, w2 is plotting window
                if idx_start>10:
                    w = 10
                    w2 = 10
                else:
                    w = 0
                    w2 = 10

                # plt.plot(range(0,len(voltage[idx_start-w:idx_end+w2])), voltage[idx_start-w:idx_end+w2])
                # plt.plot(range(0,len(current[idx_start-w:idx_end+w2])), current[idx_start-w:idx_end+w2])
                #
                # plt.plot(range(w, w+len(voltage[idx_start:idx_end])), voltage[idx_start:idx_end])
                # plt.plot(range(w, w+len(current[idx_start:idx_end])), current[idx_start:idx_end])
                # plt.show()
                return ocv_soc_rel_mean.get_soc_from_ocv_cell(np.mean(voltage[idx_end]))
            else:
                # NOT STATISFIED. STD too big or snipped to short
                start=False

    else:
        return -1




    # for idx, i in enumerate(current):
    #     if abs(i) < THRESHOLD_I_ZERO:
    #         i_sum = 0
    #         i_range = 10
    #         for idx_2 in range(0, i_range):
    #             i_sum = i_sum + current[idx+idx_2]
    #         i_mean = i_sum/i_range
    #         if abs(i_mean) < THRESHOLD_I_ZERO:
    #             plt.figure()
    #             plt.title(c)
    #             plt.plot(range(0,10+i_range+9), voltage[idx-10:idx+idx_2+10])
    #             plt.plot(range(0,10+i_range+9),current[idx-10:idx+idx_2+10])
    #             plt.plot(range(10, 9+i_range),voltage[idx:idx+idx_2])
    #             plt.plot(range(10, 9+i_range),current[idx:idx+idx_2])
    #             plt.plot(10+idx_2-1, voltage[idx+idx_2-1:idx+idx_2], marker="*")
    #             plt.show()
    #             return ocv_soc_rel_mean.get_soc_from_ocv_cell(np.mean(voltage[idx+idx_2-1:idx+idx_2]))
    # else:
    #     return -1

def get_defined_input_current(length_input, dt):
    '''
    :return: defined input step
    '''
    n = 500
    # # Fake EIS Current input
    # fake_index = pd.date_range(start="1.1.2018 00:00:00", end="1.1.2018 01:00:00", freq="0.5S")
    # excitation= np.zeros(fake_index.size, )
    # f=0.1  # 1/s
    # excitation = np.sin([x * 2 * np.pi * f for x in list(range(0, len(fake_index)))])
    # i_fake = pd.Series(excitation, index=fake_index)
    #
    # return

    # Simulate Step
    i_step = np.zeros((int(length_input), 1))
    delta_ampere = 50  # 50A step
    i_step[10:10 + int(10 / dt)] = delta_ampere

    return i_step, delta_ampere

def fit(raw_data, dt, label_current):
    #################################################################
    # Setup Parameters and Input Data
    #################################################################
    p0, bounds_min, bounds_max = get_start_parameter()
    # p0 = [R0 * 1, R1 * 1, C1 * 1, R2 * 1, C2 * 1]


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
        i = - df_measure[label_current] / (- PARALLEL_CELLS)

        idx_slice = df_measure[c].notna()
        voltage_original = df_measure[c].loc[idx_slice].values / 1000
        i_in = i.loc[idx_slice].values
        param_name = c

    # Datasource 2: in docker model from Datalake
    # Dataframe with this signature: {"current": cycle.Cycle.current.value, "voltage": cycle.Cycle.voltage.value, "time": cycle.Cycle.timestamp.value}
    else:
        flag_save = False
        i = - raw_data["current"] / (- PARALLEL_CELLS)
        idx_slice = raw_data["voltage"].notna()
        voltage_original = raw_data["voltage"].loc[idx_slice].values / 1000
        i_in = i.loc[idx_slice].values
        c = "data_from_datalake"
        param_name=c

    # We have to estimate the SOC from ocv-soc before fitting.
    START_SOC = find_start_soc(i_in, voltage_original, dt) / 100
    if START_SOC == -1:
        print("** INFO **: No legal Start SOC found")
        return -1
    print("Cellname: {}".format(c))
    print("Got start SOC: {}".format(START_SOC))

    results_vector=[]
    #################################################################
    # Fitting using scipy optimize least_squares
    #################################################################
    res_1 = least_squares(simulate_residuals, p0, args=(i_in, voltage_original, dt, START_SOC), method="trf", ftol=1e-10, xtol=9e-7, gtol=1e-8, verbose=0, bounds=(bounds_min, bounds_max), max_nfev=2000)
    if flag_save:
        pickle.dump(res_1, open("params/cells/2rc_model_params_{}".format(param_name),"wb"))

    # Print results
    [print(par, res_1.x[i]) for i, par in enumerate(["R0", "R1", "C1", "R2", "C2"])]
    print("Status: ", res_1.status, res_1.message)
    print("Tau1:", res_1.x[1]*res_1.x[2], "Tau2:", res_1.x[3]*res_1.x[4])

    # Simulate with erg
    erg_fit = simulate(i_in, *res_1.x, dt, START_SOC=START_SOC, verbose=True)  # fitted parameters
    erg_orig = simulate(i_in, *p0, dt, START_SOC=START_SOC)  # original parameters

    # Simulate Step
    # i_step, delta_ampere = get_defined_input_current(len(i_in), dt)
    standard_length = 30 / dt
    i_step, delta_ampere = get_defined_input_current(standard_length, dt)
    erg_step = simulate(i_step, *res_1.x, dt, START_SOC=START_SOC)
    r_0_10s = (erg_step[2] - erg_step[8 + int(10 / dt)]) / delta_ampere
    print("R_0_10s: {} Ohm".format(r_0_10s))

    # Print RMSE
    rmse_startparameters = np.sqrt(np.mean((voltage_original[1:] - erg_orig[1:]) ** 2))
    rmse = np.sqrt(np.mean((voltage_original[1:] - erg_fit[1:]) ** 2))
    print("RMSE of signal using startparameters vs. original voltage: {}".format(rmse_startparameters))
    print("RMSE of fitted signal vs. original voltage: {}".format(rmse))

    # Save results in results-object
    if flag_save:
        results = {}
        results["cell"] = c
        results["start_soc"] = START_SOC
        results["r0"] = res_1.x[0]
        results["r_0_10s"] = r_0_10s
        results["fitted_params"] = res_1
        results["rmse_fit"] = rmse
        results["pre"] = raw_data["pre"]
        results["post"] = raw_data["post"]
        results_vector.append(results)
        results_file_name = "results_{}".format(file_name)
        pickle.dump(results_vector,open("results/cells/"+results_file_name, "wb"))

    # Plot
    # plt.figure()
    # plt.title("cell: {}, rmse: {}".format(c, rmse))
    # plt.plot(voltage_original, marker="", label="Original")
    # plt.plot(erg_fit, label="ocv_model_fitted")
    # # plt.plot(erg_orig, label="param_orig")
    # plt.legend()


    # # #################################################################
    # # # Nyquist of Model, calculating Z directly
    # # #################################################################
    # # R0 = 1
    # # R1 = 1
    # # C1 = 0.01
    # # R2 = 0.003
    # # C2 = 10000
    # log_space = np.logspace(-6, 3, 500)
    # re = lambda R0, R1, C1, R2, C2, f: R0 +  ((R1) / (1+(2*np.pi*f*R1*C1)**2)) + ((R2) / (1+(2*np.pi*f*R2*C2)**2))
    # im = lambda Ro, R1, C1, R2, C2, f:(2*np.pi*f*C1*(R1**2)) / ((1+(2*np.pi*f*R1*C1)**2)) + (2*np.pi*f*C2*(R2**2)) / ((1+(2*np.pi*f*R2*C2)**2))
    #
    # plt.figure()
    # for idx,f in enumerate(log_space.tolist()):
    #     re_ = re(*res_1.x, f)
    #     im_ = im(*res_1.x, f)
    #     #print("Freq: {}, re: {}, im: {}".format(f, re_, im_))
    #     plt.scatter(re_, im_)
    #     if idx % 10 == 0:
    #         plt.annotate(round(f, 3), (re_, im_))
    #
    # plt.grid()
    # plt.xlabel("ReZ Real")
    # plt.ylabel("-ImZ Imaginary")
    #
    # plt.show()

    return res_1, START_SOC

def inference_Ri(fitted_params, i=None, dt=0.2, start_soc=0.5, **kwargs):

    print("** Inference started...")
    # If no i given, use the predefined 10s i_step
    if i==None:
        standard_length = 30/dt  # standard length should be 30s.
        i_step, delta_ampere = get_defined_input_current(standard_length, dt)
        i=i_step

    print("** Test_params: {}, dt: {}, start_soc: {}".format(fitted_params, dt, start_soc))
    erg_step = simulate(i, *fitted_params, dt, START_SOC=start_soc)

    r_0 = (erg_step[2] - erg_step[8+int(10/dt)]) / delta_ampere
    print("** Result: R_0: {} Ohm".format(r_0))
    return r_0


if __name__ == "__main__2":
    test_params=[0.003528031966078907, 0.0004940309510702485, 0.0003271800558362171, 117.37811645021696, 1049.3971085596634]
    start_soc= 0.9585036527798383
    dt= 0.5
    inference_Ri(test_params, i=None, dt=dt, start_soc=start_soc)

if __name__ == "__main__":
    raw_datas, file_name, dt, label_current = load_data()
    pool = Pool(10)
    results = [pool.apply_async(fit, args=(raw_data, dt, label_current)) for raw_data in raw_datas]
    pool.close()
    pool.join()

    test_params = results[0]._value[0].x
    START_SOC = results[0]._value[1]

    inference_Ri(test_params, i=None, dt=dt, start_soc=START_SOC)