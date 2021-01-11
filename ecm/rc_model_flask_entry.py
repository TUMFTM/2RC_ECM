__author__ = "Lukas Merkle"
__copyright__ = "Copyright 2020, 31.07.20"
__email__ = 'lukas.merkle@tum.de'

import model_fitting_2rc_system
import model_fitting_2rc_automated_soc as model_fitting_2rc_cell
from flask import Flask, request, Response
app = Flask(__name__)
import sys
import time
import  pandas as pd
import json
import matplotlib.pyplot as plt
import pickle
'''
This file serves as an entry into the flask-calls. API calls are answered here by the respective functions.
Layer between the API and the RC-Model for system and cell

'''

def current_well_formed(current):
    i0 = current[0]
    step1 = False
    step2 = False
    i1=None

    for i in current:
        if (i - i0) > 10:
            step1 = True
            i1 = i

        if step1 and (i - i1) < 10:
            step2 = True

    if step1 and step2:
        return True
    else:
        return False

def estimate_twin_type(twin_type_string):
    # Cell Level
    if "cell" in twin_type_string:
        return "CELL"

    # System Level
    if "cell" not in twin_type_string:
        return "SYSTEM"

@app.route("/rc_model/fit", methods=['POST'])
def fit():

    input_data_json  = request.json

    if "current" not in input_data_json or "voltage" not in input_data_json or "timestamps" not in input_data_json or "temperature" not in input_data_json:
        resp = Response(status=500, mimetype="application/text")
        resp.set_data("Input parameters malformed!")
        return resp

    current = input_data_json["current"]
    voltage = input_data_json["voltage"]
    timestamps = input_data_json["timestamps"]
    fit_level=input_data_json["fit_level"]

    # generate input data
    df_measure = pd.DataFrame({"current": current, "voltage": voltage,"time": timestamps})

    # cycle temperature
    temperature = input_data_json["temperature"]



    # Prepare input data
    df_measure.index = pd.to_datetime(df_measure["time"], unit="s")
    # Resample
    df_measure = df_measure.dropna(axis="index")
    df_measure = df_measure.resample("200ms").mean()
    dt = 0.2


    # print("** Input Dataframe Head: --------------------------------")
    print(df_measure.head())
    print("---------------------------------------------------------")


    # Check if there is enough variability in the current
    THRESHOLD_CURRENT_STD = 4
    current_std = df_measure["current"].std()
    current_min = df_measure["current"].min()
    current_max = df_measure["current"].max()
    current_range = abs(current_max-current_min)


    if current_range < 10:
        resp = Response(status=500, mimetype="application/text")
        resp.set_data("Current range too small! (It is below 10 A change)")
        return resp

    if not current_well_formed(df_measure["current"]):
        resp = Response(status=500, mimetype="application/text")
        resp.set_data("Current signal malformed!")
        return resp


    # No we need to decide if we call the cell-level or the module level or the system level:
    #######################################################################################
    # Serve Cell Level
    #######################################################################################
    if fit_level == "CELL":
        # Do the fitting #params; raw_data, dt, label_current
        fitted_params, start_soc, rmse = model_fitting_2rc_cell.fit(raw_data=df_measure, dt=dt, label_current="label_current")


    #######################################################################################
    # Serve System Level
    #######################################################################################
    if fit_level == "SYSTEM":
        # Do the fitting
        fitted_params, rmse = model_fitting_2rc_system.fit(df_measure=df_measure, file_name=time.ctime(), dt=dt)

    #######################################################################################
    # Save the params to the db and return
    #######################################################################################
    # Save the new params to database using dt_access
    # get the proper path to set the params
    ECM_params_value_to_set =  {
                    "R0": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[0]
                    },
                    "R1": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[1]
                    },
                    "C1": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[2]
                    },
                    "R2": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[3]
                    },
                    "C2": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[4]
                    },
                    "soc": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[5]
                    },
                    "capacity": {
                        "type": {
                            "float": {}
                        },
                        "value": fitted_params.x[6]
                    },
                    "temperature": {
                        "type": {
                            "float": {}
                        },
                        "value": temperature
                    },
                    "dt":{
                        "type": {
                            "float": {}
                        },
                        "value": dt
                    },
                    "timestamp": {
                        "type": {
                            "float": {}
                        },
                        "value": time.time()
                    },
                    "rmse": {
                        "type": {
                            "float": {}
                        },
                        "value": rmse
                    },
                    "fit_level": fit_level
                }


    
    # save current params
    pickle.dump(ECM_params_value_to_set, open("ECM_params_value_to_set.p", "wb"))
    resp = Response(status=200, mimetype='application/json')
    resp.set_data(json.dumps(ECM_params_value_to_set))
    return resp



@app.route("/rc_model/infer", methods=['POST', 'GET'])
def infer(**kwargs):
    print("________________________________________________________________________________")
    print("** New Request -----------------------------------------------------------------")
    print("Infer called...")

    input_data_json  = request.json

    ECM_params_value_to_set = pickle.load(open("ECM_params_value_to_set.p", "rb"))


    try:
        start_soc = ECM_params_value_to_set["soc"]["value"]
        temperature = ECM_params_value_to_set["temperature"]["value"]
        dt = ECM_params_value_to_set["dt"]["value"]
        fitted_params = [ECM_params_value_to_set["R0"]["value"],
                         ECM_params_value_to_set["R1"]["value"],
                         ECM_params_value_to_set["C1"]["value"],
                         ECM_params_value_to_set["R2"]["value"],
                         ECM_params_value_to_set["C2"]["value"],
                         ECM_params_value_to_set["soc"]["value"],
                         ECM_params_value_to_set["capacity"]["value"]]

        fit_level = ECM_params_value_to_set["fit_level"]
    except:
        resp = Response(status=500, mimetype='application/text')
        resp.set_data("Input data (also the pickle created by fit()) malformed!")
        return resp

    

    # Cell Level
    if fit_level == "CELL":
        RMSE_THRESHOLD_INFER = 0.01

    # System Level
    if fit_level == "SYSTEM":
        RMSE_THRESHOLD_INFER = 1

   


    # Call the inference function
    if fit_level == "CELL":
        r_0_10s = model_fitting_2rc_cell.inference_Ri(fitted_params=fitted_params, i=None, dt=dt, start_soc=start_soc)

    if fit_level == "SYSTEM":
        r_0_10s = model_fitting_2rc_system.inference_Ri(fitted_params=fitted_params, i=None, dt=dt, start_soc=start_soc)

    return_dict={
        "R_0": r_0_10s,
        "soc": start_soc,
        "temperature": temperature,
    }

    resp = Response(status=200, mimetype='application/json')
    resp.set_data(json.dumps(return_dict))
    return resp


@app.route("/rc_model/healthcheck", methods=['GET'])
def healthchecker():
    return "0"



if __name__ == "__main__":
    # get_resources_system()
    app.run(host="0.0.0.0", processes=8, threaded=False)