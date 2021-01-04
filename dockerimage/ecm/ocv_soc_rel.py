import pandas as pd
import numpy as np
import json
import scipy
from scipy.interpolate import interp1d
class NotInRange(Exception):
    pass


''' 
OCV - SOC Relationship
----------------------
    
This class provides data and methods to transform:

- OCV --> SOC
- SOV --> OCV
- CAPACITY --> SOC
- CAPACITY --> OCV

So far only dat for 25°C is available and integrated.

'''
class OCV_SOC_REL():
    def __init__(self, **kwargs):

        if "vehicle" in kwargs:
            self.vehicle = kwargs["vehicle"]
        else:
            raise "You have to specify a vehicle. Possible: 'J1', 'eGolf', 'aCar_eGolf'"

        # Load Settings #################
        settings = json.load(open("ocv_soc_settings.json","r"))

        SERIAL_CELL_BLOCKS_IN_MODULE = settings[self.vehicle]["serial_cell_blocks_in_module"]
        PARALLEL_CELLS_PER_BLOCK = settings[self.vehicle]["parallel_cells_per_block"]
        SERIAL_MODULES = settings[self.vehicle]["serial_modules"]

        # We want to get the cell_capacity
        # we take the one we give in the kwargs (actual one,with aging etc.)
        # if this is not passed:
        # either we have it directly saved in the settings
        # or we have to calculate it from others (system/module)...
        if "aging_factor_capacity" in kwargs:
            aging_factor_capacity = kwargs["aging_factor_capacity"]
        else:
            aging_factor_capacity = 1

        if "CELL_CONTACT_RESISTANCE_DROP_VOLTAGE" in settings[self.vehicle]:
            CELL_CONTACT_RESISTANCE_DROP_VOLTAGE = settings[self.vehicle]["CELL_CONTACT_RESISTANCE_DROP_VOLTAGE"]
        else:
            CELL_CONTACT_RESISTANCE_DROP_VOLTAGE = 0

        if "cell_capacity" in kwargs:
            self.cell_capacity = kwargs["cell_capacity"]
        else:
            if "cell_capacity" in settings[self.vehicle]:
                self.cell_capacity = settings[self.vehicle]["cell_capacity"]
                # self.cell_capacity = 60  # Ah
            if "system_capacity" in settings[self.vehicle]:
                self.cell_capacity = (settings[self.vehicle]["system_capacity"]) / PARALLEL_CELLS_PER_BLOCK

        if "direction" in kwargs:
            self.direction = kwargs["direction"]
        else:
            self.direction = "mean"

        self.module_capacity = self.cell_capacity * PARALLEL_CELLS_PER_BLOCK
        self.SOC = settings[self.vehicle]["soc"]

        # we want to get the ocv modulewise since this is our base value for later
        # If we have it directly in the settings:
        if "ocv_module_charge" in settings[self.vehicle]:
            self.OCV_Module_charge = settings[self.vehicle]["ocv_module_charge"]

        if "ocv_module_discharge" in settings[self.vehicle]:
            self.OCV_Module_discharge = settings[self.vehicle]["ocv_module_discharge"]

        # If we have to calculate it from the systems ocv:
        if "ocv_system_charge" in settings[self.vehicle]:
            self.OCV_Module_charge = [(value/SERIAL_MODULES)+CELL_CONTACT_RESISTANCE_DROP_VOLTAGE for value in settings[self.vehicle]["ocv_system_charge"]]
        if "ocv_system_discharge" in settings[self.vehicle]:
            self.OCV_Module_discharge = [(value/SERIAL_MODULES)+CELL_CONTACT_RESISTANCE_DROP_VOLTAGE for value in settings[self.vehicle]["ocv_system_discharge"]]

        # If we have to calculate it from the cells ocv:
        if "ocv_cell_charge" in settings[self.vehicle]:
            self.OCV_Module_charge = [((value*SERIAL_CELL_BLOCKS_IN_MODULE)-CELL_CONTACT_RESISTANCE_DROP_VOLTAGE) for value in settings[self.vehicle]["ocv_cell_charge"]]
        if "ocv_cell_discharge" in settings[self.vehicle]:
            self.OCV_Module_discharge = [((value*SERIAL_CELL_BLOCKS_IN_MODULE)-CELL_CONTACT_RESISTANCE_DROP_VOLTAGE) for value in settings[self.vehicle]["ocv_cell_discharge"]]


        if self.direction == "charge":
            self.OCV_Module_aggregated = self.OCV_Module_charge
        if self.direction == "discharge":
            self.OCV_Module_aggregated = self.OCV_Module_discharge
        if self.direction == "mean":
            self.OCV_Module_aggregated = 0.5 * (pd.Series(self.OCV_Module_charge) + pd.Series(self.OCV_Module_discharge))


        self.df_OCV_SOC = pd.DataFrame(list(zip(self.SOC, self.OCV_Module_aggregated, self.OCV_Module_aggregated)), columns=["SOC", "OCV_Module", "OCV_Cell"])

        # Convert Module to Cell level by dividing by cell n in module
        self.df_OCV_SOC["OCV_Cell"] = self.df_OCV_SOC["OCV_Module"] / SERIAL_CELL_BLOCKS_IN_MODULE

        # Convert Module to System level
        self.df_OCV_SOC["OCV_System"] = (self.df_OCV_SOC["OCV_Module"] * SERIAL_MODULES) - CELL_CONTACT_RESISTANCE_DROP_VOLTAGE

        # Add Ah-colum
        self.df_OCV_SOC["Capacity_Cell"] = self.df_OCV_SOC["SOC"] * self.cell_capacity * aging_factor_capacity      # Cell capacity in Ah
        self.df_OCV_SOC["Capacity_Module"] = self.df_OCV_SOC["SOC"] * self.module_capacity * aging_factor_capacity  # Cell capacity in Ah
        # print("OCV-REL class initialized...")



    ################ GET SOC From X ##########################################################################################################################
    def get_soc_from_ocv_cell(self, ocv):
        if ocv < min(self.df_OCV_SOC["OCV_Cell"]) or ocv > max(self.df_OCV_SOC["OCV_Cell"]):
            print("Input 'ocv'={} not in legal range of {}V - {}V!".format(ocv, min(self.df_OCV_SOC["OCV_Cell"]), max(self.df_OCV_SOC["OCV_Cell"])))
            raise NotInRange
        # interpolation to get the soc value
        soc_calculated = np.interp(ocv, self.df_OCV_SOC["OCV_Cell"], self.df_OCV_SOC["SOC"]) * 100
        return soc_calculated

    def get_soc_from_ocv_module(self, ocv):
        # check valid input range:
        if ocv < min(self.df_OCV_SOC["OCV_Module"]) or ocv > max(self.df_OCV_SOC["OCV_Module"]):
            print("Input 'ocv' not in legal range of {}V - {}V!".format(min(self.df_OCV_SOC["OCV_Module"]), max(self.self.df_OCV_SOC["OCV_Module"])))
            raise NotInRange
        # interpolation to get the soc value
        soc_calculated = np.interp(ocv, self.df_OCV_SOC["OCV_Module"], self.df_OCV_SOC["SOC"]) * 100
        return soc_calculated

    def get_soc_from_ocv_system(self, ocv):
        # check valid input range:
        if ocv < min(self.df_OCV_SOC["OCV_System"]) or ocv > max(self.df_OCV_SOC["OCV_System"]):
            print("Input 'ocv' not in legal range of {}V - {}V!".format(min(self.df_OCV_SOC["OCV_System"]), max(self.self.df_OCV_SOC["OCV_System"])))
            raise NotInRange
        # interpolation to get the soc value
        soc_calculated = np.interp(ocv, self.df_OCV_SOC["OCV_System"], self.df_OCV_SOC["SOC"]) * 100
        return soc_calculated

    def get_soc_from_capacity_cell(self, capacity):
        if capacity < min(self.df_OCV_SOC["Capacity_Cell"]) or capacity > max(self.df_OCV_SOC["Capacity_Cell"]):
            print("Input 'capacity' {} not in legal range of {}Ah - {}Ah!".format(capacity,min(self.df_OCV_SOC["Capacity_Cell"]), max(self.df_OCV_SOC["Capacity_Cell"])))
            raise NotInRange
        # interpolation to get the soc value
        soc_calculated = np.interp(capacity, self.df_OCV_SOC["Capacity_Cell"], self.df_OCV_SOC["SOC"]) * 100
        return soc_calculated

    def get_soc_from_capacity_module(self, capacity):
        # check valid input range:
        if capacity < min(self.df_OCV_SOC["Capacity_Module"]) or capacity > max(self.df_OCV_SOC["Capacity_Module"]):
            print("Input 'capacity' {} not in legal range of {}Ah - {}Ah!".format(capacity,min(self.df_OCV_SOC["Capacity_Module"]), max(self.df_OCV_SOC["Capacity_Module"])))
            raise NotInRange
        # interpolation to get the soc value
        soc_calculated = np.interp(capacity, self.df_OCV_SOC["Capacity_Module"], self.df_OCV_SOC["SOC"]) * 100
        return soc_calculated

    def get_soc_from_capacity_system(self, capacity):
        # check valid input range:
        if capacity < min(self.df_OCV_SOC["Capacity_Module"]) or capacity > max(self.df_OCV_SOC["Capacity_Module"]):
            print("Input 'capacity' {} not in legal range of {}Ah - {}Ah!".format(capacity,min(self.df_OCV_SOC["Capacity_Module"]), max(self.df_OCV_SOC["Capacity_Module"])))
            raise NotInRange
        # interpolation to get the soc value
        soc_calculated = np.interp(capacity, self.df_OCV_SOC["Capacity_Module"], self.df_OCV_SOC["SOC"]) * 100
        return soc_calculated



    ################ GET OCV From X ##########################################################################################################################
    def get_ocv_from_soc_cell(self, soc):
        if soc < min(self.df_OCV_SOC["SOC"]) or soc > max(self.df_OCV_SOC["SOC"]):
            print("Input 'soc' not in legal range of {}% - {}%!".format(min(self.df_OCV_SOC["SOC"]), max(self.df_OCV_SOC["SOC"])))
            raise NotInRange
        # interpolation to get the ocv value
        ocv = np.interp(soc, self.df_OCV_SOC["SOC"], self.df_OCV_SOC["OCV_Cell"])
        return ocv

    def get_ocv_from_soc_module(self, soc):
        if soc < min(self.df_OCV_SOC["SOC"]) or soc > max(self.df_OCV_SOC["SOC"]):
            print("Input 'soc' not in legal range of {}% - {}%!".format(min(self.df_OCV_SOC["SOC"]), max(self.df_OCV_SOC["SOC"])))
            raise NotInRange
        # interpolation to get the ocv value
        ocv = np.interp(soc, self.df_OCV_SOC["SOC"], self.df_OCV_SOC["OCV_Module"])
        return ocv

    def get_ocv_from_soc_system(self, soc):
        if soc < min(self.df_OCV_SOC["SOC"]) or soc > max(self.df_OCV_SOC["SOC"]):
            print(f"Input 'soc': {soc} not in legal range of {min(self.df_OCV_SOC['SOC'])}% - {max(self.df_OCV_SOC['SOC'])}%!")
            raise NotInRange
        # interpolation to get the ocv value
        ocv = np.interp(soc, self.df_OCV_SOC["SOC"], self.df_OCV_SOC["OCV_System"])
        return ocv


    def get_ocv_from_capacity_cell(self, capacity):
        if capacity < min(self.df_OCV_SOC["Capacity_Cell"]) or capacity > max(self.df_OCV_SOC["Capacity_Cell"]):
            print("Input 'capacity {} not in legal range of {}Ah - {}Ah!".format(capacity, min(self.df_OCV_SOC["Capacity_Cell"]), max(self.df_OCV_SOC["Capacity_Cell"])))
            raise NotInRange
        # interpolation to get the ocv value

        # ocv_smooth = interp1d(self.df_OCV_SOC["Capacity_Cell"], self.df_OCV_SOC["OCV_Cell"], kind="quadratic")
        # ocv = ocv_smooth(capacity)
        ocv = np.interp(capacity, self.df_OCV_SOC["Capacity_Cell"], self.df_OCV_SOC["OCV_Cell"])
        return ocv

    def get_ocv_from_capacity_module(self, capacity):
        if capacity < min(self.df_OCV_SOC["Capacity_Module"]) or capacity > max(self.df_OCV_SOC["Capacity_Module"]):
            print("Input 'capacity {} not in legal range of {}Ah - {}Ah!".format(capacity, min(self.df_OCV_SOC["Capacity_Module"]), max(self.df_OCV_SOC["Capacity_Module"])))
            raise NotInRange
        # interpolation to get the ocv value
        ocv = np.interp(capacity, self.df_OCV_SOC["Capacity_Module"], self.df_OCV_SOC["OCV_Module"])
        return ocv

    def get_ocv_from_capacity_system(self, capacity):
        if capacity < min(self.df_OCV_SOC["Capacity_Module"]) or capacity > max(self.df_OCV_SOC["Capacity_Module"]):
            print("Input 'capacity {} not in legal range of {}Ah - {}Ah!".format(capacity, min(self.df_OCV_SOC["Capacity_Module"]), max(self.df_OCV_SOC["Capacity_Module"])))
            raise NotInRange
        # interpolation to get the ocv value
        ocv = np.interp(capacity, self.df_OCV_SOC["Capacity_Module"], self.df_OCV_SOC["OCV_System"])
        return ocv


    ################ GET Capacity From X ##########################################################################################################################
    def get_capacity_from_ocv_system(self, ocv):
        # check valid input range:
        if ocv < min(self.df_OCV_SOC["OCV_System"]) or ocv > max(self.df_OCV_SOC["OCV_System"]):
            print("Input 'ocv' not in legal range of {}V - {}V!".format(min(self.df_OCV_SOC["OCV_System"]),max(self.self.df_OCV_SOC["OCV_System"])))
            raise NotInRange
        # interpolation to get the c value
        c = np.interp(ocv, self.df_OCV_SOC["OCV_System"], self.df_OCV_SOC["Capacity_Module"])
        return c

#
# c = OCV_SOC_REL(direction="charge", vehicle="eGolf")
#
# print(c.get_soc_from_ocv_cell(4.1))
# print(c.get_ocv_from_soc_cell(0.20))
# print(c.get_ocv_from_soc_module(0.20))
# print(c.get_ocv_from_capacity_cell(23))
# print(c.get_soc_from_capacity_cell(23))
# print(c.get_soc_from_capacity_system(23))
#
# print("-------------- discharge-------------")
# c = OCV_SOC_REL(direction="discharge")
# print(c.get_capacity_from_ocv_system(815))
# print(c.get_soc_from_capacity_system(60))