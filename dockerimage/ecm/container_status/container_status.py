__author__ = "Lukas Merkle"
__copyright__ = "Copyright 2020, 31.07.20"
__email__ = 'lukas.merkle@tum.de'
import psutil
import time
import multiprocessing
from multiprocessing import Process
import requests
import json
import uuid


class Container_Status(Process):

    def __init__(self, container_name="rc_model"):
        super(Container_Status, self).__init__()

        self.container_name =container_name+"_"+uuid.uuid4().__str__()[0:4]

        self.url = ""

    def save_report_to_s3(self):
        pass


    def run(self):
        print("Started Container_Status!")
        x = requests.post(self.url, json=json.dumps({"text": f"Registered {self.container_name}"}))


        while(1):
            loadavg = psutil.getloadavg()
            data = {"container": self.container_name, 'loadavg': loadavg}
            x = requests.post(self.url, json=json.dumps(data))

            time.sleep(5)






if __name__ == "__main__":
    cs = Container_Status()

    cs.start()
    # cs.join()