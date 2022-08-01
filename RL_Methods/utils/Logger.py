from genericpath import isfile
import os
import json 

class Logger:

    def __init__(self, log_filename) -> None:        
        self.directory = log_filename[:log_filename.rfind("/")+1]
        self.filename = log_filename[log_filename.rfind("/")+1:] + ".json"

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        self.data = {}

    def log(self, name, value):
        name_keys = name.split("/")
        values = self.data
        for key in name_keys:
            if not key in values:
                values[key] = {}
            values = values[key]

        if not 'data' in values:
            values['data'] = []
        values['data'].append(value)

    def merge(self, values, target):
        print(values, target)
        if 'data' in target:
            if 'data' in values:
                values['data'].extend(target['data'])
            else:
                values['data'] = target['data']
            return

        for key in target.keys():
            if not key in values:
                values[key] = {}
            self.merge(values[key], target[key])
        return values

    def dump(self):
        loaded_data = self.retrieve()
        self.data = self.merge(loaded_data, self.data)
        with open(self.directory + self.filename, 'w') as f:
            json.dump(self.data, f)
        self.data.clear()

    def retrieve(self, filename=None):
        if filename is None:
            filename = self.directory + self.filename

        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except:
            return {}

    def load(self, filename=None):
        self.data = self.retrieve(filename)

# logger = Logger("experiments/logger/teste")
# logger.log("parameters/alpha", 10)
# logger.log("parameters/beta", 1)
# logger.dump()
# input()
# logger.log("parameters/alpha", 11)
# logger.log("parameters/alpha", 12)
# logger.log("parameters/alpha", 13)
# logger.log("parameters/gamma", 0.1)
# logger.dump()
# input()
# del logger

# logger = Logger("experiments/logger/teste")
# logger.log("parameters/gamma", 0.01)
# logger.log("parameters/gamma", 0.001)
# logger.dump()
# input()
