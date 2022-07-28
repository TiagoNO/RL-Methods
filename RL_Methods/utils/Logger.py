from genericpath import isfile
import os
import json 

class Logger:

    def __init__(self, log_filename) -> None:        
        self.directory = log_filename[:log_filename.rfind("/")+1]
        self.filename = log_filename[log_filename.rfind("/")+1:] + ".json"

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        if os.path.isfile(self.directory + self.filename):
            self.load()
        else:
            self.data = {}

    def log(self, name, value):
        name_keys = name.split("/")
        values = self.data
        # value = round(value, 8)
        for key in name_keys:
            if not key in values:
                values[key] = {}
            values = values[key]

        if not 'data' in values:
            values['data'] = []
        values['data'].append(value)

    def dump(self):
        with open(self.directory + self.filename, 'w') as f:
            json.dump(self.data, f)

    def load(self):
        with open(self.directory + self.filename, 'r') as f:
            self.data = json.load(f)

