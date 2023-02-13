import os
from enum import Enum

class LogLevel:
    OFF = 0
    INFO = 1
    DEBUG = 2
    ALL = 3

class Logger:

    def __init__(self, log_filename : str = None) -> None:        

        if log_filename is None:
            self.dump_to_file = False
        else:
            self.directory = log_filename[:log_filename.rfind("/")+1]
            self.filename = log_filename[log_filename.rfind("/")+1:] + ".log"
            self.dump_to_file = True
            if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

        self.data = {}
        self.tabs_exp = 55
        self.level = LogLevel.INFO

    def setLevel(self, level : LogLevel):
        self.level = level

    def log(self, level : LogLevel, name : str, value):
        print(self.level, level)
        if self.level < level:
            return

        if name in self.data:
            self.data[name].append(value)
        else:
            self.data[name] = [value]

    def update(self, level : LogLevel, name, value):
        if self.level < level:
            return

        self.data[name] = [value]

    def merge(self, values, target):
        for key in target.keys():
            if key in values:
                values[key].extend(target[key])
            else:
                values[key] = target[key]
        return values

    def print(self):
        print("\n\n|" + "=" * (self.tabs_exp-1) + "|")
        for key in self.data.keys():
            name = key[key.rfind("/")+1:]
            value = self.data[key][-1]
            print("|{}: {}\t|".format(name, value).expandtabs(self.tabs_exp))
        print("|" + "=" * (self.tabs_exp-1) + "|")

    def dump(self):
        if self.dump_to_file:
            f = open(self.directory + self.filename, 'a')
            for key in self.data.keys():
                f.write("{}:".format(key))
                for v in self.data[key]:
                    f.write("{} ".format(v))
                f.write("\n")
            f.close()
        self.clear()

    def retrieve(self, filename=None):
        if filename is None:
            filename = self.directory + self.filename
        try:
            retrieved_data = {}
            f = open(filename, 'r')
            for lines in f.readlines():
                key, list_values = lines.split(':')
                data = []
                for v in list_values.split(" ")[:-1]:
                    data.append(float(v))

                if key in retrieved_data:
                    retrieved_data[key].extend(data)
                else:
                    retrieved_data[key] = data
            f.close()
            return retrieved_data
        except Exception as e:
            print(e)
            return {}

    def clear(self):
        self.data.clear()

    def load(self, filename=None):
        self.data = self.retrieve(filename)
        if len(self.data.keys()) > 0:
            return True 
        return False


# logger = Logger("experiments/logger/teste")
# logger.log("parameters/alpha", 10)
# logger.log("parameters/beta", 1)
# logger.dump()
# input()

# for i in range(1000000):
#     # print(i)
#     logger.log("parameters/alpha", i)
#     if i % 100 == 0:
#         begin = time.time()
#         logger.dump()
#         print("{}: {}".format(i, time.time()-begin))
# logger.log("parameters/gamma", 0.1) 
# logger.dump()
# input()
# del logger

# logger = Logger("experiments/logger/teste")
# for i in range(100):
#     logger.log("parameters/gamma", 0.01)
#     logger.log("parameters/gamma", 0.001)
# logger.dump()
# input()

# logger.load()
# logger.print()