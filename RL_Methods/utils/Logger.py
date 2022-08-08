import os
import time
class Logger:

    def __init__(self, log_filename) -> None:        
        self.directory = log_filename[:log_filename.rfind("/")+1]
        self.filename = log_filename[log_filename.rfind("/")+1:] + ".log"

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        self.data = {}
        self.tabs_exp = 55

    def log(self, name, value):
        if name in self.data:
            self.data[name].append(value)
        else:
            self.data[name] = [value]

    def update(self, name, value):
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
        with open(self.directory + self.filename, 'a') as f:
            for key in self.data.keys():
                f.write("{}:".format(key))
                for v in self.data[key]:
                    f.write("{} ".format(v))
                f.write("\n")
        self.clear()

    def retrieve(self, filename=None):
        if filename is None:
            filename = self.directory + self.filename
        try:
            retrieved_data = {}
            with open(filename, 'r') as f:
                for lines in f.readlines():
                    key, list_values = lines.split(':')
                    data = []
                    for v in list_values.split(" ")[:-1]:
                        data.append(float(v))

                    if key in retrieved_data:
                        retrieved_data[key].extend(data)
                    else:
                        retrieved_data[key] = data
            return retrieved_data
        except Exception as e:
            # print(e)
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