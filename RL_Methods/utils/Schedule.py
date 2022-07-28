from abc import abstractclassmethod
import numpy as np

class Schedule:

    def __init__(self, initial_value, delta=0, final_value=None) -> None:
        self.initial_value = initial_value
        self.final_value = final_value
        self.delta = delta
        self.cur_value = self.initial_value

    def update(self):
        pass

    def reset(self):
        self.cur_value = self.initial_value

    def get(self):
        return self.cur_value        

class Constant(Schedule):
    def __init__(self, initial_value) -> None:
        super().__init__(initial_value, 0, 0)


class LinearSchedule (Schedule):

    def __init__(self, initial_value, delta, final_value=None) -> None:
        super().__init__(initial_value, delta, final_value)

    def update(self):
        if self.final_value is None:
            self.cur_value += self.delta

        elif self.delta > 0:
            self.cur_value = min(self.cur_value + self.delta, self.final_value)

        elif self.delta < 0:
            self.cur_value = max(self.cur_value + self.delta, self.final_value)
