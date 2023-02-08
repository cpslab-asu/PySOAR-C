import time

class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0
        self.point_history = []
        self.simultation_time = []
        self.mode_count = 0
        self.modes = []

    def __call__(self, sample, mode, region):
        self.count = self.count + 1
        sim_time_start = time.perf_counter()
        rob_val = self.func.get_robustness(sample)
        hybrid_dist = self.func.get_hybrid_distance(sample)
        time_elapsed = time.perf_counter() - sim_time_start
        self.simultation_time.append(time_elapsed)

        if mode == 1:
            self.mode_count += 1

        self.modes.append(self.mode_count)
        self.point_history.append([self.count, sample, self.mode_count, mode, region, rob_val, hybrid_dist])
        return rob_val, hybrid_dist