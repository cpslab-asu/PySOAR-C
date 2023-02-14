import time

class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *arg):
        self.count = self.count + 1
        
        hybrid_dist = self.func(*arg)
        
        return hybrid_dist