import numpy as np
class ReplayBuffer():
    def __init__(self):
        self.replay_buffer = []
    
    def add(self, tup):
        self.replay_buffer.append(tup)

    def save(self):
        arr = np.array(self.replay_buffer)
        np.save("ReplayBuffer", arr)
        