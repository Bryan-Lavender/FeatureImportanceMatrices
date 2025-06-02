import numpy as np
def half_split(x):
  
    sorted_x = np.sort(x)
    buckets = []
    for i in range(1,len(x)):
        index_1 = i - 1
        index_2 = i
        buckets.append((sorted_x[index_2]+sorted_x[index_1])/2)
    
    return buckets
