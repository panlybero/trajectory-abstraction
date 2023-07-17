import pickle as pkl
import numpy as np
from state_description import describe_state
from sklearn.decomposition import PCA

if __name__=='__main__':
    data = pkl.load(open("trajectories.pkl","rb"))
    
    state_descriptions = []
    for ep in data:
        for d in ep:
            state_descriptions.append(describe_state(d[0]))


