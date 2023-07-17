import numpy as np

from itertools import combinations
import numpy as np
from torch.nn import functional as F
import torch

def entropy(p):
    return -np.sum(p*np.log(p))

def score_partition(partition, pairs, n_actions=4):
    #gather all pairs with the same partition
    partitioned_samples ={0:[],1:[]}
    for i,p in enumerate(pairs):
        partitioned_samples[partition[i]].append(p)
    
    #compute action distribution for each partition

    action_dists = {i:compute_action_distribution(partitioned_samples[i], n_actions=n_actions) for i in partitioned_samples}
    #compute cross entropy between action distributions
    score = F.cross_entropy(torch.tensor(action_dists[0]).reshape(1,-1),torch.tensor(action_dists[1]).reshape(1,-1)).item() + F.cross_entropy(torch.tensor(action_dists[1]).reshape(1,-1),torch.tensor(action_dists[0]).reshape(1,-1)).item()
    if np.isnan(score):
        score = -100
    return score
   # print(actiond)
    #print(action_dists)
    entropies = [entropy(action_dists[i]) for i in action_dists]
    if any([np.isnan(e) for e in entropies]):
        return -100
    else:
        return -sum(entropies)
    print(score)

    return score



def maximize_cross_entropy(pairs, n_actions=4):

    initial_assignments = np.random.randint(2,size=len(pairs))
    print(initial_assignments)
    score = score_partition(initial_assignments, pairs, n_actions)
    print('initial score', score)
    action_scores = np.zeros(len(pairs))

    curr_assignments = initial_assignments.copy()
    while True:
        #print("here")
        for i in range(len(curr_assignments)):
            assignments = curr_assignments.copy()
            assignments[i] = 1 - assignments[i]
            new_score = score_partition(assignments, pairs, n_actions)
            action_scores[i] = new_score
        
        #print(action_scores)
        best_action = np.argmax(action_scores)
       
        #print(action_scores[best_action], score)

        new_score = action_scores[best_action]
        if new_score - score<= 0:
            break
        score = new_score
        curr_assignments = curr_assignments.copy()
        curr_assignments[best_action] = 1 - curr_assignments[best_action]
        print('assignment',curr_assignments,'curr score', score)
        

        
    #print(curr_assignments)
    partitions = {0:[],1:[]}
    for i,p in enumerate(pairs):
        partitions[curr_assignments[i]].append(p)
    
    return partitions


def compute_action_distribution(pairs, n_actions=4):
    actions = [pair[1] for pair in pairs]
    unique_actions = np.unique(actions)
    counts = np.bincount(actions)
    action_counts = np.zeros(n_actions)
    for i,c in enumerate(counts):
        action_counts[i] = c
    
    action_dist = action_counts / len(pairs)
    return action_dist

if __name__=='__main__':

    samples = list(range(20))
    values = np.random.randint(0,4,20)
    pairs = list(zip(samples,values))
    dist = compute_action_distribution(pairs, n_actions=4)
    print(dist)
    partitions= maximize_cross_entropy(pairs, n_actions=4)
    print(partitions)

    dist1 = compute_action_distribution(partitions[0], n_actions=4)
    dist2= compute_action_distribution(partitions[1], n_actions=4)
    print(dist1,dist2)
    pass
