from hmmmodel import label_sequences_with_hmm
import pickle as pkl
import numpy as np
from train import make_token_sequences
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from state_description import describe_array_state,get_truth_fractions_per_group
from sklearn.model_selection import cross_val_score

def get_states(data):
    states = []
    for ep in data:
        for d in ep:
            
            state,_,_ = d
            distances = state['distance_to_wood']
            inventory = state['inventory']
            combined = np.concatenate((distances,inventory))
            states.append(combined)

    states=  np.array(states)
    print(states.shape)
    return states



def perform_cross_validation(X, y, model):
    scores = cross_val_score(model, X, y, cv=5)  # Change cv value as desired (default is 5-fold)
    return scores

if __name__ == "__main__":
    max_abstraction = 8
    data = pkl.load(open("trajectories.pkl","rb"))
    token_seq = make_token_sequences(data)
    labels = label_sequences_with_hmm(data, max_abstraction)
    X = get_states(data)

    for abstraction_level in range(2,max_abstraction):
        
        y = labels[abstraction_level]
        y = np.concatenate(y)
        clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)#SVC(gamma='auto')
        #clf = KNeighborsClassifier()
        scores = perform_cross_validation(X,y,clf)
        print(abstraction_level, scores.mean(), scores.std())
        get_truth_fractions_per_group(X,y)



    

