from hmmlearn import hmm
import torch
import torch.nn as nn
from model import Sequence
import numpy as np
from train import get_fake_token_sequences
import pickle as pkl
from train import make_token_sequences
class HMMModel(nn.Module):

    def __init__(self, n_low_level_actions,max_n_abstract_actions) -> None:
            
            super().__init__()
            self.n_low_level_actions = n_low_level_actions
            self.max_n_abstract_actions = max_n_abstract_actions

            self.n_abstract_actions_params = nn.Parameter(torch.randn(self.max_n_abstract_actions))
    
    def forward(self, data,lengths, n_components):
        
        model = hmm.MultinomialHMM(n_components=n_components, n_iter=100)

        model.fit(data,lengths=lengths)
        
        # model_logprob = model.score(data,lengths=lengths)
        # model_prob = np.exp(model_logprob)
        #print(model_prob, (1+n_components.item())/self.max_n_abstract_actions)
        return model#,log_prob



def loss(model, data, lengths):
    hmm_model, log_prob = model(data,lengths)

    try:
        reward = np.exp(hmm_model.score(data,lengths=lengths)) -1* hmm_model.n_components/model.max_n_abstract_actions
    except ValueError:
        
        reward=-1
        
    return -log_prob*reward


def onehot_token(seq,max_tokens):
    onehot = np.zeros((len(seq),max_tokens))
    for i,token in enumerate(seq):
        onehot[i,int(token)] = 1
    return onehot

def from_onehot(seq):
    return np.argmax(seq,axis=1)


def label_sequences_with_hmm(data, max_abstract_states):
    
    token_seq = make_token_sequences(data)
    token_seq = [list(t) for t in token_seq]
    lengths= [len(t) for t in token_seq]
    
    X = np.concatenate(token_seq).reshape(-1,1)
    X = np.array(X,dtype=np.int32)
    #print(X,lengths)

    labels=  {}
    
    for i in range(2,max_abstract_states+1):
        model = hmm.CategoricalHMM(n_components=i, n_iter=100)
        model.fit(X,lengths=lengths)
        
        state_labels = []
        for seq in token_seq:
            x = np.array(seq, dtype = np.int32).reshape(-1,1)
            _,z = model.decode(x)
            state_labels.append(z)


        labels[i] = state_labels

    return labels

def main():


    max_abstract_states= 7
    #token_seq = get_fake_token_sequences(10, max_tokens=max_abstract_states)

    data = pkl.load(open("trajectories.pkl","rb"))
    label_sequences_with_hmm(data, max_abstract_states)
    exit()
    token_seq = make_token_sequences(data)
    token_seq = [list(t) for t in token_seq]
    onehot_token_seq = [onehot_token(t, max_abstract_states) for t in token_seq]
    lengths= [len(t) for t in token_seq]
    
    X = np.concatenate(token_seq).reshape(-1,1)
    X = np.array(X,dtype=np.int32)
    #print(X,lengths)

    
    for i in range(2,max_abstract_states+1):
        model = hmm.CategoricalHMM(n_components=i, n_iter=100)
        model.fit(X,lengths=lengths)
        
        print(i, model.score(X,lengths=lengths))
        print(token_seq[0])
        print(model.decode(X[:lengths[0]]))
        x,z = model.sample(lengths[0])
        print(x.flatten(),z)
    


if __name__ == "__main__":
    main()
    exit()
    from train import get_fake_token_sequences
    
    token_seq = get_fake_token_sequences(10)
    token_seq = [onehot_token(t, 3) for t in token_seq]
    lengths= [len(t) for t in token_seq]
    
    X = np.concatenate(token_seq)
    X = np.array(X,dtype=np.int32)
    print(X,lengths)
    model = HMMModel(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(100):
        opt.zero_grad()
        l = loss(model, X, lengths)
        l.backward()
        opt.step()

    print(model.n_abstract_actions_params)
    hmm_model = model(X,lengths)[0]
    print(hmm_model.sample(10))

