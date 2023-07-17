import pickle as pkl
import torch
from model import ActionAbstractionModel
from train import make_token_sequences

if __name__=="__main__":

    data = pkl.load(open("trajectories.pkl","rb"))
    token_sequences = make_token_sequences(data)

    model = ActionAbstractionModel(2, 2)
    model.load_state_dict(torch.load("model.pt"))

    for i in range(1):
        seq = model()
        print(seq.components)
        print(seq.tokens)
        
        
    print(model)
