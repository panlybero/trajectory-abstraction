import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
from loss import compute_loss, Discriminator, compute_discriminator_loss
from model import ActionAbstractionModel
import tqdm
def make_token_sequences(trajectories):
    token_sequences = []
    for trajectory in trajectories:
        token_sequence = []
        for step in trajectory:
            
            token_sequence.append(str(step[1])) # action
        token_sequences.append("".join(token_sequence))
    return token_sequences
      


def get_fake_token_sequences(n, max_tokens=3):
    sequences = []
    for i in range(n):
        seq = [[j]*np.random.randint(1,6) for j in range(max_tokens)]
        seq = [item for sublist in seq for item in sublist]
        sequences.append("".join([str(s) for s in seq]))
    return sequences


if __name__=="__main__":

    data = pkl.load(open("trajectories.pkl","rb"))
    token_sequences = get_fake_token_sequences(1000)#make_token_sequences(data)
    n_actions= 2
    model = ActionAbstractionModel(n_actions, n_actions)
    discriminator = Discriminator(n_actions)
    model.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in tqdm.tqdm(range(1000)):
        optimizer.zero_grad()
        sample_seqs = [model() for _ in range(100)]
        #loss,mean_score = compute_discriminator_loss(discriminator, sample_seqs, token_sequences, n_actions)
        loss,mean_score = compute_loss(sample_seqs, token_sequences, n_actions)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss}", "Avg_score",mean_score)

        if i % 10 == 0:
            torch.save(model.state_dict(), "model.pt")
            print(model().tokens)
            print(model().tokens)
            print(model().tokens)

    print(model())


    