import torch
import torch.nn as nn
import editdistance

class Discriminator(nn.Module):
    def __init__(self, n_tokens) -> None:
        super().__init__()

        self.n_tokens = n_tokens
        self.emb_layer = nn.Embedding(n_tokens+1, 16)

        self.lstm = nn.LSTM(16, 16, batch_first=True)
        self.linear = nn.Linear(16, 1)

        self.token_to_idx = {str(i):i for i in range(n_tokens)}

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.update_every = 5
        self.counter = 0

    
    def forward(self, x):
        x = self.emb_layer(x.long())
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def _tokenize(self,tokens):
        return torch.tensor([self.token_to_idx[token] for token in tokens])

    def update(self,loss):
        self.counter += 1
        if self.counter % self.update_every == 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.counter = 0



def compute_discriminator_loss(discriminator, seqs, true_sequences, max_n_abstract_actions):

    #compute loss for generated sequences
    tokenized_generated_seqs = [discriminator._tokenize(seq.tokens) for seq in seqs]
    logprobs = torch.stack([seq.logprob for seq in seqs])
    tokenized_generated_seqs = nn.utils.rnn.pad_sequence(tokenized_generated_seqs, batch_first=True)
    disc_scores = discriminator(tokenized_generated_seqs)
    generated_loss = torch.mean(disc_scores)

    #compute loss for real sequences
    tokenized_real_seqs = [discriminator._tokenize(seq) for seq in true_sequences]
    tokenized_real_seqs = nn.utils.rnn.pad_sequence(tokenized_real_seqs, batch_first=True, padding_value=discriminator.n_tokens)
    real_loss = torch.mean(discriminator(tokenized_real_seqs))

    
    discriminator_loss = generated_loss - real_loss
    generator_loss = (-disc_scores.detach() * logprobs).mean()

    discriminator.update(discriminator_loss)
    #discriminator_loss.backward()
    #discriminator.optimizer.step()


    return generator_loss, {"Avg real score":real_loss.item(), "Avg generated score":generated_loss.item()}
        


def score_sample(seq_sample,true_sequences, max_n_abstract_actions):
    
    distances = [editdistance.eval(seq_sample.tokens, seq)*1.0 for seq in true_sequences]

    mean_distance = torch.mean(torch.tensor(distances))
    
    n_abstracts_used = seq_sample.n_abstract_actions

    score = mean_distance + n_abstracts_used
    return score

def compute_loss(sample_seqs,true_sequences, max_n_abstract_actions):
    scores = []
    logprobs = []
    for seq in sample_seqs:
        scores.append(score_sample(seq,true_sequences, max_n_abstract_actions))
        logprobs.append(seq.logprob)

    scores = torch.tensor(scores).to(logprobs[0].device)
    logprobs = torch.stack(logprobs)

    loss_proxy = torch.mean(scores*logprobs)

    return loss_proxy, torch.mean(scores)




    