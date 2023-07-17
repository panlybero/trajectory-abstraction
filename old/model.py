import torch
import torch.nn as nn
import pickle as pkl


class Sequence:
    def __init__(self):

        self.components = []
        self.full_sequence =[]
        self.abstract_actions_used =set()
        self.total_logprob = 0

    def add_component(self,component):
        self.components.append(component)
        self.full_sequence += component['tokens']
        self.abstract_actions_used.add(component['AbstractActionName'])
        self.total_logprob += component["logprob"]
    
    @property
    def length(self):
        return len(self.full_sequence)

    @property
    def n_abstract_actions(self):
        return len(self.abstract_actions_used)
    
    @property
    def logprob(self):
        return self.total_logprob

    @property
    def tokens(self):
        """Returns the sequence of tokens"""

        ts = "".join([str(token) for token in self.full_sequence])
        return ts

class AbstractActionv2(nn.Module):
    def __init__(self, n_low_level_actions,n_transitions, name = 'op') -> None:
        super().__init__()
        self.name=  name
        self.n_low_level_actions = n_low_level_actions
        self.n_action_options = n_low_level_actions # for stop
        self.n_transitions = n_transitions
        self.n_transition_options = n_transitions # for stop


        #action parameters
        self.action_params = nn.Parameter(torch.randn(self.n_action_options))
        #transition parameters
        self.transition_params = nn.Parameter(torch.randn(self.n_transition_options))

        self.change_state = nn.Parameter(torch.randn(1))

    def forward(self):
        
        curr_token = None
        total_logprob = 0
        sample = {'tokens':[], 'next_state':None, 'logprob':None,"AbstractActionName":self.name}
        while curr_token is not self.n_action_options:
            #sample action
            params = self.action_params
            action_dist = torch.distributions.Categorical(logits=params)
            token = action_dist.sample()
            action_log_prob = action_dist.log_prob(token)
            total_logprob += action_log_prob
            sample["tokens"].append(token.item())
            curr_token = token
            
            change_state_dist = torch.distributions.Bernoulli(logits = self.change_state)
            change_state = change_state_dist.sample()
            
            change_state_log_prob = change_state_dist.log_prob(change_state)
            #print(total_logprob,change_state_log_prob)
            total_logprob += change_state_log_prob[0]


            if change_state > 0:
                break
            
            

        transition_dist = torch.distributions.Categorical(logits=self.transition_params)
        next_action = transition_dist.sample()
        transition_log_prob = transition_dist.log_prob(next_action)
        total_logprob += transition_log_prob

        sample["next_state"] = next_action
        sample['logprob'] = total_logprob
        return sample
    
    def __str__(self) -> str:
        return super().__str__() + f" {self.name} {self.action_params.data} {self.transition_params.data}"

    def __repr__(self):
        return super().__repr__() + f" {self.name} {self.action_params.data} {self.transition_params.data}"


class AbstractAction(nn.Module):
    def __init__(self, n_low_level_actions,n_transitions, name = 'op') -> None:
        super().__init__()
        self.name=  name
        self.n_low_level_actions = n_low_level_actions
        self.n_action_options = n_low_level_actions + 1 # for stop
        self.n_transitions = n_transitions
        self.n_transition_options = n_transitions + 1 # for stop


        #action parameters
        self.action_params = nn.Parameter(torch.randn(self.n_action_options))
        #transition parameters
        self.transition_params = nn.Parameter(torch.randn(self.n_transition_options))

    def forward(self):
        
        curr_token = None
        total_logprob = 0
        sample = {'tokens':[], 'next_state':None, 'logprob':None,"AbstractActionName":self.name}
        while curr_token is not self.n_action_options:
            #sample action
            params = self.action_params
            #if curr_token is None:
            #    params = self.action_params[:-1]
            action_dist = torch.distributions.Categorical(logits=params)
            token = action_dist.sample()
            action_log_prob = action_dist.log_prob(token)
            total_logprob += action_log_prob
            
            if token == self.n_action_options-1:
                
                break
            
            sample["tokens"].append(token.item())
            curr_token = token

        transition_dist = torch.distributions.Categorical(logits=self.transition_params)
        next_action = transition_dist.sample()
        transition_log_prob = transition_dist.log_prob(next_action)
        total_logprob += transition_log_prob

        sample["next_state"] = next_action
        sample['logprob'] = total_logprob
        return sample
    
    def __str__(self) -> str:
        return super().__str__() + f" {self.name} {self.action_params.data} {self.transition_params.data}"

    def __repr__(self):
        return super().__repr__() + f" {self.name} {self.action_params.data} {self.transition_params.data}"

class ActionAbstractionModel(nn.Module):

    def __init__(self, n_low_level_actions,max_n_abstract_actions) -> None:

        super().__init__()
        self.n_low_level_actions = n_low_level_actions
        self.max_n_abstract_actions = max_n_abstract_actions
        self.n_transition_options = max_n_abstract_actions + 1 # for stop 

        self.abstract_actions = nn.ModuleList([AbstractAction(n_low_level_actions, max_n_abstract_actions, name = f"op_{i}") for i in range(1,max_n_abstract_actions+1)])

        self.init_state_params =  nn.Parameter(torch.randn(self.max_n_abstract_actions))

    def forward(self):

        seq = Sequence()
        
        start_state_dist = torch.distributions.Categorical(logits=self.init_state_params)
        start_state = start_state_dist.sample()
        start_state_log_prob = start_state_dist.log_prob(start_state)
        seq.total_logprob += start_state_log_prob
        curr_state = start_state
        curr_state = 0
        while curr_state != self.n_transition_options-1:
            aa = self.abstract_actions[curr_state]
            sample = aa()
            seq.add_component(sample)
            curr_state = sample['next_state']
            


        return seq



    def __str__(self) -> str:
        return super().__str__() 


        
if __name__=="__main__":
    import pickle as pkl

    data = pkl.load(open("trajectories.pkl","rb"))

    






