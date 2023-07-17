import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph 

class MarkovProcessModel:
    def __init__(self, num_states, smoothing_factor=1):
        self.num_states = num_states
        self.transition_counts = None
        self.transition_probs = None
        self.smoothing_factor = smoothing_factor
        self.sink_states = None

    def fit(self, X):
        self.transition_counts, self.sink_states = self._count_state_transitions(X)
        self.transition_probs = self._normalize_transition_counts()

    def _count_state_transitions(self, data):
        transition_counts = np.zeros((self.num_states, self.num_states), dtype=int)
        sink_states = np.ones(self.num_states, dtype=bool)

        for trajectory in data:
            for i in range(len(trajectory) - 1):
                current_state = trajectory[i]
                next_state = trajectory[i + 1]
                transition_counts[current_state, next_state] += 1
                sink_states[current_state] = False

        return transition_counts, sink_states



    def _normalize_transition_counts(self):
        transition_probs = self.transition_counts + self.smoothing_factor
        transition_probs = transition_probs / np.sum(transition_probs, axis=1, keepdims=True)
        return transition_probs

    def predict(self, start_state, n_steps=1):
        if self.transition_probs is None:
            raise ValueError("Model not trained. Call fit() before making predictions.")

        if n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")

        if start_state < 0 or start_state >= self.num_states:
            raise ValueError("Invalid start_state value.")

        current_state = start_state
        trajectory = [current_state]

        for _ in range(n_steps):
            next_state = np.random.choice(self.num_states, p=self.transition_probs[current_state])
            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def plot_transition_graph(self, node_labels=None,fname='file.dot'):
        if self.transition_probs is None:
            raise ValueError("Model not trained. Call fit() before plotting the transition graph.")

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes to the graph
        for i in range(self.num_states):
            G.add_node(i)

        # Add edges to the graph with transition probabilities as labels
        for i in range(self.num_states):
            for j in range(self.num_states):
                prob = self.transition_probs[i, j]
                if prob > 0:
                    G.add_edge(i, j, weight=prob)

        for u,v,d in G.edges(data=True):
            d['label'] = f"{d.get('weight',''):.3f}" if not self.sink_states[u] else '0.0'


        if node_labels is not None:
            for i in range(self.num_states):
                G.nodes[i]['label'] = node_labels[i]
                if self.sink_states[i]:
                    G.nodes[i]['color'] = 'red' 
                    G.remove_edges_from([(i,j) for j in range(self.num_states)])

        
        #prune edges with count 1
        for i in range(self.num_states):
            for j in range(self.num_states):
                if self.transition_counts[i,j] == 0:
                    if (i,j) in G.edges():
                        G.remove_edge(i,j)
                
        # Get the positions of the nodes for plotting
        pos = nx.circular_layout(G)

        A = to_agraph(G)
        A.layout('dot')          
                                                           
        A.draw('multi.png') 
        A.write(fname)
        # Draw edges with labels
        edge_labels = {(u, v): f'{G.edges[u, v]["weight"]:.3f}' for u, v in G.edges}
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # Draw nodes
        #nx.draw_networkx_nodes(G, pos)

        
        #plt.savefig('transition_graph.png')