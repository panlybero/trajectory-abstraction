from StateDescription import StateDescription
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch

class StateDescriptionClustering:
    def __init__(self, state_descriptions, counts_vectors):
        self.state_descriptions = state_descriptions
        self.counts_vectors = np.array(counts_vectors)
        self.cluster_labels = None
        self.clusters = None

    def _normalize_counts(self, counts):
        total_counts = np.sum(counts, axis=1)
        normalized_counts = counts / total_counts[:, np.newaxis]
        normalized_counts[np.isnan(normalized_counts)] = 0.0  # Handle division by zero
        return normalized_counts

    def _compute_distance_matrix(self, counts_vectors):
        normalized_counts = self._normalize_counts(counts_vectors)
        distance_matrix = np.zeros((len(normalized_counts), len(normalized_counts)))

        for i in range(len(normalized_counts)):
            for j in range(i+1, len(normalized_counts)):
                cross_entropy = np.sum(self.cross_entropy_distance(normalized_counts[i], normalized_counts[j]))
                distance_matrix[i, j] = cross_entropy
                distance_matrix[j, i] = cross_entropy


        return distance_matrix
    
    
    
    def cross_entropy_distance(self, p1, p2):
        
        cross_entropy = torch.nn.CrossEntropyLoss()
        p1 = torch.tensor(p1).reshape(1,-1)
        p2 = torch.tensor(p2).reshape(1,-1)

        val = cross_entropy(p1, p2).item()

        return val
    
    def cluster_state_descriptions(self, num_clusters):

        normalized_counts = self._normalize_counts(self.counts_vectors)
        distance_matrix = self._compute_distance_matrix(self.counts_vectors)
        
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
        labels = clustering.fit_predict(distance_matrix)
        clusters = {}
        
        for state_desc, label in zip(self.state_descriptions, labels):
            if label not in clusters:
                clusters[label] = {
                    'states': [],
                    'least_general_generalization': None
                }
            clusters[label]['states'].append(state_desc)
        
        clusters = self._compute_generalizations(clusters)
        
        self.clusters = clusters
        self.cluster_labels = labels

        return clusters

    def _compute_generalizations(self,clusters):
        for label, cluster in clusters.items():
            state_descriptions = cluster['states']
            least_general_generalization = StateDescription.least_general_generalization(state_descriptions)
            cluster['least_general_generalization'] = least_general_generalization


        return clusters

    def get_generalizations(self):
        generalizations = []
        
        for cluster_label, cluster_info in self.clusters.items():
            generalizations.append(cluster_info['least_general_generalization'])
        
        return generalizations

    def compute_cluster_counts(self,as_dict=False):
        cluster_counts = {}
        #print(self.cluster_labels.shape, self.counts_vectors.shape)
       
        for label in np.unique(self.cluster_labels):
            counts = self.counts_vectors[self.cluster_labels == label]
            #cluster_counts.append(np.sum(counts, axis=0))
            cluster_counts[label] = np.sum(counts, axis=0)
        

        for label,cluster in self.clusters.items():
            cluster['counts'] = cluster_counts[label]

        if as_dict:
            return cluster_counts

        return np.array(list(cluster_counts.values()))


    def compute_cluster_distance_matrix(self):
        if self.clusters is None:
            raise ValueError("Clustering has not been performed.")
        

        cluster_counts = self.compute_cluster_counts()
        
        normalized_counts = self._normalize_counts(cluster_counts)
        
        distance_matrix = self._compute_distance_matrix(normalized_counts)
        return distance_matrix

    def merge_lowest_entropy_clusters(self, threshold):
        if self.clusters is None:
            raise ValueError("Clustering has not been performed.")
        
        distance_matrix = self.compute_cluster_distance_matrix()
        
        num_clusters = len(self.clusters)
        
        min_distance = np.inf
        min_i, min_j = -1, -1
        
        for i in range(num_clusters):
            for j in range(i+1, num_clusters):
                if distance_matrix[i, j] < min_distance:
                    min_distance = distance_matrix[i, j]
                    min_i = i
                    min_j = j
        
        if min_distance <= threshold:
            self.clusters[min_i]['states'] += self.clusters[min_j]['states']
            self.clusters.pop(min_j)
            self.cluster_labels[self.cluster_labels == min_j] = min_i

        self.clusters = self._compute_generalizations(self.clusters)

        
    
    def get_action_probability(self,state_description, action):
        self.cluster_counts = self.compute_cluster_counts()
        
        cluster_indices = [i for i, cluster in self.clusters.items() if cluster['least_general_generalization'].subsumes(state_description)]
        action_probabilities = {}

        for index in cluster_indices:
            counts_vector = self.cluster_counts[index]
            total_counts = np.sum(counts_vector)
            action_index = action
            action_probability = counts_vector[action_index] / total_counts
            action_probabilities[self.clusters[index]['least_general_generalization']] = (action_probability)

        return action_probabilities

    def add_cluster(self,state_description):

        if state_description in self.generalizations:
            raise ValueError("State description already exists in clusters.")
        
        cluster = {
            'states': [],
            'least_general_generalization': state_description,
            'counts':np.zeros((len(self.counts_vectors[0])))
        }

        self.clusters[len(self.clusters)]= (cluster)


        


    @property
    def generalizations(self):
        return self.get_generalizations()

    def get_cluster(self,state_description):
        for cluster in self.clusters.values():
            if cluster['least_general_generalization'] == state_description:
                return cluster

        raise ValueError("State description not found in clusters.")


    def cluster_index(self, generalization):
        for idx, cluster in self.clusters.items():
            if cluster['least_general_generalization'] == generalization:
                return idx
        raise ValueError("State description not found in clusters.")


    def add_step_to_cluster(self,state_description,count_idx):
        
        cluster = self.get_cluster(state_description)
        print(cluster)
        if state_description not in cluster['states']:
            cluster['states'].append(state_description)
        print("-------------------------")
        print(self.state_descriptions)
        print("Adding",state_description)
        print(self.counts_vectors)
        print("-------------------------")
        
        cluster['counts'][count_idx] += 1

        if state_description not in self.state_descriptions:
            self.state_descriptions.append(state_description)
            count_vec = np.zeros((1,len(self.counts_vectors[0])))
            count_vec[0,count_idx] = 1
            
            self.counts_vectors =np.concatenate((self.counts_vectors, count_vec), axis=0)
            self.cluster_labels = np.concatenate((self.cluster_labels, np.array([self.cluster_index(cluster['least_general_generalization'])])), axis=0)
            
        else:
            state_idx = self.state_descriptions.index(state_description)
            self.counts_vectors[state_idx,count_idx] += 1
            





    def add_step(self,state_description,count_idx):
        generalizations = self.generalizations
        applicable_gen = state_description.get_applicable_generalizations(generalizations)
        if applicable_gen is None:
            raise ValueError("No applicable generalizations found.")
        
        self.add_step_to_cluster(applicable_gen[0],count_idx)

    




