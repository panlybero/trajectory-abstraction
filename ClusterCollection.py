class ClusterCollection:
    def __init__(self):
        self.clusters = []

    def _cluster_names(self):

        return {cluster.state_description: f'cluster_{i}' for i, cluster in enumerate(self.clusters)}

    def __getitem__(self, description):

        for cluster in self.clusters:
            if cluster.state_description == description:
                return cluster

        raise KeyError(f"Cluster with description {description} not found")

    def __setitem__(self, description, cluster):
        # if found, replace
        for i in range(len(self.clusters)):
            if self.clusters[i].state_description == description:
                self.clusters[i] = cluster
                return

        # if not found, append

        self.clusters.append(cluster)

    def __iter__(self):
        return iter(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def items(self):
        return [(cluster.state_description, cluster) for cluster in self.clusters]

    def values(self):
        return self.clusters

    def keys(self):
        return [cluster.state_description for cluster in self.clusters]

    def pop(self, description):
        for i in range(len(self.clusters)):
            if self.clusters[i].state_description == description:
                return self.clusters.pop(i)

        raise KeyError(f"Cluster with description {description} not found")
