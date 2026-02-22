from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from collections import Set
from algorithm import parallelize

struct DBSCAN:
    """A density based clustering method that expands clusters from samples that have more neighbors within a radius."""
    var eps: Float32
    """The maximum distance between two samples for one to be considered as in the neighborhood of the other."""
    var min_samples: Int
    """The number of samples in a neighborhood for a point to be considered as a core point."""
    var metric: String
    """Metric to use for distance computation:
    Euclidean -> 'euc';
    Manhattan -> 'man'.
    """
    var labels: List[Int]

    fn __init__(out self, eps: Float32 = 1.0, min_samples: Int = 5, metric: String = 'euc') raises:
        self.metric = metric.lower()
        self.eps = eps ** 2 if self.metric == 'euc' else eps
        self.min_samples = min_samples
        self.labels = List[Int]()

    fn fit(mut self, X: Matrix) raises:
        """Perform clustering."""
        self.labels = List[Int](capacity=X.height)
        self.labels.resize(X.height, -2)
        var kdtree = KDTree(X, self.metric)

        var neighborhoods = List[List[Int]](capacity=X.height)
        neighborhoods.resize(X.height, List[Int]())
        @parameter
        fn p(i: Int):
            var kd_results = KDTreeResultVector()
            kdtree.r_nearest(Span(ptr=X[i, unsafe=True].data, length=X.width), self.eps, kd_results)
            for idp in range(len(kd_results)):
                neighborhoods[i].append(kd_results[idp].idx)
        parallelize[p](X.height)
        
        var current_cluster = 0

        for idx in range(X.height):
            # Label is not undefined.
            if self.labels[idx] != -2:
                continue

            # Check density.
            if len(neighborhoods[idx]) < self.min_samples:
                self.labels[idx] = -1
                continue

            var nbs_next = Set(neighborhoods[idx])
            var nbs_visited = Set[Int](idx)

            self.labels[idx] = current_cluster

            while len(nbs_next) > 0:
                var nb = nbs_next.pop()
                nbs_visited.add(nb)

                # Noise label.
                if self.labels[nb] == -1:
                    self.labels[nb] = current_cluster

                # Not undefined label.
                if self.labels[nb] != -2:
                    continue

                self.labels[nb] = current_cluster

                if len(neighborhoods[nb]) >= self.min_samples:
                    for qnb in neighborhoods[nb]:
                        if qnb not in nbs_visited:
                            nbs_next.add(qnb)

            current_cluster += 1

    fn fit_predict(mut self, X: Matrix) raises -> List[Int]:
        """Perform clustering and predict cluster indices.

        Returns:
            List of cluster indices.
        """
        self.fit(X)
        return self.labels.copy()
