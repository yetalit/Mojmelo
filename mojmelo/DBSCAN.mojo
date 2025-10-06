from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from buffer import NDBuffer
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

    fn __init__(out self, eps: Float32 = 1.0, min_samples: Int = 5, metric: String = 'euc') raises:
        self.metric = metric.lower()
        self.eps = eps ** 2 if self.metric == 'euc' else eps
        self.min_samples = min_samples

    fn predict(mut self, X: Matrix) raises -> Matrix:
        """Predict cluster indices.

        Returns:
            Vector of cluster indices.
        """
        var kdtree = KDTree(X, self.metric)
        var labels = Matrix.full(X.height, 1, -2.0)

        var neighborhoods = List[List[Int]](capacity=X.height)
        neighborhoods.resize(X.height, List[Int]())
        @parameter
        fn p(i: Int):
            var kd_results = KDTreeResultVector()
            kdtree.r_nearest(NDBuffer[dtype=DType.float32, rank=1](X[i, unsafe=True].data, X.width), self.eps, kd_results)
            for idp in range(len(kd_results)):
                neighborhoods[i].append(kd_results[idp].idx)
        parallelize[p](X.height)
        
        var current_cluster = 0

        for idx in range(X.height):
            # Label is not undefined.
            if labels.data[idx] != -2.0:
                continue

            # Check density.
            if len(neighborhoods[idx]) < self.min_samples:
                labels.data[idx] = -1.0
                continue

            var nbs_next = Set(neighborhoods[idx])
            var nbs_visited = Set[Int](idx)

            labels.data[idx] = current_cluster

            while len(nbs_next) > 0:
                var nb = nbs_next.pop()
                nbs_visited.add(nb)

                # Noise label.
                if labels.data[nb] == -1.0:
                    labels.data[nb] = current_cluster

                # Not undefined label.
                if labels.data[nb] != -2.0:
                    continue

                labels.data[nb] = current_cluster

                if len(neighborhoods[nb]) >= self.min_samples:
                    for qnb in neighborhoods[nb]:
                        if qnb not in nbs_visited:
                            nbs_next.add(qnb)

            current_cluster += 1

        return labels^
