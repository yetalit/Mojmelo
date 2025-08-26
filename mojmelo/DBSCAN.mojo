from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from buffer import NDBuffer
from algorithm import parallelize

struct DBSCAN:
    var squared_eps: Float32
    var min_samples: Int

    fn __init__(out self, eps: Float32 = 1.0, min_samples: Int = 5) raises:
        self.squared_eps = eps ** 2
        self.min_samples = min_samples

    fn predict(mut self, X: Matrix) raises -> Matrix:
        var kdtree = KDTree(X, 'euc')
        var result = Matrix.full(X.height, 1, -1.0)

        var neighborhoods = List[KDTreeResultVector](capacity=X.height)
        neighborhoods.resize(X.height, KDTreeResultVector())
        @parameter
        fn p(i: Int):
            kdtree.r_nearest(NDBuffer[dtype=DType.float32, rank=1](X[i, unsafe=True].data, X.width), self.squared_eps, neighborhoods[i])
        parallelize[p](X.height)
        
        var current_cluster = 0
        var stack = List[Int]()

        for i in range(X.height):
            if result.data[i] != -1 or len(neighborhoods[i]) < self.min_samples:
                continue

            # Depth-first search starting from i, ending at the non-core points.
            # This is very similar to the classic algorithm for computing connected
            # components, the difference being that we label non-core points as
            # part of a cluster (component), but don't expand their neighborhoods.
            while True:
                if result.data[i] == -1:
                    result.data[i] = current_cluster
                    var neighb = neighborhoods[i]
                    if len(neighb) >= self.min_samples:
                        for i in range(len(neighb)):
                            if result[neighb[i].idx] == -1:
                                stack.append(neighb[i].idx)

                if len(stack) == 0:
                    break
                i = stack.pop()

            current_cluster += 1

        return result^
