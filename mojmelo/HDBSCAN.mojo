from mojmelo.utils.Matrix import Matrix
import math
from mojmelo.utils.hdbscan.KDTreeBoruvka import KDTreeBoruvka
from mojmelo.utils.hdbscan.hdbscan_boruvka import HDBSCANBoruvka
from mojmelo.utils.hdbscan.hdbscan_linkage import label
from mojmelo.utils.hdbscan.hdbscan_tree import condense_tree, get_clusters, compute_stability, simplify_hierarchy

struct HDBSCAN:
    var min_samples: Int
    var min_cluster_size: Int
    var cluster_selection_method: String
    var alpha: Float32
    var leaf_size: Int
    var cluster_selection_epsilon: Float32
    var cluster_selection_epsilon_max: Float32
    var cluster_selection_persistence: Float32
    var max_cluster_size: Int
    var allow_single_cluster: Bool
    var match_reference_implementation: Bool
    var search_deepness_coef: Int

    var labels_: List[Scalar[DType.int]]
    var probabilities_: List[Float32]
    var cluster_persistence_: List[Float32]
    var condensed_tree_: Dict[String, List[Scalar[DType.int]]]
    var condensed_tree_lambda_: List[Float32]
    var single_linkage_tree_: Matrix

    fn __init__(out self,
        min_samples: Int = 5,
        min_cluster_size: Int = 5,
        cluster_selection_method: String = 'eom',
        alpha: Float32 = 1.0,
        leaf_size: Int=32,
        cluster_selection_epsilon: Float32 = 0,
        cluster_selection_epsilon_max: Float32 = math.inf[DType.float32](),
        cluster_selection_persistence: Float32 = 0,
        max_cluster_size: Int = 0,
        allow_single_cluster: Bool = False,
        match_reference_implementation: Bool = False,
        search_deepness_coef: Int = 1
    ):
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_method = cluster_selection_method.lower()
        self.alpha = alpha
        self.leaf_size = leaf_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_epsilon_max = cluster_selection_epsilon_max
        self.cluster_selection_persistence = cluster_selection_persistence
        self.max_cluster_size = max_cluster_size
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.search_deepness_coef = search_deepness_coef
        
        self.labels_ = List[Scalar[DType.int]]()
        self.probabilities_ = List[Float32]()
        self.cluster_persistence_ = List[Float32]()
        self.condensed_tree_ = Dict[String, List[Scalar[DType.int]]]()
        self.condensed_tree_lambda_ = List[Float32]()
        self.single_linkage_tree_ = Matrix(0, 0)

    fn fit(mut self, X: Matrix) raises:
        if self.min_samples < 1:
            raise Error('min_samples cannot be smaller than 1!')
        if self.min_cluster_size < 2:
            raise Error('min_cluster_size cannot be smaller than 2!')
        if self.cluster_selection_method != 'eom' and self.cluster_selection_method != 'leaf':
            raise Error('Invalid cluster_selection_method value!')

        var tree = KDTreeBoruvka(X, min_samples=self.min_samples, leaf_size=self.leaf_size, search_deepness_coef=self.search_deepness_coef)
        var boruvka_alg = HDBSCANBoruvka(UnsafePointer(to=tree), min_samples=self.min_samples, alpha=self.alpha)
        var mst_edges = boruvka_alg.spanning_tree()
        _ = tree
        mst_edges = mst_edges[mst_edges['', 2].argsort()]
        var hierarchy = label(mst_edges)
        var resulted_tree = condense_tree(hierarchy, min_cluster_size=self.min_cluster_size)
        if self.cluster_selection_persistence > 0.0 and len(resulted_tree[1]) > 0:
            var tree = resulted_tree[0].copy()
            var lambda_ = resulted_tree[1].copy()
            resulted_tree = simplify_hierarchy(tree, lambda_, self.cluster_selection_persistence)
        var condensed_tree = resulted_tree[0].copy()
        var lambda_vals = resulted_tree[1].copy()
        var stability = compute_stability(condensed_tree, lambda_vals)

        var resulted_clusters = get_clusters(condensed_tree, lambda_vals, stability,
        cluster_selection_method=self.cluster_selection_method,
        allow_single_cluster=self.allow_single_cluster,
        match_reference_implementation=self.match_reference_implementation,
        cluster_selection_epsilon=self.cluster_selection_epsilon,
        max_cluster_size=self.max_cluster_size,
        cluster_selection_epsilon_max=self.cluster_selection_epsilon_max
        )

        self.labels_ = resulted_clusters[0].copy()
        self.probabilities_ = resulted_clusters[1].copy()
        self.cluster_persistence_ = resulted_clusters[2].copy()
        self.condensed_tree_ = condensed_tree^
        self.condensed_tree_lambda_ = lambda_vals^
        self.single_linkage_tree_ = hierarchy^
    
    fn fit_predict(mut self, X: Matrix) raises -> List[Scalar[DType.int]]:
        self.fit(X)
        return self.labels_.copy()
