from mojmelo.utils.Matrix import Matrix
import math
from mojmelo.utils.hdbscan.KDTreeBoruvka import KDTreeBoruvka
from mojmelo.utils.hdbscan.hdbscan_boruvka import HDBSCANBoruvka
from mojmelo.utils.hdbscan.hdbscan_linkage import label
from mojmelo.utils.hdbscan.hdbscan_tree import condense_tree, get_clusters, compute_stability, simplify_hierarchy

struct HDBSCAN:
    """Cluster data using hierarchical density-based clustering."""
    var min_samples: Int
    """The number of samples in a neighborhood for a point to be considered as a core point."""
    var min_cluster_size: Int
    """The minimum number of samples in a group for that group to be considered a cluster."""
    var cluster_selection_method: String
    """The method used to select clusters from the condensed tree -> 'eom', 'leaf'."""
    var alpha: Float32
    """A distance scaling parameter."""
    var cluster_selection_epsilon: Float32
    """A distance threshold. Clusters below this value will be merged."""
    var cluster_selection_epsilon_max: Float32
    """A distance threshold. Clusters above this value will be split.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon."""
    var cluster_selection_persistence: Float32
    """A persistence threshold. Clusters with a persistence lower than this
        value will be merged."""
    var max_cluster_size: Int
    """A limit to the size of clusters returned by the eom algorithm."""
    var allow_single_cluster: Bool
    """By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset."""
    var match_reference_implementation: Bool
    """There exist some interpretational differences between this
        HDBSCAN* implementation and the original authors reference
        implementation in Java. This can result in very minor differences
        in clustering results. Setting this flag to True will, at a some
        performance cost, ensure that the clustering results match the
        reference implementation."""
    var search_deepness_coef: Int
    """Current KDTree implementation applies some approximation to its search results.
        Increasing search_deepness_coef can lead to more accurate results at the cost of performance.
        This can be useful for small datasets."""

    var labels: List[Scalar[DType.int]]
    """Cluster labels for each point in the dataset given to fit()."""
    var probabilities: List[Float32]
    """The strength with which each sample is a member of its assigned cluster."""
    var cluster_persistence: List[Float32]
    """A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster."""
    var condensed_tree: Dict[String, List[Scalar[DType.int]]]
    """The condensed tree produced by HDBSCAN."""
    var condensed_tree_lambda: List[Float32]
    """The condensed tree lambda values produced by HDBSCAN."""
    var single_linkage_tree: Matrix
    """The single linkage tree produced by HDBSCAN."""

    fn __init__(out self,
        min_samples: Int = 5,
        min_cluster_size: Int = 5,
        cluster_selection_method: String = 'eom',
        alpha: Float32 = 1.0,
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
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_epsilon_max = cluster_selection_epsilon_max
        self.cluster_selection_persistence = cluster_selection_persistence
        self.max_cluster_size = max_cluster_size
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.search_deepness_coef = search_deepness_coef
        
        self.labels = List[Scalar[DType.int]]()
        self.probabilities = List[Float32]()
        self.cluster_persistence = List[Float32]()
        self.condensed_tree = Dict[String, List[Scalar[DType.int]]]()
        self.condensed_tree_lambda = List[Float32]()
        self.single_linkage_tree = Matrix(0, 0)

    fn fit(mut self, X: Matrix) raises:
        """Find clusters based on hierarchical density-based clustering."""

        if self.min_samples < 1:
            raise Error('min_samples cannot be smaller than 1!')
        if self.min_cluster_size < 2:
            raise Error('min_cluster_size cannot be smaller than 2!')
        if self.cluster_selection_method != 'eom' and self.cluster_selection_method != 'leaf':
            raise Error('Invalid cluster_selection_method value!')

        var tree = KDTreeBoruvka(X, min_samples=self.min_samples, leaf_size=max(32, 2 * self.min_samples), search_deepness_coef=self.search_deepness_coef)
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

        self.labels = resulted_clusters[0].copy()
        self.probabilities = resulted_clusters[1].copy()
        self.cluster_persistence = resulted_clusters[2].copy()
        self.condensed_tree = condensed_tree^
        self.condensed_tree_lambda = lambda_vals^
        self.single_linkage_tree = hierarchy^
    
    fn fit_predict(mut self, X: Matrix) raises -> List[Scalar[DType.int]]:
        """Cluster X and return the associated cluster labels.
        
        Returns:
            List of cluster indices.
        """
        self.fit(X)
        return self.labels.copy()
