Mojo struct

# `HDBSCAN`

```mojo
@memory_only
struct HDBSCAN
```

Cluster data using hierarchical density-based clustering.

## Aliases

- `__del__is_trivial = False`

## Fields

- **min_samples** (`Int`): The number of samples in a neighborhood for a point to be considered as a core point.
- **min_cluster_size** (`Int`): The minimum number of samples in a group for that group to be considered a cluster.
- **cluster_selection_method** (`String`): The method used to select clusters from the condensed tree -> 'eom', 'leaf'.
- **alpha** (`Float32`): A distance scaling parameter.
- **cluster_selection_epsilon** (`Float32`): A distance threshold. Clusters below this value will be merged.
- **cluster_selection_epsilon_max** (`Float32`): A distance threshold. Clusters above this value will be split. Has no effect when using leaf clustering (where clusters are usually small regardless) and can also be overridden in rare cases by a high value for cluster_selection_epsilon.
- **cluster_selection_persistence** (`Float32`): A persistence threshold. Clusters with a persistence lower than this value will be merged.
- **max_cluster_size** (`Int`): A limit to the size of clusters returned by the eom algorithm.
- **allow_single_cluster** (`Bool`): By default HDBSCAN* will not produce a single cluster, setting this to True will override this and allow single cluster results in the case that you feel this is a valid result for your dataset.
- **match_reference_implementation** (`Bool`): There exist some interpretational differences between this HDBSCAN* implementation and the original authors reference implementation in Java. This can result in very minor differences in clustering results. Setting this flag to True will, at a some performance cost, ensure that the clustering results match the reference implementation.
- **search_deepness_coef** (`Int`): Current KDTree implementation applies some approximation to its search results. Increasing search_deepness_coef can lead to more accurate results at the cost of performance. This can be useful for small datasets.
- **labels** (`List[Scalar[DType.index]]`): Cluster labels for each point in the dataset given to fit().
- **probabilities** (`List[Float32]`): The strength with which each sample is a member of its assigned cluster.
- **cluster_persistence** (`List[Float32]`): A score of how persistent each cluster is. A score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster.
- **condensed_tree** (`Dict[String, List[Scalar[DType.index]]]`): The condensed tree produced by HDBSCAN.
- **condensed_tree_lambda** (`List[Float32]`): The condensed tree lambda values produced by HDBSCAN.
- **single_linkage_tree** (`Matrix`): The single linkage tree produced by HDBSCAN.

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, min_samples: Int = 5, min_cluster_size: Int = 5, cluster_selection_method: String = "eom", alpha: Float32 = 1, cluster_selection_epsilon: Float32 = 0, cluster_selection_epsilon_max: Float32 = inf[DType.float32](), cluster_selection_persistence: Float32 = 0, max_cluster_size: Int = 0, allow_single_cluster: Bool = False, match_reference_implementation: Bool = False, search_deepness_coef: Int = 1)
```

**Args:**

- **min_samples** (`Int`)
- **min_cluster_size** (`Int`)
- **cluster_selection_method** (`String`)
- **alpha** (`Float32`)
- **cluster_selection_epsilon** (`Float32`)
- **cluster_selection_epsilon_max** (`Float32`)
- **cluster_selection_persistence** (`Float32`)
- **max_cluster_size** (`Int`)
- **allow_single_cluster** (`Bool`)
- **match_reference_implementation** (`Bool`)
- **search_deepness_coef** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
fn fit(mut self, X: Matrix)
```

Find clusters based on hierarchical density-based clustering.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Raises:**

### `fit_predict`

```mojo
fn fit_predict(mut self, X: Matrix) -> List[Scalar[DType.index]]
```

Cluster X and return the associated cluster labels.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List`: List of cluster indices.

**Raises:**


