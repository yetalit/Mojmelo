from memory import memset_zero
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list
import math
from algorithm import reduction, vectorize
from utils.numerics import nan, isfinite, isinf
from collections import Set
from sys import CompilationTarget, simd_width_of

comptime simd_width: Int = 4 * simd_width_of[DType.int]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[DType.int]()

@always_inline
fn arange(start: Scalar[DType.int], stop: Scalar[DType.int]) -> List[Int]:
    var start_i = Int(start)
    var stop_i = Int(stop)
    var buff = List[Int](capacity=stop_i - start_i)
    buff.resize(stop_i - start_i, 0)
    for i in range(stop_i - start_i):
        buff[i] = i + start_i
    return buff^

@always_inline
fn arange(start: Int, stop: Int) -> List[Scalar[DType.int]]:
    var buff = List[Scalar[DType.int]](capacity=stop - start)
    buff.resize(stop - start, 0)
    for i in range(stop - start):
        buff[i] = i + start
    return buff^

fn bfs_from_hierarchy(hierarchy: Matrix, bfs_root: Int) raises -> List[Int]:
    var dim = hierarchy.height
    var max_node = 2 * dim
    var num_points = max_node - dim + 1

    var to_process = [bfs_root]
    var result = List[Int]()
    var visited = Dict[Int, Bool]()
    visited[bfs_root] = True

    while len(to_process) > 0:
        var next = List[Int]()
        for x in to_process:
            result.append(x)

            if x >= num_points:
                var idx = x - num_points
                var l = Int(hierarchy[idx, 0])
                var r = Int(hierarchy[idx, 1])

                if not l in visited:
                    next.append(l)
                    visited[l] = True

                if not r in visited:
                    next.append(r)
                    visited[r] = True

        to_process = next^

    return result^


fn condense_tree(hierarchy: Matrix, min_cluster_size: Int=10) raises -> Tuple[Dict[String, List[Scalar[DType.int]]], List[Float32]]:
    var left: Int
    var right: Int
    var lambda_value: Float32
    var left_count: Int
    var right_count: Int

    var root = 2 * hierarchy.height
    var num_points = hierarchy.height + 1
    var next_label = num_points + 1

    var node_list = bfs_from_hierarchy(hierarchy, root)

    var relabel = List[Scalar[DType.int]](capacity=root + 1)
    relabel.resize(root + 1, 0)
    relabel[root] = num_points

    var result_int = Dict[String, List[Scalar[DType.int]]]()
    result_int['parent'] = List[Scalar[DType.int]]()
    result_int['child'] = List[Scalar[DType.int]]()
    result_int['child_size'] = List[Scalar[DType.int]]()

    var result_float = List[Float32]()
    var ignore = List[Int8](capacity=root + 1)
    ignore.resize(root + 1, 0)

    for node in node_list:

        if ignore[node] or node < num_points:
            continue

        ignore[node] = 1

        left = Int(hierarchy[node - num_points, 0])
        right = Int(hierarchy[node - num_points, 1])

        if hierarchy[node - num_points, 2] > 0.0:
            lambda_value = 1.0 / Float32(hierarchy[node - num_points, 2])
        else:
            lambda_value = math.inf[DType.float32]()

        if left >= num_points:
            left_count = Int(hierarchy[left - num_points, 3])
        else:
            left_count = 1

        if right >= num_points:
            right_count = Int(hierarchy[right - num_points, 3])
        else:
            right_count = 1

        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = next_label
            next_label += 1
            result_int['parent'].append(relabel[node])
            result_int['child'].append(relabel[left])
            result_int['child_size'].append(left_count)
            result_float.append(lambda_value)

            relabel[right] = next_label
            next_label += 1
            result_int['parent'].append(relabel[node])
            result_int['child'].append(relabel[right])
            result_int['child_size'].append(right_count)
            result_float.append(lambda_value)

        elif left_count < min_cluster_size and right_count < min_cluster_size:
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_int['parent'].append(relabel[node])
                    result_int['child'].append(sub_node)
                    result_int['child_size'].append(1)
                    result_float.append(lambda_value)
                ignore[sub_node] = 1

            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_int['parent'].append(relabel[node])
                    result_int['child'].append(sub_node)
                    result_int['child_size'].append(1)
                    result_float.append(lambda_value)
                ignore[sub_node] = 1

        elif left_count < min_cluster_size:
            relabel[right] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_int['parent'].append(relabel[node])
                    result_int['child'].append(sub_node)
                    result_int['child_size'].append(1)
                    result_float.append(lambda_value)
                ignore[sub_node] = 1

        else:
            relabel[left] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_int['parent'].append(relabel[node])
                    result_int['child'].append(sub_node)
                    result_int['child_size'].append(1)
                    result_float.append(lambda_value)
                ignore[sub_node] = 1

    return result_int^, result_float^


fn compute_stability(condensed_tree: Dict[String, List[Scalar[DType.int]]], lambda_vals: List[Float32]) raises -> Dict[Scalar[DType.int], Float32]:

    var largest_child = reduction.max(Span[Scalar[DType.int], origin_of(condensed_tree['child'])](ptr=condensed_tree['child'].unsafe_ptr(), length=len(condensed_tree['child'])))
    var smallest_cluster = reduction.min(Span[Scalar[DType.int], origin_of(condensed_tree['parent'])](ptr=condensed_tree['parent'].unsafe_ptr(), length=len(condensed_tree['parent'])))
    var num_clusters = (reduction.max(Span[Scalar[DType.int], origin_of(condensed_tree['parent'])](ptr=condensed_tree['parent'].unsafe_ptr(), length=len(condensed_tree['parent']))) -
                                   smallest_cluster + 1)

    var parents = condensed_tree['parent'].copy()
    var sizes = condensed_tree['child_size'].copy()
    var lambdas = lambda_vals.copy()

    if largest_child < smallest_cluster:
        largest_child = smallest_cluster

    var sorted_children = condensed_tree['child'].copy()
    var sorted_lambdas = List[Float32](capacity=len(lambdas))
    sorted_lambdas.resize(len(lambdas), 0)

    var sorted_indices = fill_indices_list(len(sorted_children))
    @parameter
    fn cmp_int(a: Scalar[DType.int], b: Scalar[DType.int]) -> Bool:
        return sorted_children[a] < sorted_children[b]

    sort[cmp_int](
            Span[
                Scalar[DType.int],
                origin_of(sorted_indices),
            ](ptr=sorted_indices.unsafe_ptr(), length=len(sorted_indices)))
    for i, idx in enumerate(sorted_indices):
        sorted_children[i] = condensed_tree['child'][idx]
        sorted_lambdas[i] = lambdas[idx]

    var births = List[Float32](capacity=Int(largest_child) + 1)
    births.resize(Int(largest_child) + 1, nan[DType.float32]())

    var current_child = -1
    var min_lambda: Float32 = 0.0

    for row in range(len(sorted_children)):
        var child = Int(sorted_children[row])
        var lambda_ = sorted_lambdas[row]

        if child == current_child:
            min_lambda = min(min_lambda, lambda_)
        elif current_child != -1:
            births[current_child] = min_lambda
            current_child = child
            min_lambda = lambda_
        else:
            # Initialize
            current_child = child
            min_lambda = lambda_

    if current_child != -1:
        births[current_child] = min_lambda
    births[smallest_cluster] = 0.0

    var result_arr = List[Float32](capacity=Int(num_clusters))
    result_arr.resize(Int(num_clusters), 0)

    for i in range(len(parents)):
        var parent = parents[i]
        var lambda_ = lambdas[i]
        var child_size = sizes[i]
        var result_index = parent - smallest_cluster

        result_arr[result_index] += (lambda_ - births[parent]) * Int(child_size)

    var ids = arange(Int(smallest_cluster), Int(reduction.max(Span[Scalar[DType.int], origin_of(condensed_tree['parent'])](ptr=condensed_tree['parent'].unsafe_ptr(), length=len(condensed_tree['parent'])))) + 1)
    var result_pre_dict = Dict[Scalar[DType.int], Float32]()
    for i in range(num_clusters):
        result_pre_dict[ids[i]] = result_arr[i]

    return result_pre_dict^


fn bfs_from_cluster_tree(tree: Dict[String, List[Scalar[DType.int]]], bfs_root: Scalar[DType.int]) raises -> List[Scalar[DType.int]]:

    var result = List[Scalar[DType.int]]()
    var to_process = [bfs_root]

    while len(to_process) > 0:
        result.extend(to_process.copy())
        var to_process_dict = Dict[Scalar[DType.int], Scalar[DType.int]].fromkeys(to_process)
        to_process.clear()
        for i in range(len(tree['parent'])):
            if tree['parent'][i] in to_process_dict:
                to_process.append(tree['child'][i])

    return result^


fn max_lambdas(tree: Dict[String, List[Scalar[DType.int]]], lambda_vals: List[Float32]) raises -> List[Float32]:
    var largest_parent = Int(reduction.max(Span[Scalar[DType.int], origin_of(tree['parent'])](ptr=tree['parent'].unsafe_ptr(), length=len(tree['parent']))))

    var deaths = List[Float32](capacity=largest_parent + 1)
    deaths.resize(largest_parent + 1, 0)

    var sorted_parents = tree['parent'].copy()
    var sorted_lambdas = List[Float32](capacity=len(lambda_vals))
    sorted_lambdas.resize(len(lambda_vals), 0)

    var sorted_indices = fill_indices_list(len(sorted_parents))
    @parameter
    fn cmp_int(a: Scalar[DType.int], b: Scalar[DType.int]) -> Bool:
        return sorted_parents[a] < sorted_parents[b]

    sort[cmp_int](
            Span[
                Scalar[DType.int],
                origin_of(sorted_indices),
            ](ptr=sorted_indices.unsafe_ptr(), length=len(sorted_indices)))
    for i, idx in enumerate(sorted_indices):
        sorted_parents[i] = tree['parent'][idx]
        sorted_lambdas[i] = lambda_vals[idx]

    var current_parent = -1
    var max_lambda: Float32 = 0.0

    for row in range(len(sorted_parents)):
        var parent = Int(sorted_parents[row])
        var lambda_ = sorted_lambdas[row]

        if parent == current_parent:
            max_lambda = max(max_lambda, lambda_)
        elif current_parent != -1:
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_
        else:
            # Initialize
            current_parent = parent
            max_lambda = lambda_
    
    deaths[current_parent] = max_lambda # value for last parent

    return deaths^


struct TreeUnionFind:
    var _data: UnsafePointer[Scalar[DType.int], MutAnyOrigin]
    var size: Int
    comptime width = 2
    var is_component: List[Bool]

    @always_inline
    fn __init__(out self, size: Int):
        self._data = alloc[Scalar[DType.int]](size * self.width)
        memset_zero(self._data, size * self.width)
        self.size = size
        self.is_component = List[Bool](capacity=size)
        self.is_component.resize(size, True)
        var _arange = arange(0, size)
        var tmpPtr = self._data
        @parameter
        fn v[simd_width: Int](idx: Int) unified {mut}:
            tmpPtr.strided_store[width=simd_width](_arange._data.load[width=simd_width](idx), self.width)
            tmpPtr += simd_width * self.width
        vectorize[simd_width](self.size, v)

    fn union_(mut self, x: Int, y: Int):
        var x_root = self.find(x)
        var y_root = self.find(y)

        if self._data[x_root * self.width + 1] < self._data[y_root * self.width + 1]:
            self._data[x_root * self.width] = y_root
        elif self._data[x_root * self.width + 1] > self._data[y_root * self.width + 1]:
            self._data[y_root * self.width] = x_root
        else:
            self._data[y_root * self.width] = x_root
            self._data[x_root * self.width + 1] += 1

        return

    fn find(mut self, x: Scalar[DType.int]) -> Scalar[DType.int]:
        if self._data[x * self.width] != x:
            self._data[x * self.width] = self.find(self._data[x * self.width])
            self.is_component[x] = False
        return self._data[x * self.width]

    fn components(self) -> List[Int]:
        var args = List[Int]()
        for i in range(self.size):
            if self.is_component[i]:
                args.append(i)
        return args^

    @always_inline
    fn __del__(deinit self):
        if self._data:
            self._data.free()

fn labelling_at_cut(
        linkage: Matrix,
        cut: Float32,
        min_cluster_size: Int) raises -> List[Int]:

    var root = 2 * linkage.height
    var num_points = root // 2 + 1

    var result = List[Int](capacity=num_points)
    result.resize(num_points, 0)

    var union_find = TreeUnionFind(root + 1)

    var cluster = num_points
    for i in range(linkage.height):
        var row = linkage[i]
        if row[0, 2] < cut:
            union_find.union_(Int(row[0, 0]), cluster)
            union_find.union_(Int(row[0, 1]), cluster)
        cluster += 1

    var cluster_size = List[Int](capacity=cluster)
    cluster_size.resize(cluster, 0)

    for n in range(num_points):
        cluster = Int(union_find.find(n))
        cluster_size[cluster] += 1
        result[n] = cluster

    var cluster_label_map = {-1: -1}
    var cluster_label = 0
    var unique_labels = Set(result)

    for cluster in unique_labels:
        if cluster_size[cluster] < min_cluster_size:
            cluster_label_map[cluster] = -1
        else:
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1

    for n in range(num_points):
        result[n] = cluster_label_map[result[n]]

    return result^


fn do_labelling(
        tree: Dict[String, List[Scalar[DType.int]]], lambda_array: List[Float32],
        clusters: Set[Scalar[DType.int]],
        cluster_label_map: Dict[Scalar[DType.int], Int],
        allow_single_cluster: Int,
        cluster_selection_epsilon: Float32,
        match_reference_implementation: Int) raises -> List[Scalar[DType.int]]:

    var child_array = tree['child'].copy()
    var parent_array = tree['parent'].copy()

    var root_cluster = reduction.min(Span[Scalar[DType.int], origin_of(parent_array)](ptr=parent_array.unsafe_ptr(), length=len(parent_array)))
    var result = List[Scalar[DType.int]](capacity=Int(root_cluster))
    result.resize(Int(root_cluster), 0)

    var union_find = TreeUnionFind(Int(reduction.max(Span[Scalar[DType.int], origin_of(parent_array)](ptr=parent_array.unsafe_ptr(), length=len(parent_array)))) + 1)

    for n in range(len(parent_array)):
        var child = Int(child_array[n])
        var parent = Int(parent_array[n])
        if child not in clusters:
            union_find.union_(parent, child)

    for n in range(root_cluster):
        var cluster = union_find.find(n)
        if cluster < root_cluster:
            result[n] = -1
        elif cluster == root_cluster:
            if len(clusters) == 1 and allow_single_cluster and cluster in cluster_label_map:
                # check if `cluster` still exists in `cluster_label_map` and that it was not pruned
                # by `max_cluster_size` or `cluster_selection_epsilon_max` before executing this
                if cluster_selection_epsilon != 0.0:
                    if lambda_array[child_array.index(n)] >= 1 / cluster_selection_epsilon:
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1
                else:
                    var max_value = -math.inf[DType.float32]()
                    for i in range(len(parent_array)):
                        if parent_array[i] == cluster:
                            if lambda_array[i] > max_value:
                                max_value = lambda_array[i]
                    if lambda_array[child_array.index(n)] >= max_value:
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1
            else:
                result[n] = -1
        else:
            if match_reference_implementation:
                point_lambda = lambda_array[child_array.index(n)]
                cluster_lambda = lambda_array[child_array.index(cluster)]
                if point_lambda > cluster_lambda:
                    result[n] = cluster_label_map[cluster]
                else:
                    result[n] = -1
            else:
                result[n] = cluster_label_map[cluster]

    return result^


fn get_probabilities(tree: Dict[String, List[Scalar[DType.int]]], lambda_array: List[Float32], cluster_map: Dict[Int, Scalar[DType.int]], labels: List[Scalar[DType.int]], deaths: List[Float32]) raises -> List[Float32]:
    var child_array = tree['child'].copy()
    var parent_array = tree['parent'].copy()

    var result = List[Float32](capacity=len(labels))
    result.resize(len(labels), 0)
    var root_cluster = reduction.min(Span[Scalar[DType.int], origin_of(parent_array)](ptr=parent_array.unsafe_ptr(), length=len(parent_array)))

    for n in range(len(parent_array)):
        var point = child_array[n]
        if point >= root_cluster:
            continue

        var cluster_num = Int(labels[point])

        if cluster_num == -1:
            continue

        var cluster = cluster_map[cluster_num]
        var max_lambda = deaths[cluster]
        if max_lambda == 0.0 or not isfinite(lambda_array[n]):
            result[point] = 1.0
        else:
            lambda_ = min(lambda_array[n], max_lambda)
            result[point] = lambda_ / max_lambda

    return result^


fn outlier_scores(tree: Dict[String, List[Scalar[DType.int]]], lambda_array: List[Float32]) raises -> List[Float32]:
    var child_array = tree['child'].copy()
    var parent_array = tree['parent'].copy()

    var deaths = max_lambdas(tree, lambda_array)
    var root_cluster = reduction.min(Span[Scalar[DType.int], origin_of(parent_array)](ptr=parent_array.unsafe_ptr(), length=len(parent_array)))
    var result = List[Float32](capacity=Int(root_cluster))
    result.resize(Int(root_cluster), 0)

    for n in range(len(parent_array) - 1, -1, -1):
        var cluster = child_array[n]
        if cluster < root_cluster:
            break

        var parent = parent_array[n]
        if deaths[cluster] > deaths[parent]:
            deaths[parent] = deaths[cluster]

    for n in range(len(parent_array)):
        point = child_array[n]
        if point >= root_cluster:
            continue

        cluster = parent_array[n]
        lambda_max = deaths[cluster]


        if lambda_max == 0.0 or not isfinite(lambda_array[n]):
            result[point] = 0.0
        else:
            result[point] = (lambda_max - lambda_array[n]) / lambda_max

    return result^


fn get_stability_scores(mut labels: List[Scalar[DType.int]], clusters: Set[Scalar[DType.int]],
                                      stability: Dict[Scalar[DType.int], Float32], max_lambda: Float32) raises -> List[Float32]:

    var result = List[Float32](capacity=len(clusters))
    result.resize(len(clusters), 0)
    @parameter
    fn cmp_int(a: Scalar[DType.int], b: Scalar[DType.int]) -> Bool:
        return a < b
    var sorted_clusters = List[Scalar[DType.int]](clusters)
    sort[cmp_int](
            Span[
                Scalar[DType.int],
                origin_of(sorted_clusters),
            ](ptr=sorted_clusters.unsafe_ptr(), length=len(sorted_clusters)))
    for n, c in enumerate(sorted_clusters):
        var n_ = n
        var cluster_size = 0
        @parameter
        fn v[simd_width: Int](idx: Int) unified {mut}:
            cluster_size += labels._data.load[width=simd_width](idx).eq(n_).reduce_bit_count()
        vectorize[simd_width](len(labels), v)
        if isinf(max_lambda) or max_lambda == 0.0 or cluster_size == 0:
            result[n] = 1.0
        else:
            result[n] = stability[c] / (cluster_size * max_lambda)

    return result^


fn recurse_leaf_dfs(cluster_tree: Dict[String, List[Scalar[DType.int]]], current_node: Scalar[DType.int]) raises -> List[Scalar[DType.int]]:
    var parent_array = cluster_tree['parent'].copy()
    var children_array = cluster_tree['child'].copy()
    var children = List[Scalar[DType.int]]()
    for i in range(len(parent_array)):
        if parent_array[i] == current_node:
            children.append(children_array[i])
    if len(children) == 0:
        return [current_node]
    else:
        var result = List[Scalar[DType.int]]()
        for child in children:
            result.extend(recurse_leaf_dfs(cluster_tree, child))
        return result^


fn get_cluster_tree_leaves(cluster_tree: Dict[String, List[Scalar[DType.int]]]) raises -> List[Scalar[DType.int]]:
    var parent_array = cluster_tree['parent'].copy()
    if len(parent_array) == 0:
        return []
    var root = reduction.min(Span[Scalar[DType.int], origin_of(parent_array)](ptr=parent_array.unsafe_ptr(), length=len(parent_array)))
    return recurse_leaf_dfs(cluster_tree, root)


fn traverse_upwards(cluster_tree: Dict[String, List[Scalar[DType.int]]], lambda_array: List[Float32], cluster_selection_epsilon: Float32, leaf: Scalar[DType.int], allow_single_cluster: Int) raises -> Scalar[DType.int]:
    var parent_array = cluster_tree['parent'].copy()
    var children_array = cluster_tree['child'].copy()

    var root = reduction.min(Span[Scalar[DType.int], origin_of(parent_array)](ptr=parent_array.unsafe_ptr(), length=len(parent_array)))
    var parent = parent_array[children_array.index(leaf)]
    if parent == root:
        if allow_single_cluster:
            return parent
        else:
            return leaf #return node closest to root

    var parent_eps = 1/lambda_array[children_array.index(parent)]
    if parent_eps > cluster_selection_epsilon:
        return parent
    else:
        return traverse_upwards(cluster_tree, lambda_array, cluster_selection_epsilon, parent, allow_single_cluster)


fn epsilon_search(leaves: Set[Scalar[DType.int]], cluster_tree: Dict[String, List[Scalar[DType.int]]], lambda_array: List[Float32], cluster_selection_epsilon: Float32, allow_single_cluster: Int) raises -> Set[Scalar[DType.int]]:
    var selected_clusters = List[Scalar[DType.int]]()
    var processed = List[Scalar[DType.int]]()

    for leaf in leaves:
        var eps = 1/lambda_array[cluster_tree['child'].index(leaf)]
        if eps < cluster_selection_epsilon:
            if leaf not in processed:
                epsilon_child = traverse_upwards(cluster_tree, lambda_array, cluster_selection_epsilon, leaf, allow_single_cluster)
                selected_clusters.append(epsilon_child)

                for sub_node in bfs_from_cluster_tree(cluster_tree, epsilon_child):
                    if sub_node != epsilon_child:
                        processed.append(sub_node)
        else:
            selected_clusters.append(leaf)

    return Set(selected_clusters)


fn simplify_hierarchy(mut condensed_tree: Dict[String, List[Scalar[DType.int]]], mut lambda_array: List[Float32], persistence_threshold: Float32) raises -> Tuple[Dict[String, List[Scalar[DType.int]]], List[Float32]]:
    """Remove leaves with persistence below threshold."""
    var n_points = condensed_tree['parent'][0]
    var cluster_tree = Dict[String, List[Scalar[DType.int]]]()
    cluster_tree['parent'] = List[Scalar[DType.int]]()
    cluster_tree['child'] = List[Scalar[DType.int]]()
    var cluster_lambda_array = List[Float32]()
    for i in range(len(condensed_tree['child'])):
        if condensed_tree['child'][i] >= n_points:
            cluster_tree['parent'].append(condensed_tree['parent'][i])
            cluster_tree['child'].append(condensed_tree['child'][i])
            cluster_lambda_array.append(lambda_array[i])
    var n_nodes = cluster_tree['child'][len(cluster_tree['child']) - 1] + 1

    # track state and changes
    var leaf_indicator = List[Bool](capacity=Int(n_nodes - n_points))
    leaf_indicator.resize(Int(n_nodes - n_points), True)
    var indices = List[Scalar[DType.int]](capacity=len(cluster_tree['parent']))
    indices.resize(len(cluster_tree['parent']), 0)
    @parameter
    fn v1[simd_width: Int](idx: Int) unified {mut}:
        try:
            indices._data.store[width=simd_width](idx, cluster_tree['parent']._data.load[width=simd_width](idx) - n_points)
        except e:
            print(e)
    vectorize[simd_width](len(indices), v1)
    for idx in indices:
        leaf_indicator[idx] = False

    var max_births = List[Float32](capacity=Int(n_nodes - n_points))
    max_births.resize(Int(n_nodes - n_points), -math.inf[DType.float32]())
    indices = List[Scalar[DType.int]](capacity=len(condensed_tree['parent']))
    indices.resize(len(condensed_tree['parent']), 0)
    @parameter
    fn v2[simd_width: Int](idx: Int) unified {mut}:
        try:
            indices._data.store[width=simd_width](idx, condensed_tree['parent']._data.load[width=simd_width](idx) - n_points)
        except e:
            print(e)
    vectorize[simd_width](len(indices), v2)
    for i, idx in enumerate(indices):
        max_births[idx] = lambda_array[i]

    var parent_map = arange(Int(n_points), Int(n_nodes))
    var lambda_map = Dict[Scalar[DType.int], Float32]()

    # reverse order guarantees children are processed before parents
    for idx in range(len(cluster_tree['parent']) - 1, 0, -2):
        var parent = cluster_tree['parent'][idx]
        var children = cluster_tree['child'][idx - 1 : idx + 1]
        var death = cluster_lambda_array[idx]
        var node_indices = List[Scalar[DType.int]](capacity=len(children))
        node_indices.resize(len(children), 0)
        @parameter
        fn v[simd_width: Int](idx: Int) unified {mut}:
            node_indices._data.store[width=simd_width](idx, children._data.load[width=simd_width](idx) - n_points)
        vectorize[simd_width](len(node_indices), v)
        var births = List[Float32](capacity=len(node_indices))
        births.resize(len(node_indices), 0)
        for i, idx in enumerate(node_indices):
            births[i] = max_births[idx]

        # propagate max density so only leaves can fail the persistence threshold
        max_births[parent - n_points] = max(max_births[parent - n_points], reduction.max(Span[Float32, origin_of(births)](ptr=births.unsafe_ptr(), length=len(births))))
        if (reduction.min(Span[Float32, origin_of(births)](ptr=births.unsafe_ptr(), length=len(births))) - death) >= persistence_threshold:
            continue

        # sibling is the most persistent child
        var sibling_idx = Int(births[0] <= births[1])
        if leaf_indicator[node_indices[sibling_idx]]:
            leaf_indicator[parent - n_points] = True
        else:
            lambda_map[children[1 - sibling_idx]] = death
        for idx in node_indices:
            parent_map[idx] = parent

    # propagate and relabel for consecutive numbering
    var n_skipped = List[Scalar[DType.int]](capacity=len(parent_map))
    n_skipped.resize(len(parent_map), 0)
    for idx, parent in enumerate(parent_map):
        parent_map[idx] = parent_map[parent - n_points]
    for idx, parent in enumerate(parent_map):
        n_skipped[idx] = 1 if parent != (idx + n_points) else 0
        var n_skipped_selected = n_skipped[: Int(parent - n_points)]
        parent_map[idx] = parent - reduction.sum(n_skipped_selected)

    # apply changes
    var keep_mask = List[Bool](capacity=len(condensed_tree['parent']))
    keep_mask.resize(len(condensed_tree['parent']), True)
    for i in range(len(keep_mask)):
        var parent = condensed_tree['parent'][i]
        var child = condensed_tree['child'][i]
        var death = lambda_array[i]

        keep_mask[i] = not n_skipped[max(child - n_points, 0)]
        if not keep_mask[i]:
            continue

        lambda_array[i] = lambda_map.get(parent, death)
        condensed_tree['parent'][i] = parent_map[parent - n_points]
        if child >= n_points:
            condensed_tree['child'][i] = parent_map[child - n_points]

    var result_tree = Dict[String, List[Scalar[DType.int]]]()
    result_tree['parent'] = List[Scalar[DType.int]]()
    result_tree['child'] = List[Scalar[DType.int]]()
    result_tree['child_size'] = List[Scalar[DType.int]]()
    var result_lambda_array = List[Float32]()
    for i in range(len(result_tree['parent'])):
        if keep_mask[i]:
            result_tree['parent'].append(condensed_tree['parent'][i])
            result_tree['child'].append(condensed_tree['child'][i])
            result_tree['child_size'].append(condensed_tree['child_size'][i])
            result_lambda_array.append(lambda_array[i])
    return result_tree^, result_lambda_array^

fn get_clusters(tree: Dict[String, List[Scalar[DType.int]]], mut lambda_array: List[Float32], mut stability: Dict[Scalar[DType.int], Float32],
                        cluster_selection_method: String='eom',
                        allow_single_cluster: Bool=False,
                        match_reference_implementation: Bool=False,
                        cluster_selection_epsilon: Float32=0.0,
                        var max_cluster_size: Scalar[DType.int]=0,
                        cluster_selection_epsilon_max: Float32=math.inf[DType.float32]()) raises -> Tuple[List[Scalar[DType.int]], List[Float32], List[Float32]]:
    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree; This is valid given the
    # current implementation above, so don't change that ... or
    # if you do, change this accordingly!
    var node_list = List[Scalar[DType.int]]()
    for key in stability.keys():
        node_list.append(key)
    @parameter
    fn cmp_int[ascending: Bool = True](a: Scalar[DType.int], b: Scalar[DType.int]) -> Bool:
        @parameter
        if ascending:
            return a < b
        else:
            return a > b
    sort[cmp_int[False]](
            Span[
                Scalar[DType.int],
                origin_of(node_list),
            ](ptr=node_list.unsafe_ptr(), length=len(node_list)))
    if not allow_single_cluster:
        node_list = List[Scalar[DType.int]](node_list[:len(node_list)-1])

    var cluster_tree = Dict[String, List[Scalar[DType.int]]]()
    cluster_tree['parent'] = List[Scalar[DType.int]]()
    cluster_tree['child'] = List[Scalar[DType.int]]()
    cluster_tree['child_size'] = List[Scalar[DType.int]]()
    cluster_lambda_array = List[Float32]()
    var max_child_val: Scalar[DType.int] = -2
    var size_sum: Scalar[DType.int] = 0
    for i, size in enumerate(tree['child_size']):
        if size > 1:
            cluster_tree['parent'].append(tree['parent'][i])
            if tree['parent'][i] == node_list[len(node_list) - 1]:
                size_sum += size
            cluster_tree['child'].append(tree['child'][i])
            cluster_tree['child_size'].append(size)
            cluster_lambda_array.append(lambda_array[i])
        elif size == 1:
            if tree['child'][i] > max_child_val:
                max_child_val = tree['child'][i]
    var is_cluster = {cluster: True for cluster in node_list}
    var num_points = max_child_val + 1
    var max_lambda = reduction.max(Span[Float32, origin_of(lambda_array)](ptr=lambda_array.unsafe_ptr(), length=len(lambda_array)))
    var deaths = max_lambdas(tree, lambda_array)

    if max_cluster_size <= 0:
        max_cluster_size = num_points + 1  # Set to a value that will never be triggered
    var cluster_sizes = {child: child_size for child, child_size
                 in zip(cluster_tree['child'], cluster_tree['child_size'].copy())}
    var node_eps = {child: 1/l for child, l
                 in zip(cluster_tree['child'], cluster_lambda_array)}
    if allow_single_cluster:
        # Compute cluster size for the root node
        cluster_sizes[node_list[len(node_list) - 1]] = size_sum
        var max_value = -math.inf[DType.float32]()
        @parameter
        fn v[simd_width: Int](idx: Int) unified {mut}:
            var max_in_simd = (1.0 / lambda_array._data.load[width=simd_width](idx)).reduce_max()
            if max_in_simd > max_value:
                max_value = max_in_simd
        vectorize[Matrix.simd_width](len(lambda_array), v)
        node_eps[node_list[len(node_list) - 1]] = max_value

    if cluster_selection_method == 'eom':
        for node in node_list:
            var child_selection = List[Scalar[DType.int]]()
            for i, parent in enumerate(cluster_tree['parent']):
                if parent == node:
                    child_selection.append(cluster_tree['child'][i])
            var subtree = [stability[child] for child in child_selection]
            var subtree_stability = reduction.sum(Span[Float32, origin_of(subtree)](ptr=subtree.unsafe_ptr(), length=len(subtree)))
            _ = subtree
            if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size or node_eps[node] > cluster_selection_epsilon_max:
                is_cluster[node] = False
                stability[node] = subtree_stability
            else:
                for sub_node in bfs_from_cluster_tree(cluster_tree, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False

        if cluster_selection_epsilon != 0.0 and len(cluster_tree['parent']) > 0:
            var eom_clusters = [c for c in is_cluster.copy() if is_cluster[c]]
            # first check if eom_clusters only has root node, which skips epsilon check.
            if (len(eom_clusters) == 1 and eom_clusters[0] == reduction.min(Span[Scalar[DType.int], origin_of(cluster_tree['parent'])](ptr=cluster_tree['parent'].unsafe_ptr(), length=len(cluster_tree['parent'])))):
                var selected_clusters = List[Scalar[DType.int]]()
                if allow_single_cluster:
                    selected_clusters = eom_clusters^
                for c in is_cluster:
                    if c in selected_clusters:
                        is_cluster[c] = True
                    else:
                        is_cluster[c] = False
            else:
                var selected_clusters = epsilon_search(Set(eom_clusters), cluster_tree, cluster_lambda_array, cluster_selection_epsilon, Int(allow_single_cluster))
                for c in is_cluster:
                    if c in selected_clusters:
                        is_cluster[c] = True
                    else:
                        is_cluster[c] = False

    elif cluster_selection_method == 'leaf':
        var leaves = Set(get_cluster_tree_leaves(cluster_tree))
        if len(leaves) == 0:
            for c in is_cluster:
                is_cluster[c] = False
            is_cluster[reduction.min(Span[Scalar[DType.int], origin_of(tree['parent'])](ptr=tree['parent'].unsafe_ptr(), length=len(tree['parent'])))] = True

        if cluster_selection_epsilon != 0.0:
            selected_clusters = epsilon_search(leaves, cluster_tree, cluster_lambda_array, cluster_selection_epsilon, Int(allow_single_cluster))
        else:
            selected_clusters = leaves^

        for c in is_cluster:
                if c in selected_clusters:
                    is_cluster[c] = True
                else:
                    is_cluster[c] = False
    else:
        raise Error('Invalid Cluster Selection Method: %s\n'
                         'Should be one of: "eom", "leaf"\n')

    var clusters = Set([c for c in is_cluster.copy() if is_cluster[c]])
    var sorted_clusters = List[Scalar[DType.int]](clusters.copy())
    sort[cmp_int[True]](
            Span[
                Scalar[DType.int],
                origin_of(sorted_clusters),
            ](ptr=sorted_clusters.unsafe_ptr(), length=len(sorted_clusters)))
    var cluster_map = {c: n for n, c in enumerate(sorted_clusters)}
    var reverse_cluster_map = {e.value: e.key for e in cluster_map.items()}

    var labels = do_labelling(tree, lambda_array, clusters, cluster_map,
                          Int(allow_single_cluster), cluster_selection_epsilon,
                          Int(match_reference_implementation))
    var probs = get_probabilities(tree, lambda_array, reverse_cluster_map, labels, deaths)
    stabilities = get_stability_scores(labels, clusters, stability, max_lambda)

    return labels^, probs^, stabilities^
