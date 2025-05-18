# I reimplemented the kd-tree implementation in C++ (and Fortran) by Matthew B. Kennel (https://github.com/jmhodges/kdtree2/) with some modifications.

from mojmelo.utils.Matrix import Matrix
from memory import UnsafePointer
from buffer import NDBuffer
import math

@always_inline
fn Abs(val: Float32) -> Float32:
    return abs(val)

@always_inline
fn Squared(val: Float32) -> Float32:
    return val ** 2

@value
struct interval:
    var lower: Float32
    var upper: Float32

    fn __init__(out self, lower: Float32, upper: Float32):
        self.lower = lower
        self.upper = upper

@value
struct KDTreeResult:
    var dis: Float32  # its square Euclidean distance
    var idx: Int    # which neighbor was found

    fn __init__(out self, dis: Float32, idx: Int):
        self.dis = dis
        self.idx = idx

    @always_inline
    fn __gt__(self, rhs: Self) -> Bool:
        return self.dis > rhs.dis

    @always_inline
    fn __lt__(self, rhs: Self) -> Bool:
        return self.dis < rhs.dis

    fn __le__(self, rhs: Self) capturing -> Bool:
        return self.dis <= rhs.dis

@value
struct KDTreeResultVector:
    var _self: List[KDTreeResult]
    
    fn __init__(out self):
        self._self = List[KDTreeResult]()

    @always_inline
    fn __getitem__(self, index: Int) -> KDTreeResult:
        return self._self[index]

    @always_inline
    fn __setitem__(mut self, index: Int, val: KDTreeResult):
        self._self[index] = val

    @always_inline
    fn __len__(self) -> Int:
        return len(self._self)

    fn append_heap(mut self):
        var child = len(self) - 1; # Last element
        var parent = (child - 1) // 2;  # Parent of the last element

        # Bubble up the new element to its correct position in the heap
        while child > 0 and self[child] > self[parent]:
            self._self.swap_elements(child, parent)  # Swap the child and parent
            child = parent;                  # Move the child pointer up
            parent = (child - 1) // 2        # Update the parent pointer

    fn append_element_and_heapify(mut self, e: KDTreeResult):
        self._self.append(e)
        self.append_heap()

    fn pop_heap(mut self):
        self._self.swap_elements(0, len(self) - 1)

        var parent = 0
        var size = 0
        while True:
            var left_child = 2 * parent + 1
            var right_child = 2 * parent + 2
            var largest = parent
            # Check if left child is larger than parent
            if left_child < size and self[left_child] > self[largest]:
                largest = left_child
            # Check if right child is larger than the largest so far
            if right_child < size and self[right_child] > self[largest]:
                largest = right_child
            # If the largest is still the parent, heap is valid
            if largest == parent:
                break

            # Swap the parent with the largest child
            self._self.swap_elements(parent, largest)
            parent = largest

    fn max_value(self) -> Float32:
        return self[0].dis

    fn replace_maxpri_elt_return_new_maxpri(mut self, e: KDTreeResult) -> Float32:
        self.pop_heap()
        _ = self._self.pop()
        self._self.append(e) # insert new
        self.append_heap()  # and heapify.
        return self.max_value()

struct SearchRecord:
    var qv: UnsafePointer[Float32]
    var dim: Int
    var rearrange: Bool
    var nn: UInt
    var ballsize: Float32
    var centeridx: Int
    var correltime: Int
    var result: UnsafePointer[KDTreeResultVector]
    var data: UnsafePointer[Matrix] 
    var ind: UnsafePointer[List[Int]]

    fn __init__(out self, qv_in: NDBuffer[type=DType.float32, rank=1], tree_in: KDTree, result_in: KDTreeResultVector):  
        self.qv = qv_in.data
        self.result = UnsafePointer(to=result_in)
        self.data = UnsafePointer(to=tree_in._data)
        self.ind = UnsafePointer(to=tree_in.ind) 
        self.dim = tree_in.dim
        self.rearrange = tree_in.rearrange
        self.ballsize = math.inf[DType.float32]() 
        self.nn = 0
        self.centeridx = self.correltime = 0

@always_inline
fn dis_from_bnd(x: Float32, amin: Float32, amax: Float32) -> Float32:
    if x > amax:
        return x-amax
    if x < amin:
        return amin-x
    return 0.0

@value
struct KDTreeNode:
    var cut_dim: Int # dimension to cut
    var cut_val: Float32
    var cut_val_left: Float32
    var cut_val_right: Float32
    var l: Int # extents in index array for searching
    var u: Int
    var box: List[interval] # [min,max] of the box enclosing all points
    var left: UnsafePointer[KDTreeNode]
    var right: UnsafePointer[KDTreeNode]
    var metric: fn(Float32) -> Float32

    fn __init__(out self, dim: Int, metric: fn(Float32) -> Float32):
        self.cut_dim = self.l = self.u = 0
        self.cut_val = self.cut_val_left = self.cut_val_right = 0.0
        self.box = List[interval](capacity=dim)
        self.box.resize(dim, interval(0.0, 0.0))
        self.left = UnsafePointer[KDTreeNode]()
        self.right = UnsafePointer[KDTreeNode]()
        self.metric = metric

    fn search(self, mut sr: SearchRecord): # recursive innermost core routine for searching.. 
        if not (self.left or self.right):
            # we are on a terminal node
            if sr.nn == 0:
                self.process_terminal_node_fixedball(sr)
            else:
                self.process_terminal_node(sr)
        else:
            var ncloser: UnsafePointer[KDTreeNode]
            var nfarther: UnsafePointer[KDTreeNode]

            var extra: Float32
            var qval = sr.qv[self.cut_dim]
            # value of the wall boundary on the cut dimension. 
            if qval < self.cut_val:
                ncloser = self.left
                nfarther = self.right
                extra = self.cut_val_right-qval
            else:
                ncloser = self.right
                nfarther = self.left
                extra = qval-self.cut_val_left

            if ncloser:
                ncloser[].search(sr)

            if nfarther and self.metric(extra) < sr.ballsize:
                # first cut
                if nfarther[].box_in_search_range(sr):
                    nfarther[].search(sr)

    @always_inline
    fn box_in_search_range(self, sr: SearchRecord) -> Bool:
        # does the bounding box, represented by minbox[*],maxbox[*]
        # have any point which is within 'sr.ballsize' to 'sr.qv'??
        var dim = sr.dim
        var dis2: Float32 = 0.0 
        var ballsize = sr.ballsize 
        for i in range(dim):
            dis2 += self.metric(dis_from_bnd(sr.qv[i],self.box[i].lower,self.box[i].upper))
            if dis2 > ballsize:
                return False
        return True

    # for processing final buckets. 
    fn process_terminal_node(self, mut sr: SearchRecord):
        var centeridx  = sr.centeridx
        var correltime = sr.correltime
        var nn = sr.nn
        var dim = sr.dim
        var ballsize = sr.ballsize
        var rearrange = sr.rearrange
        var data = sr.data

        for i in range(self.l, self.u + 1):
            var indexofi: Int  # sr.ind[i]; 
            var dis: Float32
            var early_exit: Bool

            if rearrange:
                early_exit = False
                dis = 0.0
                for k in range(dim):
                    dis += self.metric(data[].load[1](i, k) - sr.qv[k])
                    if dis > ballsize:
                        early_exit=True
                        break
                if early_exit:
                    continue
                indexofi = sr.ind[][i]
            else:
                # but if we are not using the rearranged data, then
                # we must always 
                indexofi = sr.ind[][i]
                early_exit = False
                dis = 0.0
                for k in range(dim):
                    dis += self.metric(data[].load[1](indexofi, k) - sr.qv[k])
                    if dis > ballsize:
                        early_exit= True 
                        break
                if early_exit:
                    continue
            if centeridx > 0:
                # we are doing decorrelation interval
                if abs(indexofi-centeridx) < correltime:
                    continue # skip this point. 

            # here the point must be added to the list.
            # two choices for any point.  The list so far is either
            # undersized, or it is not.
            if len(sr.result[]) < nn:
                var e = KDTreeResult(dis, indexofi)
                sr.result[].append_element_and_heapify(e)
                if len(sr.result[]) == nn:
                    ballsize = sr.result[].max_value()
            else:
                # if we get here then the current node, has a squared 
                # distance smaller
                # than the last on the list, and belongs on the list.
                var e = KDTreeResult(dis, indexofi)
                ballsize = sr.result[].replace_maxpri_elt_return_new_maxpri(e)
        sr.ballsize = ballsize

    fn process_terminal_node_fixedball(self, sr: SearchRecord):
        var centeridx  = sr.centeridx
        var correltime = sr.correltime
        var dim = sr.dim
        var ballsize = sr.ballsize
        var rearrange = sr.rearrange
        var data = sr.data

        for i in range(self.l, self.u + 1):
            var indexofi = sr.ind[][i]
            var dis: Float32
            var early_exit: Bool

            if rearrange:
                early_exit = False
                dis = 0.0
                for k in range(dim):
                    dis += self.metric(data[].load[1](i, k) - sr.qv[k])
                    if dis > ballsize:
                        early_exit=True
                        break
                if early_exit:
                    continue
                # why do we do things like this?  because if we take an early
                # exit (due to distance being too large) which is common, then
                # we need not read in the actual point index, thus saving main
                # memory bandwidth.  If the distance to point is less than the
                # ballsize, though, then we need the index.
                indexofi = sr.ind[][i]
            else:
                # but if we are not using the rearranged data, then
                # we must always 
                indexofi = sr.ind[][i]
                early_exit = False
                dis = 0.0
                for k in range(dim):
                    dis += self.metric(data[].load[1](indexofi, k) - sr.qv[k])
                    if dis > ballsize:
                        early_exit= True
                        break
                if early_exit:
                    continue
            
            if centeridx > 0:
                # we are doing decorrelation interval
                if abs(indexofi-centeridx) < correltime:
                    continue # skip this point.
            var e = KDTreeResult(dis, indexofi)
            sr.result[]._self.append(e)

@value
struct KDTree[sort_results: Bool = False, rearrange: Bool = True]:
    var _data: Matrix
    var N: Int   # number of data points
    var dim: Int
    var root: UnsafePointer[KDTreeNode] # the root pointer
    var ind: List[Int] 
    # the index for the tree leaves.  Data in a leaf with bounds [l,u] are
    # in  'the_data[ind[l],*] to the_data[ind[u],*]
    var metric: fn(Float32) -> Float32
    alias bucketsize = 12

    fn __init__(out self, X: Matrix, metric: String = 'euc', *, build: Bool = True):
        self._data = X
        self.N = self._data.height
        self.dim = self._data.width
        self.root = UnsafePointer[KDTreeNode]()
        self.ind = List[Int](capacity=self.N)
        self.ind.resize(self.N, 0)
        if metric.lower() == 'man':
            self.metric = Abs
        else:
            self.metric = Squared

        if build:
            self.build_tree()

            if rearrange:
                var rearranged_data = Matrix(self.N, self.dim)
        
                # permute the data for it.
                for i in range(self.N):
                    for j in range(self.dim):
                        rearranged_data.store[1](i, j, self._data.load[1](self.ind[i], j))
                self._data = rearranged_data^

    fn __moveinit__(out self, owned existing: Self):
        self._data = existing._data^
        self.N = existing.N
        self.dim = existing.dim
        self.root = existing.root
        self.ind = existing.ind^
        self.metric = existing.metric
        existing.N = existing.dim = 0
        existing.root = UnsafePointer[KDTreeNode]()

    fn build_tree(mut self): # builds the tree.  Used upon construction
        for i in range(self.N):
            self.ind[i] = i
        self.root = self.build_tree_for_range(0, self.N-1, UnsafePointer[KDTreeNode]())

    fn build_tree_for_range(mut self, l: Int, u: Int, parent: UnsafePointer[KDTreeNode]) -> UnsafePointer[KDTreeNode]:
        # recursive function to build 
        var node = UnsafePointer[KDTreeNode].alloc(1)
        node.init_pointee_move(KDTreeNode(self.dim, self.metric))
        # the newly created node. 

        if u<l:
            return UnsafePointer[KDTreeNode]() # no data in this node. 
      
        if (u-l) <= self.bucketsize:
            # create a terminal node. 

            # always compute true bounding box for terminal node. 
            for i in range(self.dim):
                self.spread_in_coordinate(i,l,u,node[].box[i])
    
            node[].cut_dim = 0
            node[].cut_val = 0.0
            node[].l = l
            node[].u = u
            node[].left = node[].right = UnsafePointer[KDTreeNode]()

        else:
            # Compute an APPROXIMATE bounding box for this node.
            # if parent == NULL, then this is the root node, and
            # we compute for all dimensions.
            # Otherwise, we copy the bounding box from the parent for
            # all coordinates except for the parent's cut dimension.  
            # That, we recompute ourself.
            var c = -1
            var maxspread: Float32 = 0.0
            var m: Int 

            for i in range(self.dim):
                if (not parent) or (parent[].cut_dim == i):
                    self.spread_in_coordinate(i,l,u,node[].box[i])
                else:
                    node[].box[i] = parent[].box[i]
                var spread = node[].box[i].upper - node[].box[i].lower 
                if spread > maxspread:
                    maxspread = spread
                    c=i

            # now, c is the identity of which coordinate has the greatest spread

            var sum: Float32 = 0.0
            var average: Float32

            for k in range(l, u+1):
                sum += self._data.load[1](self.ind[k], c)
            average = sum / (u-l+1)
	
            m = self.select_on_coordinate_value(c,average,l,u)


            # move the indices around to cut on dim 'c'.
            node[].cut_dim=c
            node[].l = l
            node[].u = u

            node[].left = self.build_tree_for_range(l,m,node)
            node[].right = self.build_tree_for_range(m+1,u,node)

            if not node[].right:
                for i in range(self.dim):
                    node[].box[i] = node[].left[].box[i] 
                node[].cut_val = node[].left[].box[c].upper
                node[].cut_val_left = node[].cut_val_right = node[].cut_val
            elif not node[].left:
                for i in range(self.dim):
                    node[].box[i] = node[].right[].box[i]
                node[].cut_val = node[].right[].box[c].upper
                node[].cut_val_left = node[].cut_val_right = node[].cut_val
            else:
                node[].cut_val_right = node[].right[].box[c].lower
                node[].cut_val_left  = node[].left[].box[c].upper
                node[].cut_val = (node[].cut_val_left + node[].cut_val_right) / 2.0
                # now recompute true bounding box as union of subtree boxes.
                # This is now faster having built the tree, being logarithmic in
                # N, not linear as would be from naive method.
                for i in range(self.dim):
                    node[].box[i].upper = max(node[].left[].box[i].upper,
                                node[].right[].box[i].upper)
                    
                    node[].box[i].lower = min(node[].left[].box[i].lower,
                                node[].right[].box[i].lower)
        return node

    fn spread_in_coordinate(self, c: Int, l: Int, u: Int, mut interv: interval):
        # return the minimum and maximum of the indexed data between l and u in
        var smin: Float32
        var smax: Float32
        var lmin: Float32
        var lmax: Float32
        var i = l+2

        smin = self._data.load[1](self.ind[l], c)
        smax = smin
        
        # process two at a time.
        while i<= u:
            lmin = self._data.load[1](self.ind[i-1], c)
            lmax = self._data.load[1](self.ind[i], c)

            if lmin > lmax:
                swap(lmin,lmax)

            if smin > lmin:
                smin = lmin
            if smax <lmax:
                smax = lmax
            i += 2

        # is there one more element? 
        if i == u+1:
            var last = self._data.load[1](self.ind[u], c)
            if smin>last:
                smin = last
            if smax<last:
                smax = last
        interv.lower = smin
        interv.upper = smax

    fn select_on_coordinate(mut self, c: Int, k: Int, owned l: Int, owned u: Int):
        #  Move indices in ind[l..u] so that the elements in [l .. k] 
        #  are less than the [k+1..u] elmeents, viewed across dimension 'c'. 
        while l < u:
            var t = self.ind[l]
            var m = l

            for i in range(l+1, u+1):
                if self._data.load[1](self.ind[i], c) < self._data.load[1](t, c):
                    m += 1
                    self.ind.swap_elements(i, m)
            self.ind.swap_elements(l, m)

            if (m <= k):
                l = m+1
            if (m >= k):
                u = m-1

    fn select_on_coordinate_value(mut self, c: Int, alpha: Float32, l: Int, u: Int) -> Int:
        #  Move indices in ind[l..u] so that the elements in [l .. return]
        #  are <= alpha, and hence are less than the [return+1..u]
        #  elmeents, viewed across dimension 'c'.
        var lb = l
        var ub = u

        while lb < ub:
            if self._data.load[1](self.ind[lb], c) <= alpha:
                lb += 1 # good where it is.
            else:
                self.ind.swap_elements(lb, ub)
                ub -= 1

        # here ub=lb
        if self._data.load[1](self.ind[lb], c) <= alpha:
            return lb
        return lb-1

    fn n_nearest(self, qv: NDBuffer[type=DType.float32, rank=1], nn: Int, mut result: KDTreeResultVector):
        var sr = SearchRecord(qv,self,result)

        result._self.clear()

        sr.centeridx = -1
        sr.correltime = 0
        sr.nn = nn

        self.root[].search(sr)

        _ = qv.data

        if (sort_results):
            sort[KDTreeResult.__le__](Span[KDTreeResult, __origin_of(result._self)](ptr= result._self.unsafe_ptr(), length= len(result)))
        
    fn n_nearest_around_point(self, idxin: Int, correltime: Int, nn: Int,
                        mut result: KDTreeResultVector):
        var buf = UnsafePointer[Float32].alloc(self.dim)
        var qv = NDBuffer[type=DType.float32, rank=1](buf, self.dim) #  query vector
        result._self.clear()

        for i in range(self.dim):
            qv[i] = self._data.load[1](idxin, i) 
        # copy the query vector.
        
        var sr = SearchRecord(qv, self, result)
        # construct the search record.
        sr.centeridx = idxin
        sr.correltime = correltime
        sr.nn = nn; 
        self.root[].search(sr)

        buf.free()

        if (sort_results):
            sort[KDTreeResult.__le__](Span[KDTreeResult, __origin_of(result._self)](ptr= result._self.unsafe_ptr(), length= len(result)))


    fn r_nearest(self, qv: NDBuffer[type=DType.float32, rank=1], r2: Float32, mut result: KDTreeResultVector):
        # search for all within a ball of a certain radius
        var sr = SearchRecord(qv,self,result)

        result._self.clear()

        sr.centeridx = -1
        sr.correltime = 0
        sr.nn = 0
        sr.ballsize = r2

        self.root[].search(sr)

        _ = qv.data

        if (sort_results):
            sort[KDTreeResult.__le__](Span[KDTreeResult, __origin_of(result._self)](ptr= result._self.unsafe_ptr(), length= len(result)))

    fn r_count(self, qv: NDBuffer[type=DType.float32, rank=1], r2: Float32) -> Int:
        # search for all within a ball of a certain radius
        var result = KDTreeResultVector()
        sr = SearchRecord(qv,self,result)

        sr.centeridx = -1
        sr.correltime = 0
        sr.nn = 0
        sr.ballsize = r2
        
        self.root[].search(sr)
        _ = qv.data

        return len(result)

    fn r_nearest_around_point(mut self, idxin: Int, correltime: Int, r2: Float32,
                        mut result: KDTreeResultVector):
        var buf = UnsafePointer[Float32].alloc(self.dim)
        var qv = NDBuffer[type=DType.float32, rank=1](buf, self.dim) #  query vector

        result._self.clear()

        for i in range(self.dim):
            qv[i] = self._data.load[1](idxin, i) 
        
        var sr = SearchRecord(qv, self, result)
        # construct the search record.
        sr.centeridx = idxin
        sr.correltime = correltime
        sr.ballsize = r2
        sr.nn = 0
        self.root[].search(sr)

        buf.free()

        if (sort_results):
            sort[KDTreeResult.__le__](Span[KDTreeResult, __origin_of(result._self)](ptr= result._self.unsafe_ptr(), length= len(result)))

    fn r_count_around_point(self, idxin: Int, correltime: Int, r2: Float32) -> Int:
        var buf = UnsafePointer[Float32].alloc(self.dim)
        var qv = NDBuffer[type=DType.float32, rank=1](buf, self.dim) #  query vector

        for i in range(self.dim):
            qv[i] = self._data.load[1](idxin, i) 

        var result = KDTreeResultVector()
        var sr = SearchRecord(qv, self, result)
        # construct the search record.
        sr.centeridx = idxin
        sr.correltime = correltime
        sr.ballsize = r2
        sr.nn = 0
        self.root[].search(sr)
        buf.free()

        return len(result)

    fn __del__(owned self):
        if self.root:
            delTree(self.root)

fn delTree(node: UnsafePointer[KDTreeNode]):
    if node[].left:
        delTree(node[].left)
    if node[].right:
        delTree(node[].right)
    node.free()
