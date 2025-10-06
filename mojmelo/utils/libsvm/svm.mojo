from memory import memcpy, memset_zero
from .svm_node import svm_node
from .svm_parameter import svm_parameter
from sys import size_of
import math

alias TAU = 1e-12

@always_inline
fn powi(base: Float64, times: Int) -> Float64:
    var tmp = base
    var ret = 1.0

    var t = times
    while t>0:
        if t%2==1:
            ret *= tmp
        tmp = tmp * tmp
        t/=2
    return ret

@always_inline
fn dot(var px: UnsafePointer[svm_node], var py: UnsafePointer[svm_node]) -> Float64:
    var sum = 0.0
    while px[].index != -1 and py[].index != -1:
        if px[].index == py[].index:
            sum += px[].value * py[].value
            px += 1
            py += 1
        else:
            if px[].index > py[].index:
                py += 1
            else:
                px += 1

    return sum

@always_inline
fn kernel_linear(k: Kernel, i: Int, j: Int) -> Float64:
    return dot(k.x[i],k.x[j])
@always_inline
fn kernel_poly(k: Kernel, i: Int, j: Int) -> Float64:
    return powi(k.gamma*dot(k.x[i],k.x[j])+k.coef0,k.degree)
@always_inline
fn kernel_rbf(k: Kernel, i: Int, j: Int) -> Float64:
    return math.exp(-k.gamma*(k.x_square[i]+k.x_square[j]-2*dot(k.x[i],k.x[j])))
@always_inline
fn kernel_sigmoid(k: Kernel, i: Int, j: Int) -> Float64:
    return math.tanh(k.gamma*dot(k.x[i],k.x[j])+k.coef0)
@always_inline
fn kernel_precomputed(k: Kernel, i: Int, j: Int) -> Float64:
    return k.x[i][Int(k.x[j][0].value)].value

struct head_t(Copyable, Movable):
    var prev: UnsafePointer[head_t]
    var next: UnsafePointer[head_t]	# a cicular list
    var data: UnsafePointer[Float32]
    var _len: Int		# data[0,len) is cached in this entry

    @always_inline
    fn __init__(out self, p: UnsafePointer[head_t], n: UnsafePointer[head_t], _len: Int):
        self.prev = p
        self.next = n
        self._len = _len
        self.data = UnsafePointer[Float32].alloc(_len)
        memset_zero(self.data, self._len)

    @always_inline
    fn __del__(deinit self):
        if self.data:
            self.data.free()

# Kernel Cache
#
# l is the number of total data items
# size is the cache size limit in bytes
struct Cache:
    var l: Int
    var size: UInt
    var head: UnsafePointer[head_t]
    var lru_head: head_t

    fn __init__(out self, l: Int, size: UInt):
        self.l = l
        self.size = (size - (self.l * size_of[head_t]())) // 4
        self.head = UnsafePointer[head_t].alloc(self.l)
        for i in range(self.l):
            (self.head + i).init_pointee_move(head_t(UnsafePointer[head_t](), UnsafePointer[head_t](), 0)) # initialized to 0
        self.size = max(self.size, UInt(2) * UInt(l))  # cache must be large enough for two columns
        self.lru_head = head_t(UnsafePointer[head_t](), UnsafePointer[head_t](), 0)
        self.lru_head.next = self.lru_head.prev = UnsafePointer(to=self.lru_head)

    fn __del__(deinit self):
        var h = self.lru_head.next
        while h != UnsafePointer(to=self.lru_head):
            h.destroy_pointee()
            h = h[].next
        if self.head:
            self.head.free()

    fn lru_delete(self, h: UnsafePointer[head_t]):
        # delete from current location
        h[].prev[].next = h[].next
        h[].next[].prev = h[].prev

    fn lru_insert(self, h: UnsafePointer[head_t]):
        # insert to last position
        h[].next = UnsafePointer(to=self.lru_head)
        h[].prev = self.lru_head.prev
        h[].prev[].next = h
        h[].next[].prev = h

    fn get_data(mut self, index: Int, data: UnsafePointer[UnsafePointer[Float32]], var _len: Int) -> Int:
        var h = self.head + index
        if h[]._len:
            self.lru_delete(h)
        var more = _len - h[]._len

        if more > 0:
            # free old space
            while self.size < UInt(more):
                var old = self.lru_head.next
                self.lru_delete(old)
                old[].data.free()
                self.size += old[]._len
                old[].data = UnsafePointer[Float32]()
                old[]._len = 0

            # allocate new space
            var new = UnsafePointer[Float32].alloc(_len)
            memcpy(dest=new, src=h[].data, count=h[]._len)
            h[].data.free()
            h[].data = new
            self.size -= more  # previous while loop guarantees size >= more and subtraction of size_t variable will not underflow
            swap(h[]._len, _len)

        self.lru_insert(h)
        data[] = h[].data
        return _len

    fn swap_index(mut self, var i: Int, var j: Int):
        if i==j:
            return

        if self.head[i]._len:
            self.lru_delete(self.head + i)
        if self.head[j]._len:
            self.lru_delete(self.head + j)
        swap(self.head[i].data,self.head[j].data)
        swap(self.head[i]._len,self.head[j]._len)
        if self.head[i]._len:
            self.lru_insert(self.head + i)
        if self.head[j]._len:
            self.lru_insert(self.head + j)

        if i>j:
            swap(i,j)

        var h = self.lru_head.next
        while h != UnsafePointer(to=self.lru_head):
            if h[]._len > i:
                if(h[]._len > j):
                    swap(h[].data[i],h[].data[j])
                else:
                    # give up
                    self.lru_delete(h)
                    h[].data.free()
                    self.size += h[]._len
                    h[].data = UnsafePointer[Float32]()
                    h[]._len = 0
            h=h[].next

# Kernel evaluation
#
# the static method k_function is for doing single kernel evaluation
# the constructor of Kernel prepares to calculate the l*l kernel matrix
# the member function get_Q is for getting one column from the Q Matrix
#
struct Kernel:
    var x: UnsafePointer[UnsafePointer[svm_node]]
    var x_square: UnsafePointer[Float64]

    # svm_parameter
    var kernel_type: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64

    var kernel_function: fn(Kernel, Int, Int) -> Float64

    @staticmethod
    fn k_function(var x: UnsafePointer[svm_node], var y: UnsafePointer[svm_node], param: svm_parameter) -> Float64:
        if param.kernel_type == svm_parameter.LINEAR:
            return dot(x,y)
        if param.kernel_type == svm_parameter.POLY:
            return powi(param.gamma*dot(x,y)+param.coef0,param.degree)
        if param.kernel_type == svm_parameter.RBF:
            var sum = 0.0
            while x[].index != -1 and y[].index !=-1:
                if x[].index == y[].index:
                    var d = x[].value - y[].value
                    sum += d*d
                    x += 1
                    y += 1
                else:
                    if x[].index > y[].index:
                        sum += y[].value * y[].value
                        y += 1
                    else:
                        sum += x[].value * x[].value
                        x += 1

            while x[].index != -1:
                sum += x[].value * x[].value
                x += 1

            while y[].index != -1:
                sum += y[].value * y[].value
                y += 1

            return math.exp(-param.gamma*sum)
        if param.kernel_type == svm_parameter.SIGMOID:
            return math.tanh(param.gamma*dot(x,y)+param.coef0)
        if param.kernel_type == svm_parameter.PRECOMPUTED:  # x: test (validation), y: SV
            return x[Int(y[].value)].value
        else:
            return 0  # Unreachable

    fn __init__(out self, l: Int, x_: UnsafePointer[UnsafePointer[svm_node]], param: svm_parameter):
        self.kernel_type = param.kernel_type
        self.degree = param.degree
        self.gamma = param.gamma
        self.coef0 = param.coef0

        self.x = UnsafePointer[UnsafePointer[svm_node]].alloc(l)
        memcpy(dest=self.x, src=x_, count=l)

        if self.kernel_type == svm_parameter.RBF:
            self.x_square = UnsafePointer[Float64].alloc(l)
            for i in range(l):
                self.x_square[i] = dot(self.x[i], self.x[i])
        else:
            self.x_square = UnsafePointer[Float64]()

        if self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        if self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        if self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        if self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        if self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear

    fn __del__(deinit self):
        if self.x:
            self.x.free()
        if self.x_square:
            self.x_square.free()

struct SolutionInfo:
    var obj: Float64
    var rho: Float64
    var upper_bound_p: Float64
    var upper_bound_n: Float64
    var r: Float64	# for Solver_NU

# An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
# Solves:
#
#	min 0.5(\alpha^T Q \alpha) + p^T \alpha
#
#		y^T \alpha = \delta
#		y_i = +1 or -1
#		0 <= alpha_i <= Cp for y_i = 1
#		0 <= alpha_i <= Cn for y_i = -1
#
# Given:
#
#	Q, p, y, Cp, Cn, and an initial feasible point \alpha
#	l is the size of vectors and matrices
#	eps is the stopping tolerance
#
# solution will be put in \alpha, objective value will be put in obj
#
struct Solve:
    var active_size: Int
    var y: UnsafePointer[Int8]
    var G: UnsafePointer[Float64]	# gradient of objective function
    alias LOWER_BOUND: Int8 = 0
    alias UPPER_BOUND: Int8 = 1
    alias FREE: Int8 = 2
    var alpha_status: UnsafePointer[Int8]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: UnsafePointer[Float64]
    #QMatrix Q
    var QD: UnsafePointer[Float64]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: UnsafePointer[Float64]
    var active_set: UnsafePointer[Int]
    var G_bar: UnsafePointer[Float64]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    fn get_C(self, i: Int) -> Float64:
        return self.Cp if self.y[i] > 0 else self.Cn

    fn update_alpha_status(self, i: Int):
        if self.alpha[i] >= self.get_C(i):
            self.alpha_status[i] = self.UPPER_BOUND
        elif self.alpha[i] <= 0:
            self.alpha_status[i] = self.LOWER_BOUND
        else:
            self.alpha_status[i] = self.FREE

    fn is_upper_bound(self, i: Int) -> Bool:
        return self.alpha_status[i] == self.UPPER_BOUND
    fn is_lower_bound(self, i: Int) -> Bool:
        return self.alpha_status[i] == self.LOWER_BOUND
    fn is_free(self, i: Int) -> Bool:
        return self.alpha_status[i] == self.FREE

    fn swap_index(self, i: Int, j: Int):
        #Q.swap_index(i,j);
        swap(self.y[i], self.y[j])
        swap(self.G[i], self.G[j])
        swap(self.alpha_status[i], self.alpha_status[j])
        swap(self.alpha[i], self.alpha[j])
        swap(self.p[i], self.p[j])
        swap(self.active_set[i], self.active_set[j])
        swap(self.G_bar[i], self.G_bar[j])

    fn reconstruct_gradient(self):
		# reconstruct inactive elements of G from G_bar and free variables

        if self.active_size == self.l:
            return

        var nr_free = 0

        for j in range(self.active_size, self.l):
            self.G[j] = self.G_bar[j] + self.p[j]

        for j in range(self.active_size):
            if self.is_free(j):
                nr_free += 1

        if 2*nr_free < self.active_size:
            print("\nWARNING: using -h 0 may be faster\n")

        if nr_free*self.l > 2*self.active_size*(self.l-self.active_size):
            for i in range(self.active_size, self.l):
				#var Q_i = Q.get_Q(i,self.active_size)
                for j in range(self.active_size):
                    if self.is_free(j):
                        pass
						#self.G[i] += self.alpha[j] * Q_i[j]
        else:
            for i in range(self.active_size):
                if self.is_free(i):
					#var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha[i]
                    for j in range(self.active_size, self.l):
                        pass
                        #self.G[j] += alpha_i * Q_i[j]

    fn __init__(out self, l: Int, p_: UnsafePointer[Float64], y_: UnsafePointer[Int8],
                alpha_: UnsafePointer[Float64], Cp: Float64, Cn: Float64, eps: Float64, si: SolutionInfo, shrinking: Int):
        self.l = l
        #self.Q = Q
        self.QD = UnsafePointer[Float64]()
        #self.QD = Q.get_QD()
        self.p = UnsafePointer[Float64].alloc(self.l)
        memcpy(dest=self.p, src=p_, count=self.l)
        self.y = UnsafePointer[Int8].alloc(self.l)
        memcpy(dest=self.y, src=y_, count=self.l)
        self.alpha = UnsafePointer[Float64].alloc(self.l)
        memcpy(dest=self.alpha, src=alpha_, count=self.l)
        self.Cp = Cp
        self.Cn = Cn
        self.eps = eps
        self.unshrink = False

		# initialize alpha_status
        self.alpha_status = UnsafePointer[Int8].alloc(self.l)
        for i in range(self.l):
            if self.alpha[i] >= (self.Cp if self.y[i] > 0 else self.Cn):
                self.alpha_status[i] = self.UPPER_BOUND
            elif self.alpha[i] <= 0:
                self.alpha_status[i] = self.LOWER_BOUND
            else:
                self.alpha_status[i] = self.FREE

		# initialize active set (for shrinking)
        self.active_set = UnsafePointer[Int].alloc(self.l)
        for i in range(self.l):
            self.active_set[i] = i
        self.active_size = self.l

		# initialize gradient
        self.G = UnsafePointer[Float64].alloc(self.l)
        self.G_bar = UnsafePointer[Float64].alloc(self.l)
        for i in range(self.l):
            self.G[i] = self.p[i]
            self.G_bar[i] = 0
        for i in range(self.l):
            if not self.is_lower_bound(i):
                #var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha[i]
                for j in range(self.l):
                    pass
                    #self.G[j] += alpha_i*Q_i[j]
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        pass
                        #self.G_bar[j] += self.get_C(i) * Q_i[j]

        # optimization step

        var iter = 0
        var max_iter = max(10000000, Int.MAX if self.l>Int.MAX//100 else 100*self.l)
        var counter = min(self.l,1000)+1

        while iter < max_iter:
            # show progress and do shrinking
            counter -= 1
            if counter == 0:
                counter = min(self.l,1000)
                if shrinking:
                    self.do_shrinking()

            var i: Int
            var j: Int
            if self.select_working_set(i,j)!=0:
                # reconstruct the whole gradient
                self.reconstruct_gradient()
                # reset active set size and check
                self.active_size = self.l
                if self.select_working_set(i,j)!=0:
                    break
                else:
                    counter = 1	# do shrinking next iteration

            iter += 1

            # update alpha[i] and alpha[j], handle bounds carefully

            var Q_i = Q.get_Q(i,active_size)
            var Q_j = Q.get_Q(j,active_size)

            var C_i = get_C(i)
            var C_j = get_C(j)

            var old_alpha_i = alpha[i]
            var old_alpha_j = alpha[j]

            if self.y[i]!=self.y[j]:
                var quad_coef = QD[i]+QD[j]+2*Q_i[j]
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-G[i]-G[j])/quad_coef
                var diff = alpha[i] - alpha[j]
                self.alpha[i] += delta
                self.alpha[j] += delta

                if(diff > 0):
                    if alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = diff
                else:
                    if alpha[i] < 0:
                        self.alpha[i] = 0
                        self.alpha[j] = -diff
                if diff > C_i - C_j:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = C_i - diff
                else:
                    if self.alpha[j] > C_j:
                        self.alpha[j] = C_j
                        self.alpha[i] = C_j + diff
            else:
                var quad_coef = QD[i]+QD[j]-2*Q_i[j]
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (G[i]-G[j])/quad_coef
                var sum = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta

                if sum > C_i:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = sum - C_i
                else:
                    if alpha[j] < 0:
                        alpha[j] = 0
                        alpha[i] = sum
                if sum > C_j:
                    if alpha[j] > C_j:
                        self.alpha[j] = C_j
                        self.alpha[i] = sum - C_j
                else:
                    if self.alpha[i] < 0:
                        self.alpha[i] = 0
                        self.alpha[j] = sum

            # update G

            var delta_alpha_i = self.alpha[i] - old_alpha_i
            var delta_alpha_j = self.alpha[j] - old_alpha_j

            for k in range(self.active_size):
                self.G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar[k] -= C_i * Q_i[k]
                else:
                    for k in range(self.l):
                        self.G_bar[k] += C_i * Q_i[k]

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar[k] -= C_j * Q_j[k];
                else:
                    for k in range(self.l):
                        self.G_bar[k] += C_j * Q_j[k];

        if iter >= max_iter:
            if(active_size < self.l):
                # reconstruct the whole gradient to calculate objective value
                self.reconstruct_gradient()
                self.active_size = self.l
            print("\nWARNING: reaching max number of iterations\n")

        # calculate rho

        si[].rho = self.calculate_rho()

        # calculate objective value
        var v = 0
        for i in range(self.l):
            v += self.alpha[i] * (self.G[i] + self.p[i]);

        si[].obj = v/2

        # put back the solution

        for i in range(self.l):
            self.alpha_[self.active_set[i]] = self.alpha[i]

        # juggle everything back

        #for i in range(self.l):
            #while self.active_set[i] != i:
                #self.swap_index(i,self.active_set[i])
        #       # or Q.swap_index(i,self.active_set[i])


        si[].upper_bound_p = Cp
        si[].upper_bound_n = Cn

        self.p.free()
        self.y.free()
        self.alpha.free()
        self.alpha_status.free()
        self.active_set.free()
        self.G.free()
        self.G_bar.free()

    # return 1 if already optimal, return 0 otherwise
    fn select_working_set(self, mut out_i: Int, mut out_j: Int) -> Int:
        # return i,j such that
        # i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        # j: minimizes the decrease of obj value
        #    (if quadratic coefficeint <= 0, replace it with tau)
        #    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        var Gmax = -math.inf[DType.float64]()
        var Gmax2 = -math.inf[DType.float64]()
        var Gmax_idx = -1
        var Gmin_idx = -1
        var obj_diff_min = math.inf[DType.float64]()

        for t in range(self.active_size):
            if self.y[t]== 1:
                if not self.is_upper_bound(t):
                    if -self.G[t] >= Gmax:
                        Gmax = -self.G[t]
                        Gmax_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G[t] >= Gmax:
                        Gmax = self.G[t]
                        Gmax_idx = t

        var i = Gmax_idx
        var Q_i = UnsafePointer[Float32]()
        if i != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_i = Q.get_Q(i,active_size)

        for j in range(self.active_size):
            if(self.y[j]==+1):
                if not self.is_lower_bound(j):
                    var grad_diff=Gmax+self.G[j]
                    if self.G[j] >= Gmax2:
                        Gmax2 = self.G[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = QD[i]+QD[j]-2.0*self.y[i]*Q_i[j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    var grad_diff= Gmax-self.G[j]
                    if -self.G[j] >= Gmax2:
                        Gmax2 = -self.G[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff

        if Gmax+Gmax2 < eps or Gmin_idx == -1:
            return 1

        out_i = Gmax_idx
        out_j = Gmin_idx
        return 0

    fn be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64) -> Bool:
        if self.is_upper_bound(i):
            if self.y[i]==1:
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax2
        elif self.is_lower_bound(i):
            if self.y[i]==1:
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax1
        else:
            return False

    fn do_shrinking(mut self):
        var Gmax1 = -math.inf[DType.float64]()		# max { -y_i * grad(f)_i | i in I_up(\alpha) }
        var Gmax2 = -math.inf[DType.float64]()		# max { y_i * grad(f)_i | i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if self.y[i]==1:
                if not self.is_upper_bound(i):
                    if -self.G[i] >= Gmax1:
                        Gmax1 = -self.G[i]
                if not self.is_lower_bound(i):
                    if self.G[i] >= Gmax2:
                        Gmax2 = self.G[i]
            else:
                if not self.is_upper_bound(i):
                    if -self.G[i] >= Gmax2:
                        Gmax2 = -self.G[i]
                if not self.is_lower_bound(i):
                    if self.G[i] >= Gmax1:
                        Gmax1 = self.G[i]

        if self.unshrink == False and Gmax1 + Gmax2 <= self.eps*10:
            self.unshrink = True
            self.reconstruct_gradient()
            self.active_size = self.l

        for i in range(self.active_size):
            if self.be_shrunk(i, Gmax1, Gmax2):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2):
                        self.swap_index(i,self.active_size)
                        break
                    self.active_size -= 1

    fn calculate_rho(self) -> Float64:
        var r: Float64
        var nr_free = 0
        var ub = math.inf[DType.float64]()
        var lb = -math.inf[DType.float64]()
        var sum_free = 0.0
        for i in range(self.active_size):
            var yG = self.y[i]*self.G[i]

            if self.is_upper_bound(i):
                if self.y[i]==-1:
                    ub = min(ub,yG)
                else:
                    lb = max(lb,yG)
            elif self.is_lower_bound(i):
                if self.y[i]==1:
                    ub = min(ub,yG)
                else:
                    lb = max(lb,yG)
            else:
                nr_free += 1
                sum_free += yG

        if nr_free>0:
            r = sum_free/nr_free
        else:
            r = (ub+lb)/2

        return r

#
# Solver for nu-svm classification and regression
#
# additional constraint: e^T \alpha = constant
#
struct Solver_NU:
