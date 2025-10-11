from memory import memcpy, memset_zero
from .svm_node import svm_node
from .svm_parameter import svm_parameter
from .svm_problem import svm_problem
from .svm_model import svm_model
from sys import size_of
import math
from algorithm import parallelize
from mojmelo.utils.utils import fill_indices
import random

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

@fieldwise_init
struct kernel_params:
    var x: UnsafePointer[UnsafePointer[svm_node]]
    var x_square: UnsafePointer[Float64]
    # svm_parameter
    var kernel_type: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64

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

@always_inline
fn kernel_linear(k: kernel_params, i: Int, j: Int) -> Float64:
    return dot(k.x[i],k.x[j])
@always_inline
fn kernel_poly(k: kernel_params, i: Int, j: Int) -> Float64:
    return powi(k.gamma*dot(k.x[i],k.x[j])+k.coef0,k.degree)
@always_inline
fn kernel_rbf(k: kernel_params, i: Int, j: Int) -> Float64:
    return math.exp(-k.gamma*(k.x_square[i]+k.x_square[j]-2*dot(k.x[i],k.x[j])))
@always_inline
fn kernel_sigmoid(k: kernel_params, i: Int, j: Int) -> Float64:
    return math.tanh(k.gamma*dot(k.x[i],k.x[j])+k.coef0)
@always_inline
fn kernel_precomputed(k: kernel_params, i: Int, j: Int) -> Float64:
    return k.x[i][Int(k.x[j][0].value)].value

struct head_t(Copyable, Movable):
    var prev: UnsafePointer[head_t]
    var next: UnsafePointer[head_t]	# a cicular list
    var data: UnsafePointer[Float32]
    var _len: Int		# data[0,len) is cached in this entry
    
    @always_inline
    fn __init__(out self):
        self.prev = UnsafePointer[head_t]()
        self.next = UnsafePointer[head_t]()
        self.data = UnsafePointer[Float32]()
        self._len = 0

# Kernel Cache
#
# l is the number of total data items
# size is the cache size limit in bytes
struct Cache(Copyable, Movable):
    var l: Int
    var size: UInt
    var head: UnsafePointer[head_t]
    var lru_head: head_t

    @always_inline
    fn __init__(out self, l: Int, size: UInt):
        self.l = l
        self.size = (size - (self.l * size_of[head_t]())) // 4
        self.head = UnsafePointer[head_t].alloc(self.l)
        memset_zero(self.head, self.l) # initialized to 0
        self.size = max(self.size, UInt(2) * UInt(l))  # cache must be large enough for two columns
        self.lru_head = head_t()
        self.lru_head.next = self.lru_head.prev = UnsafePointer(to=self.lru_head)

    fn __del__(deinit self):
        var h = self.lru_head.next
        while h != UnsafePointer(to=self.lru_head):
            if h[].data:
                h[].data.free()
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
trait QMatrix:
    fn get_Q(mut self, column: Int, _len: Int) -> UnsafePointer[Float32]:
        ...
    fn get_QD(self) -> UnsafePointer[Float64]:
        ...
    fn swap_index(mut self, i: Int, j: Int):
        ...

struct Kernel:
    var _self: kernel_params

    var kernel_function: fn(kernel_params, Int, Int) -> Float64

    @always_inline
    fn __init__(out self, l: Int, x_: UnsafePointer[UnsafePointer[svm_node]], param: svm_parameter):
        var x = UnsafePointer[UnsafePointer[svm_node]].alloc(l)
        memcpy(dest=x, src=x_, count=l)

        var x_square: UnsafePointer[Float64]
        if param.kernel_type == svm_parameter.RBF:
            x_square = UnsafePointer[Float64].alloc(l)
            for i in range(l):
                x_square[i] = dot(x[i], x[i])
        else:
            x_square = UnsafePointer[Float64]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        if self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        if self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        if self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        if self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear

    fn swap_index(self, i: Int, j: Int):
        swap(self._self.x[i],self._self.x[j])
        if self._self.x_square:
            swap(self._self.x_square[i],self._self.x_square[j])

    fn __del__(deinit self):
        if self._self.x:
            self._self.x.free()
        if self._self.x_square:
            self._self.x_square.free()

struct SolutionInfo:
    var obj: Float64
    var rho: Float64
    var upper_bound_p: Float64
    var upper_bound_n: Float64
    var r: Float64	# for Solver_NU

    @always_inline
    fn __init__(out self):
        self.obj = 0.0
        self.rho = 0.0
        self.upper_bound_p = 0.0
        self.upper_bound_n = 0.0
        self.r = 0.0

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
struct Solver:
    var active_size: Int
    var y: UnsafePointer[Int8]
    var G: UnsafePointer[Float64]	# gradient of objective function
    alias LOWER_BOUND: Int8 = 0
    alias UPPER_BOUND: Int8 = 1
    alias FREE: Int8 = 2
    var alpha_status: UnsafePointer[Int8]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: UnsafePointer[Float64]
    var QD: UnsafePointer[Float64]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: UnsafePointer[Float64]
    var active_set: UnsafePointer[Int]
    var G_bar: UnsafePointer[Float64]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    @always_inline
    fn __init__(out self):
        self.active_size = 0
        self.y = UnsafePointer[Int8]()
        self.G = UnsafePointer[Float64]()
        self.alpha_status = UnsafePointer[Int8]()
        self.alpha = UnsafePointer[Float64]()
        self.QD = UnsafePointer[Float64]()
        self.eps = 0.0
        self.Cp = 0.0
        self.Cn = 0.0
        self.p = UnsafePointer[Float64]()
        self.active_set = UnsafePointer[Int]()
        self.G_bar = UnsafePointer[Float64]()
        self.l = 0
        self.unshrink = False

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

    fn swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int):
        Q.swap_index(i,j)
        swap(self.y[i], self.y[j])
        swap(self.G[i], self.G[j])
        swap(self.alpha_status[i], self.alpha_status[j])
        swap(self.alpha[i], self.alpha[j])
        swap(self.p[i], self.p[j])
        swap(self.active_set[i], self.active_set[j])
        swap(self.G_bar[i], self.G_bar[j])

    fn reconstruct_gradient[QM: QMatrix](self, mut Q: QM):
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
                var Q_i = Q.get_Q(i,self.active_size)
                for j in range(self.active_size):
                    if self.is_free(j):
                        self.G[i] += self.alpha[j] * Q_i[j].cast[DType.float64]()
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha[i]
                    for j in range(self.active_size, self.l):
                        self.G[j] += alpha_i * Q_i[j].cast[DType.float64]()

    fn Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: UnsafePointer[Float64], y_: UnsafePointer[Int8],
                alpha_: UnsafePointer[Float64], Cp: Float64, Cn: Float64, eps: Float64, si: UnsafePointer[SolutionInfo], shrinking: Int):
        self.l = l
        self.QD = UnsafePointer[Float64]()
        self.QD = Q.get_QD()
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
                var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha[i]
                for j in range(self.l):
                    self.G[j] += alpha_i*Q_i[j].cast[DType.float64]()
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        self.G_bar[j] += self.get_C(i) * Q_i[j].cast[DType.float64]()

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
                    self.do_shrinking(Q)

            var i = -1
            var j = -1
            if self.select_working_set(Q, i,j)!=0:
                # reconstruct the whole gradient
                self.reconstruct_gradient(Q)
                # reset active set size and check
                self.active_size = self.l
                if self.select_working_set(Q, i,j)!=0:
                    break
                else:
                    counter = 1	# do shrinking next iteration

            iter += 1

            # update alpha[i] and alpha[j], handle bounds carefully

            var Q_i = Q.get_Q(i,self.active_size)
            var Q_j = Q.get_Q(j,self.active_size)

            var C_i = self.get_C(i)
            var C_j = self.get_C(j)

            var old_alpha_i = self.alpha[i]
            var old_alpha_j = self.alpha[j]

            if self.y[i]!=self.y[j]:
                var quad_coef = self.QD[i]+self.QD[j]+2*Q_i[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-self.G[i]-self.G[j])/quad_coef
                var diff = self.alpha[i] - self.alpha[j]
                self.alpha[i] += delta
                self.alpha[j] += delta

                if(diff > 0):
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = diff
                else:
                    if self.alpha[i] < 0:
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
                var quad_coef = self.QD[i]+self.QD[j]-2*Q_i[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (self.G[i]-self.G[j])/quad_coef
                var sum = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta

                if sum > C_i:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = sum - C_i
                else:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = sum
                if sum > C_j:
                    if self.alpha[j] > C_j:
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
                self.G[k] += Q_i[k].cast[DType.float64]()*delta_alpha_i + Q_j[k].cast[DType.float64]()*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar[k] -= C_i * Q_i[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[k] += C_i * Q_i[k].cast[DType.float64]()

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar[k] -= C_j * Q_j[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[k] += C_j * Q_j[k].cast[DType.float64]()

        if iter >= max_iter:
            if(self.active_size < self.l):
                # reconstruct the whole gradient to calculate objective value
                self.reconstruct_gradient(Q)
                self.active_size = self.l
            print("\nWARNING: reaching max number of iterations\n")

        # calculate rho

        si[].rho = self.calculate_rho()

        # calculate objective value
        var v = 0.0
        for i in range(self.l):
            v += self.alpha[i] * (self.G[i] + self.p[i])

        si[].obj = v/2

        # put back the solution

        for i in range(self.l):
            alpha_[self.active_set[i]] = self.alpha[i]

        # juggle everything back

        #for i in range(self.l):
        #    while self.active_set[i] != i:
        #        self.swap_index(i,self.active_set[i])
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
    fn select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int:
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
            Q_i = Q.get_Q(i,self.active_size)

        for j in range(self.active_size):
            if(self.y[j]==+1):
                if not self.is_lower_bound(j):
                    var grad_diff=Gmax+self.G[j]
                    if self.G[j] >= Gmax2:
                        Gmax2 = self.G[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[i]+self.QD[j]-2.0*Int(self.y[i])*Q_i[j].cast[DType.float64]()
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
                        var quad_coef = self.QD[i]+self.QD[j]+2.0*Int(self.y[i])*Q_i[j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff

        if Gmax+Gmax2 < self.eps or Gmin_idx == -1:
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

    fn do_shrinking[QM: QMatrix](mut self, mut Q: QM):
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
            self.reconstruct_gradient(Q)
            self.active_size = self.l

        for i in range(self.active_size):
            if self.be_shrunk(i, Gmax1, Gmax2):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2):
                        self.swap_index(Q, i,self.active_size)
                        break
                    self.active_size -= 1

    fn calculate_rho(self) -> Float64:
        var r: Float64
        var nr_free = 0
        var ub = math.inf[DType.float64]()
        var lb = -math.inf[DType.float64]()
        var sum_free = 0.0
        for i in range(self.active_size):
            var yG = Int(self.y[i])*self.G[i]

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
    var si: UnsafePointer[SolutionInfo]

    var active_size: Int
    var y: UnsafePointer[Int8]
    var G: UnsafePointer[Float64]	# gradient of objective function
    alias LOWER_BOUND: Int8 = 0
    alias UPPER_BOUND: Int8 = 1
    alias FREE: Int8 = 2
    var alpha_status: UnsafePointer[Int8]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: UnsafePointer[Float64]
    var QD: UnsafePointer[Float64]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: UnsafePointer[Float64]
    var active_set: UnsafePointer[Int]
    var G_bar: UnsafePointer[Float64]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    @always_inline
    fn __init__(out self):
        self.si = UnsafePointer[SolutionInfo]()
        self.active_size = 0
        self.y = UnsafePointer[Int8]()
        self.G = UnsafePointer[Float64]()
        self.alpha_status = UnsafePointer[Int8]()
        self.alpha = UnsafePointer[Float64]()
        self.QD = UnsafePointer[Float64]()
        self.eps = 0.0
        self.Cp = 0.0
        self.Cn = 0.0
        self.p = UnsafePointer[Float64]()
        self.active_set = UnsafePointer[Int]()
        self.G_bar = UnsafePointer[Float64]()
        self.l = 0
        self.unshrink = False

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

    fn swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int):
        Q.swap_index(i,j)
        swap(self.y[i], self.y[j])
        swap(self.G[i], self.G[j])
        swap(self.alpha_status[i], self.alpha_status[j])
        swap(self.alpha[i], self.alpha[j])
        swap(self.p[i], self.p[j])
        swap(self.active_set[i], self.active_set[j])
        swap(self.G_bar[i], self.G_bar[j])

    fn reconstruct_gradient[QM: QMatrix](self, mut Q: QM):
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
                var Q_i = Q.get_Q(i,self.active_size)
                for j in range(self.active_size):
                    if self.is_free(j):
                        self.G[i] += self.alpha[j] * Q_i[j].cast[DType.float64]()
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha[i]
                    for j in range(self.active_size, self.l):
                        self.G[j] += alpha_i * Q_i[j].cast[DType.float64]()

    fn Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: UnsafePointer[Float64], y_: UnsafePointer[Int8],
                alpha_: UnsafePointer[Float64], Cp: Float64, Cn: Float64, eps: Float64, si: UnsafePointer[SolutionInfo], shrinking: Int):
        self.si = si
        # Solve
        self.l = l
        self.QD = UnsafePointer[Float64]()
        self.QD = Q.get_QD()
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
                var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha[i]
                for j in range(self.l):
                    self.G[j] += alpha_i*Q_i[j].cast[DType.float64]()
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        self.G_bar[j] += self.get_C(i) * Q_i[j].cast[DType.float64]()

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
                    self.do_shrinking(Q)

            var i = -1
            var j = -1
            if self.select_working_set(Q, i,j)!=0:
                # reconstruct the whole gradient
                self.reconstruct_gradient(Q)
                # reset active set size and check
                self.active_size = self.l
                if self.select_working_set(Q, i,j)!=0:
                    break
                else:
                    counter = 1	# do shrinking next iteration

            iter += 1

            # update alpha[i] and alpha[j], handle bounds carefully

            var Q_i = Q.get_Q(i,self.active_size)
            var Q_j = Q.get_Q(j,self.active_size)

            var C_i = self.get_C(i)
            var C_j = self.get_C(j)

            var old_alpha_i = self.alpha[i]
            var old_alpha_j = self.alpha[j]

            if self.y[i]!=self.y[j]:
                var quad_coef = self.QD[i]+self.QD[j]+2*Q_i[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-self.G[i]-self.G[j])/quad_coef
                var diff = self.alpha[i] - self.alpha[j]
                self.alpha[i] += delta
                self.alpha[j] += delta

                if(diff > 0):
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = diff
                else:
                    if self.alpha[i] < 0:
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
                var quad_coef = self.QD[i]+self.QD[j]-2*Q_i[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (self.G[i]-self.G[j])/quad_coef
                var sum = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta

                if sum > C_i:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = sum - C_i
                else:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = sum
                if sum > C_j:
                    if self.alpha[j] > C_j:
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
                self.G[k] += Q_i[k].cast[DType.float64]()*delta_alpha_i + Q_j[k].cast[DType.float64]()*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar[k] -= C_i * Q_i[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[k] += C_i * Q_i[k].cast[DType.float64]()

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar[k] -= C_j * Q_j[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[k] += C_j * Q_j[k].cast[DType.float64]()

        if iter >= max_iter:
            if(self.active_size < self.l):
                # reconstruct the whole gradient to calculate objective value
                self.reconstruct_gradient(Q)
                self.active_size = self.l
            print("\nWARNING: reaching max number of iterations\n")

        # calculate rho

        si[].rho = self.calculate_rho()

        # calculate objective value
        var v = 0.0
        for i in range(self.l):
            v += self.alpha[i] * (self.G[i] + self.p[i])

        si[].obj = v/2

        # put back the solution

        for i in range(self.l):
            alpha_[self.active_set[i]] = self.alpha[i]

        # juggle everything back

        #for i in range(self.l):
        #   while self.active_set[i] != i:
        #       self.swap_index(i,self.active_set[i])
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
    fn select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int:
        # return i,j such that
        # i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        # j: minimizes the decrease of obj value
        #    (if quadratic coefficeint <= 0, replace it with tau)
        #    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        var Gmaxp = -math.inf[DType.float64]()
        var Gmaxp2 = -math.inf[DType.float64]()
        var Gmaxp_idx = -1

        var Gmaxn = -math.inf[DType.float64]()
        var Gmaxn2 = -math.inf[DType.float64]()
        var Gmaxn_idx = -1

        var Gmin_idx = -1
        var obj_diff_min = math.inf[DType.float64]()

        for t in range(self.active_size):
            if self.y[t]== 1:
                if not self.is_upper_bound(t):
                    if -self.G[t] >= Gmaxp:
                        Gmaxp = -self.G[t]
                        Gmaxp_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G[t] >= Gmaxn:
                        Gmaxn = self.G[t]
                        Gmaxn_idx = t

        var i_p = Gmaxp_idx
        var i_n = Gmaxn_idx
        var Q_ip = UnsafePointer[Float32]()
        var Q_in = UnsafePointer[Float32]()
        if i_p != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_ip = Q.get_Q(i_p,self.active_size)
        if i_n != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_in = Q.get_Q(i_n,self.active_size)

        for j in range(self.active_size):
            if(self.y[j]==+1):
                if not self.is_lower_bound(j):
                    var grad_diff=Gmaxp+self.G[j]
                    if self.G[j] >= Gmaxp2:
                        Gmaxp2 = self.G[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[i_p]+self.QD[j]-2.0*Q_ip[j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    var grad_diff= Gmaxn-self.G[j]
                    if -self.G[j] >= Gmaxn2:
                        Gmaxn2 = -self.G[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[i_n]+self.QD[j]+2.0*Q_in[j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff

        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < self.eps or Gmin_idx == -1:
            return 1

        if self.y[Gmin_idx] == 1:
            out_i = Gmaxp_idx
        else:
            out_i = Gmaxn_idx
        out_j = Gmin_idx
        return 0

    fn be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64, Gmax3: Float64, Gmax4: Float64) -> Bool:
        if self.is_upper_bound(i):
            if self.y[i]==1:
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax4
        elif self.is_lower_bound(i):
            if self.y[i]==1:
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax3
        else:
            return False

    fn do_shrinking[QM: QMatrix](mut self, mut Q: QM):
        var Gmax1 = -math.inf[DType.float64]()		# max { -y_i * grad(f)_i | i in I_up(\alpha) }
        var Gmax2 = -math.inf[DType.float64]()		# max { y_i * grad(f)_i | i in I_low(\alpha) }
        var Gmax3 = -math.inf[DType.float64]()	    # max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
        var Gmax4 = -math.inf[DType.float64]()	    # max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if not self.is_upper_bound(i):
                if self.y[i]==1:
                    if -self.G[i] > Gmax1:
                        Gmax1 = -self.G[i]
                else:
                    if -self.G[i] > Gmax4:
                        Gmax4 = -self.G[i]
            if not self.is_lower_bound(i):
                if self.y[i]==1:
                    if self.G[i] > Gmax2:
                        Gmax2 = self.G[i]
                else:
                    if self.G[i] > Gmax3:
                        Gmax3 = self.G[i]

        if self.unshrink == False and max(Gmax1+Gmax2,Gmax3+Gmax4) <= self.eps*10:
            self.unshrink = True
            self.reconstruct_gradient(Q)
            self.active_size = self.l

        for i in range(self.active_size):
            if self.be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2, Gmax3, Gmax4):
                        self.swap_index(Q, i,self.active_size)
                        break
                    self.active_size -= 1

    fn calculate_rho(self) -> Float64:
        var nr_free1 = 0
        var nr_free2 = 0
        var ub1 = math.inf[DType.float64]()
        var ub2 = math.inf[DType.float64]()
        var lb1 = -math.inf[DType.float64]()
        var lb2 = -math.inf[DType.float64]()
        var sum_free1 = 0.0
        var sum_free2 = 0.0

        for i in range(self.active_size):
            if self.y[i]==1:
                if self.is_upper_bound(i):
                    lb1 = max(lb1,self.G[i])
                elif self.is_lower_bound(i):
                    ub1 = min(ub1,self.G[i])
                else:
                    nr_free1 += 1
                    sum_free1 += self.G[i]
            else:
                if self.is_upper_bound(i):
                    lb2 = max(lb2,self.G[i])
                elif self.is_lower_bound(i):
                    ub2 = min(ub2,self.G[i])
                else:
                    nr_free2 += 1
                    sum_free2 += self.G[i]

        var r1: Float64
        var r2: Float64
        if nr_free1 > 0:
            r1 = sum_free1/nr_free1
        else:
            r1 = (ub1+lb1)/2

        if nr_free2 > 0:
            r2 = sum_free2/nr_free2
        else:
            r2 = (ub2+lb2)/2

        self.si[].r = (r1+r2)/2
        return (r1-r2)/2

#
# Q matrices for various formulations
#
struct SVC_Q(QMatrix):
    var y: UnsafePointer[Int8]
    var cache: Cache
    var QD: UnsafePointer[Float64]

    var _self: kernel_params

    var kernel_function: fn(kernel_params, Int, Int) -> Float64

    @always_inline
    fn __init__(out self, prob: svm_problem, param: svm_parameter, y_: UnsafePointer[Int8]):
        # Kernel
        var x = UnsafePointer[UnsafePointer[svm_node]].alloc(prob.l)
        memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: UnsafePointer[Float64]
        if param.kernel_type == svm_parameter.RBF:
            x_square = UnsafePointer[Float64].alloc(prob.l)
            for i in range(prob.l):
                x_square[i] = dot(x[i], x[i])
        else:
            x_square = UnsafePointer[Float64]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        if self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        if self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        if self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        if self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear
        ##
        self.y = UnsafePointer[Int8].alloc(prob.l)
        memcpy(dest=self.y, src=y_, count=prob.l)

        self.cache = Cache(prob.l, UInt(param.cache_size*(1<<20)))

        self.QD = UnsafePointer[Float64].alloc(prob.l)
        for i in range(prob.l):
            self.QD[i] = self.kernel_function(self._self, i,i)

    fn get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32]:
        var data = UnsafePointer[Float32]()
        var start = self.cache.get_data(i,UnsafePointer(to=data),_len)
        if start < _len:
            @parameter
            fn p(j: Int):
                data[j+start] = (Int(self.y[i]*self.y[j+start])*self.kernel_function(self._self, i,j+start)).cast[DType.float32]()
            parallelize[p](_len - start)
        return data

    fn get_QD(self) -> UnsafePointer[Float64]:
        return self.QD

    fn swap_index(mut self, i: Int, j: Int):
        self.cache.swap_index(i,j)

        swap(self._self.x[i],self._self.x[j])
        if self._self.x_square:
            swap(self._self.x_square[i],self._self.x_square[j])
        
        swap(self.y[i],self.y[j])
        swap(self.QD[i],self.QD[j])

    fn __del__(deinit self):
        if self._self.x:
            self._self.x.free()
        if self._self.x_square:
            self._self.x_square.free()

        if self.y:
            self.y.free()
        if self.QD:
            self.QD.free()

struct ONE_CLASS_Q(QMatrix):
    var cache: Cache
    var QD: UnsafePointer[Float64]

    var _self: kernel_params

    var kernel_function: fn(kernel_params, Int, Int) -> Float64

    @always_inline
    fn __init__(out self, prob: svm_problem, param: svm_parameter):
        # Kernel
        var x = UnsafePointer[UnsafePointer[svm_node]].alloc(prob.l)
        memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: UnsafePointer[Float64]
        if param.kernel_type == svm_parameter.RBF:
            x_square = UnsafePointer[Float64].alloc(prob.l)
            for i in range(prob.l):
                x_square[i] = dot(x[i], x[i])
        else:
            x_square = UnsafePointer[Float64]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        if self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        if self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        if self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        if self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear
        ##
        self.cache = Cache(prob.l, UInt(param.cache_size*(1<<20)))

        self.QD = UnsafePointer[Float64].alloc(prob.l)
        for i in range(prob.l):
            self.QD[i] = self.kernel_function(self._self, i,i)

    fn get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32]:
        var data = UnsafePointer[Float32]()
        var start = self.cache.get_data(i,UnsafePointer(to=data),_len)
        if start < _len:
            for j in range(start, _len):
                data[j] = self.kernel_function(self._self, i,j).cast[DType.float32]()
        return data

    fn get_QD(self) -> UnsafePointer[Float64]:
        return self.QD

    fn swap_index(mut self, i: Int, j: Int):
        self.cache.swap_index(i,j)

        swap(self._self.x[i],self._self.x[j])
        if self._self.x_square:
            swap(self._self.x_square[i],self._self.x_square[j])
        
        swap(self.QD[i],self.QD[j])

    fn __del__(deinit self):
        if self._self.x:
            self._self.x.free()
        if self._self.x_square:
            self._self.x_square.free()

        if self.QD:
            self.QD.free()

struct SVR_Q(QMatrix):
    var l: Int
    var cache: Cache
    var sign: UnsafePointer[Int8]
    var index: UnsafePointer[Int]
    var next_buffer: Int
    var buffer: InlineArray[UnsafePointer[Float32], 2]
    var QD: UnsafePointer[Float64]

    var _self: kernel_params

    var kernel_function: fn(kernel_params, Int, Int) -> Float64

    @always_inline
    fn __init__(out self, prob: svm_problem, param: svm_parameter):
        # Kernel
        var x = UnsafePointer[UnsafePointer[svm_node]].alloc(prob.l)
        memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: UnsafePointer[Float64]
        if param.kernel_type == svm_parameter.RBF:
            x_square = UnsafePointer[Float64].alloc(prob.l)
            for i in range(prob.l):
                x_square[i] = dot(x[i], x[i])
        else:
            x_square = UnsafePointer[Float64]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        if self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        if self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        if self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        if self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear
        ##
        self.l = prob.l
        self.cache = Cache(self.l,UInt(param.cache_size*(1<<20)))
        self.QD = UnsafePointer[Float64].alloc(2*self.l)
        self.sign = UnsafePointer[Int8].alloc(2*self.l)
        self.index = UnsafePointer[Int].alloc(2*self.l)
        for k in range(self.l):
            self.sign[k] = 1
            self.sign[k+self.l] = -1
            self.index[k] = k
            self.index[k+self.l] = k
            self.QD[k] = self.kernel_function(self._self, k,k)
            self.QD[k+self.l] = self.QD[k]
        self.buffer = InlineArray[UnsafePointer[Float32], 2](UnsafePointer[Float32].alloc(2*self.l), UnsafePointer[Float32].alloc(2*self.l))
        self.next_buffer = 0

    fn swap_index(self, i: Int, j: Int):
        swap(self.sign[i],self.sign[j])
        swap(self.index[i],self.index[j])
        swap(self.QD[i],self.QD[j])

    fn get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32]:
        var data = UnsafePointer[Float32]()
        var real_i = self.index[i]
        if self.cache.get_data(real_i,UnsafePointer(to=data),self.l) < self.l:
            @parameter
            fn p(j: Int):
                data[j] = self.kernel_function(self._self, real_i,j).cast[DType.float32]()
            parallelize[p](self.l)
        # reorder and copy
        var buf = self.buffer[self.next_buffer]
        var next_buffer = 1 - self.next_buffer
        var si = self.sign[i]
        for j in range(_len):
            buf[j] = si.cast[DType.float32]() * self.sign[j].cast[DType.float32]() * data[self.index[j]]
        return buf

    fn get_QD(self) -> UnsafePointer[Float64]:
        return self.QD

    fn __del__(deinit self):
        if self._self.x:
            self._self.x.free()
        if self._self.x_square:
            self._self.x_square.free()

        if self.QD:
            self.QD.free()
        if self.sign:
            self.sign.free()
        if self.index:
            self.index.free()
        if self.buffer[0]:
            self.buffer[0].free()
        if self.buffer[1]:
            self.buffer[1].free()

#
# construct and solve various formulations
#
fn solve_c_svc(
    prob: svm_problem, param: svm_parameter,
    alpha: UnsafePointer[Float64], si: SolutionInfo, Cp: Float64, Cn: Float64):
    var l = prob.l
    var minus_ones = UnsafePointer[Float64].alloc(l)
    var y = UnsafePointer[Int8].alloc(l)

    for i in range(l):
        alpha[i] = 0
        minus_ones[i] = -1
        if prob.y[i] > 0:
            y[i] = 1
        else:
            y[i] = -1

    var s = Solver()
    var q = SVC_Q(prob,param,y)
    s.Solve(l, q, minus_ones, y,
        alpha, Cp, Cn, param.eps, UnsafePointer(to=si), param.shrinking)

    var sum_alpha=0.0
    for i in range(l):
        sum_alpha += alpha[i]

    for i in range(l):
        alpha[i] *= Int(y[i])

    minus_ones.free()
    y.free()

fn solve_nu_svc(
    prob: svm_problem, param: svm_parameter,
    alpha: UnsafePointer[Float64], mut si: SolutionInfo):
    var l = prob.l
    var nu = param.nu

    var y = UnsafePointer[Int8].alloc(l)

    for i in range(l):
        if prob.y[i]>0:
            y[i] = 1
        else:
            y[i] = -1

    var sum_pos = nu*l/2
    var sum_neg = nu*l/2

    for i in range(l):
        if y[i] == 1:
            alpha[i] = min(1.0,sum_pos)
            sum_pos -= alpha[i]
        else:
            alpha[i] = min(1.0,sum_neg)
            sum_neg -= alpha[i]

    var zeros = UnsafePointer[Float64].alloc(l)

    for i in range(l):
        zeros[i] = 0

    var s = Solver_NU()
    var q = SVC_Q(prob,param,y)
    s.Solve(l, q, zeros, y,
        alpha, 1.0, 1.0, param.eps, UnsafePointer(to=si), param.shrinking)
    var r = si.r

    for i in range(l):
        alpha[i] *= Int(y[i])/r

    si.rho /= r
    si.obj /= (r*r)
    si.upper_bound_p = 1/r
    si.upper_bound_n = 1/r 

    y.free()
    zeros.free()

fn solve_one_class(
    prob: svm_problem, param: svm_parameter,
    alpha: UnsafePointer[Float64], mut si: SolutionInfo):
    var l = prob.l
    var zeros = UnsafePointer[Float64].alloc(l)
    var ones = UnsafePointer[Int8].alloc(l)

    var n = Int(param.nu*prob.l)	# # of alpha's at upper bound

    for i in range(n):
        alpha[i] = 1
    if n<prob.l:
        alpha[n] = param.nu * prob.l - n
    for i in range(n+1, l):
        alpha[i] = 0

    for i in range(l):
        zeros[i] = 0
        ones[i] = 1

    var s = Solver()
    var q = ONE_CLASS_Q(prob,param)
    s.Solve(l, q, zeros, ones,
        alpha, 1.0, 1.0, param.eps, UnsafePointer(to=si), param.shrinking)

    zeros.free()
    ones.free()

fn solve_epsilon_svr(
    prob: svm_problem, param: svm_parameter,
    alpha: UnsafePointer[Float64], mut si: SolutionInfo):
    var l = prob.l
    var alpha2 = UnsafePointer[Float64].alloc(2*l)
    var linear_term = UnsafePointer[Float64].alloc(2*l)
    var y = UnsafePointer[Int8].alloc(2*l)

    for i in range(l):
        alpha2[i] = 0
        linear_term[i] = param.p - prob.y[i]
        y[i] = 1

        alpha2[i+l] = 0
        linear_term[i+l] = param.p + prob.y[i]
        y[i+l] = -1

    var s = Solver()
    var q = SVR_Q(prob,param)
    s.Solve(2*l, q, linear_term, y,
        alpha2, param.C, param.C, param.eps, UnsafePointer(to=si), param.shrinking)

    var sum_alpha = 0.0
    for i in range(l):
        alpha[i] = alpha2[i] - alpha2[i+l]
        sum_alpha += abs(alpha[i])

    alpha2.free()
    linear_term.free()
    y.free()

fn solve_nu_svr(
    prob: svm_problem, param: svm_parameter,
    alpha: UnsafePointer[Float64], mut si: SolutionInfo):
    var l = prob.l
    var C = param.C
    var alpha2 = UnsafePointer[Float64].alloc(2*l)
    var linear_term = UnsafePointer[Float64].alloc(2*l)
    var y = UnsafePointer[Int8].alloc(2*l)

    var sum = C * param.nu * l / 2
    for i in range(l):
        alpha2[i] = alpha2[i+l] = min(sum,C)
        sum -= alpha2[i]

        linear_term[i] = - prob.y[i]
        y[i] = 1

        linear_term[i+l] = prob.y[i]
        y[i+l] = -1

    var s = Solver_NU()
    var q = SVR_Q(prob,param)
    s.Solve(2*l, q, linear_term, y,
        alpha2, C, C, param.eps, UnsafePointer(to=si), param.shrinking)

    for i in range(l):
        alpha[i] = alpha2[i] - alpha2[i+l]

    alpha2.free()
    linear_term.free()
    y.free()

#
# decision_function
#
@fieldwise_init
struct decision_function(Copyable):
    var alpha: UnsafePointer[Float64]
    var rho: Float64

fn svm_train_one(
	prob: svm_problem, param: svm_parameter,
	Cp: Float64, Cn: Float64) -> decision_function:
	var alpha = UnsafePointer[Float64].alloc(prob.l)
	var si = SolutionInfo()
	if param.svm_type == svm_parameter.C_SVC:
		solve_c_svc(prob,param,alpha,si,Cp,Cn)
	elif param.svm_type == svm_parameter.NU_SVC:
		solve_nu_svc(prob,param,alpha,si)
	elif param.svm_type == svm_parameter.ONE_CLASS:
		solve_one_class(prob,param,alpha,si)
	elif param.svm_type == svm_parameter.EPSILON_SVR:
		solve_epsilon_svr(prob,param,alpha,si)
	elif param.svm_type == svm_parameter.NU_SVR:
		solve_nu_svr(prob,param,alpha,si)

	# output SVs

	var nSV = 0
	var nBSV = 0
	for i in range(prob.l):
		if abs(alpha[i]) > 0:
			nSV += 1
			if prob.y[i] > 0:
				if abs(alpha[i]) >= si.upper_bound_p:
					nBSV += 1
			else:
				if abs(alpha[i]) >= si.upper_bound_n:
					nBSV += 1

	return decision_function(alpha=alpha, rho=si.rho)

# Platt's binary SVM Probablistic Output: an improvement from Lin et al.
fn sigmoid_train(
    l: Int, dec_values: UnsafePointer[Float64], labels: UnsafePointer[Float64],
    mut A: Float64, mut B: Float64):
    var prior1 = 0.0
    var prior0 = 0.0

    for i in range(l):
        if labels[i] > 0:
            prior1 += 1
        else:
            prior0 += 1

    var max_iter=100	# Maximal number of iterations
    var min_step=1e-10	# Minimal step taken in line search
    var sigma=1e-12	# For numerically strict PD of Hessian
    var eps=1e-5
    var hiTarget=(prior1+1.0)/(prior1+2.0)
    var loTarget=1/(prior0+2.0)
    var t=UnsafePointer[Float64].alloc(l)
    var fApB: Float64; p: Float64; q: Float64; h11: Float64; h22: Float64; h21: Float64; g1: Float64; g2: Float64; det: Float64; dA: Float64; dB: Float64; gd: Float64; stepsize: Float64
    var newA: Float64; newB: Float64; newf: Float64; d1: Float64; d2: Float64
    var iter: Int

    # Initial Point and Initial Fun Value
    A=0.0
    B=math.log((prior0+1.0)/(prior1+1.0))
    var fval = 0.0

    for i in range(l):
        if (labels[i]>0):
            t[i]=hiTarget
        else:
            t[i]=loTarget
        fApB = dec_values[i]*A+B
        if fApB>=0:
        	fval += t[i]*fApB + math.log(1+math.exp(-fApB))
        else:
        	fval += (t[i] - 1)*fApB +math.log(1+math.exp(fApB))
    
    iter = 0
    while iter<max_iter:
        # Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma # numerically ensures strict PD
        h22=sigma
        h21=0.0; g1=0.0; g2=0.0
        for i in range(l):
            fApB = dec_values[i]*A+B
            if (fApB >= 0):
                p=math.exp(-fApB)/(1.0+math.exp(-fApB))
                q=1.0/(1.0+math.exp(-fApB))
            else:
                p=1.0/(1.0+math.exp(fApB))
                q=math.exp(fApB)/(1.0+math.exp(fApB))

            d2=p*q
            h11+=dec_values[i]*dec_values[i]*d2
            h22+=d2
            h21+=dec_values[i]*d2
            d1=t[i]-p
            g1+=dec_values[i]*d1
            g2+=d1

            iter += 1

        # Stopping Criteria
        if abs(g1)<eps and abs(g2)<eps:
            break

        # Finding Newton direction: -inv(H') * g
        det=h11*h22-h21*h21
        dA=-(h22*g1 - h21 * g2) / det
        dB=-(-h21*g1+ h11 * g2) / det
        gd=g1*dA+g2*dB


        stepsize = 1		# Line Search
        while stepsize >= min_step:
            newA = A + stepsize * dA
            newB = B + stepsize * dB

			# New function value
            newf = 0.0
            for i in range(l):
                fApB = dec_values[i]*newA+newB
                if fApB >= 0:
                    newf += t[i]*fApB + math.log(1+math.exp(-fApB))
                else:
                    newf += (t[i] - 1)*fApB +math.log(1+math.exp(fApB))

			# Check sufficient decrease
            if newf<fval+0.0001*stepsize*gd:
                A=newA;B=newB;fval=newf
                break
            else:
                stepsize = stepsize / 2.0

        if stepsize < min_step:
            print("Line search fails in two-class probability estimates\n")
            break

    if iter>=max_iter:
        print("Reaching maximal iterations in two-class probability estimates\n")
    t.free()

fn sigmoid_predict(decision_value: Float64, A: Float64, B: Float64) -> Float64:
    var fApB = decision_value*A+B
    # 1-p used later; avoid catastrophic cancellation
    if fApB >= 0:
        return math.exp(-fApB)/(1.0+math.exp(-fApB))
    else:
        return 1.0/(1+math.exp(fApB))

# Method 2 from the multiclass_prob paper by Wu, Lin, and Weng to predict probabilities
fn multiclass_probability(k: Int, r: UnsafePointer[UnsafePointer[Float64]], p: UnsafePointer[Float64]):
    var iter = 0
    var max_iter=max(100,k)
    var Q=UnsafePointer[UnsafePointer[Float64]].alloc(k)
    var Qp=UnsafePointer[Float64].alloc(k)
    var pQp: Float64
    var eps=0.005/k

    for t in range(k):
        p[t]=1.0/k  # Valid if k = 1
        Q[t]=UnsafePointer[Float64].alloc(k)
        Q[t][t]=0
        for j in range(t):
            Q[t][t]+=r[j][t]*r[j][t]
            Q[t][j]=Q[j][t]
        for j in range(t+1,k):
            Q[t][t]+=r[j][t]*r[j][t]
            Q[t][j]=-r[j][t]*r[t][j]
    iter = 0
    while iter<max_iter:
		# stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0.0
        for t in range(k):
            Qp[t]=0
            for j in range(k):
                Qp[t]+=Q[t][j]*p[j]
            pQp+=p[t]*Qp[t]

        var max_error=0.0
        for t in range(k):
            var error=abs(Qp[t]-pQp)
            if error>max_error:
                max_error=error

        if max_error<eps:
            break

        for t in range(k):
            var diff=(-Qp[t]+pQp)/Q[t][t]
            p[t]+=diff
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff)
            for j in range(k):
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff)
                p[j]/=(1+diff)
        
        iter += 1

    if iter>=max_iter:
        print("Exceeds max_iter in multiclass_prob\n")
    for t in range(k):
        Q[t].free()
    Q.free()
    Qp.free()

# Using cross-validation decision values to get parameters for SVC probability estimates
fn svm_binary_svc_probability(
    prob: svm_problem, param: svm_parameter,
    Cp: Float64, Cn: Float64, mut probA: Float64, mut probB: Float64):
    var nr_fold = 5
    var perm: UnsafePointer[Scalar[DType.int]]
    var dec_values = UnsafePointer[Float64].alloc(prob.l)

	# random shuffle
    try:
        perm = fill_indices(prob.l)
    except:
        perm = UnsafePointer[Scalar[DType.int]].alloc(prob.l)
        for i in range(prob.l):
            perm[i]=i
    #random.seed()
    for i in range(prob.l - 1, 0, -1):
        var j = Int(random.random_ui64(0, i))
        swap(perm[i],perm[j])

    for i in range(nr_fold):
        var begin = i*prob.l//nr_fold
        var end = (i+1)*prob.l//nr_fold
        var k = 0
        var subprob: svm_problem

        subprob.l = prob.l-(end-begin)
        subprob.x = UnsafePointer[UnsafePointer[svm_node]].alloc(subprob.l)
        subprob.y = UnsafePointer[Float64].alloc(subprob.l)

        for j in range(begin):
            subprob.x[k] = prob.x[perm[j]]
            subprob.y[k] = prob.y[perm[j]]
            k += 1

        for j in range(end, prob.l):
            subprob.x[k] = prob.x[perm[j]]
            subprob.y[k] = prob.y[perm[j]]
            k += 1

        var p_count, n_count = 0, 0
        for j in range(k):
        	if subprob.y[j]>0:
        		p_count += 1
        	else:
        		n_count += 1

        if p_count==0 and n_count==0:
        	for j in range(begin, end):
        		dec_values[perm[j]] = 0
        elif p_count > 0 and n_count == 0:
        	for j in range(begin, end):
        		dec_values[perm[j]] = 1
        elif p_count == 0 and n_count > 0:
        	for j in range(begin, end):
        		dec_values[perm[j]] = -1
        else:
            var subparam = param.copy()
            subparam.probability=0
            subparam.C=1.0
            subparam.nr_weight=2
            subparam.weight_label = UnsafePointer[Int].alloc(2)
            subparam.weight = UnsafePointer[Float64].alloc(2)
            subparam.weight_label[0]=+1
            subparam.weight_label[1]=-1
            subparam.weight[0]=Cp
            subparam.weight[1]=Cn
            #var submodel = svm_train(subprob,subparam)
            for j in range(begin, end):
                pass
                #svm_predict_values(submodel,prob.x[perm[j]],&(dec_values[perm[j]]))
                # ensure +1 -1 order; reason not using CV subroutine
                #dec_values[perm[j]] *= submodel->label[0]

        	#svm_free_and_destroy_model(&submodel)
        	#svm_destroy_param(&subparam)

        subprob.x.free()
        subprob.y.free()

    sigmoid_train(prob.l,dec_values,prob.y,probA,probB)
    dec_values.free()
    perm.free()

# Binning method from the oneclass_prob paper by Que and Lin to predict the probability as a normal instance (i.e., not an outlier)
fn predict_one_class_probability(model: svm_model, dec_value: Float64) -> Float64:
    var prob_estimate = 0.0
    var nr_marks = 10

    if dec_value < model.prob_density_marks[0]:
    	prob_estimate = 0.001
    elif dec_value > model.prob_density_marks[nr_marks-1]:
    	prob_estimate = 0.999
    else:
    	for i in range(1,nr_marks):
    		if dec_value < model.prob_density_marks[i]:
    			prob_estimate = i/nr_marks
    			break

    return prob_estimate

# Get parameters for one-class SVM probability estimates
