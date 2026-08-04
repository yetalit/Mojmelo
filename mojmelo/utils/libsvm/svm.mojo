# Re-implementation of libsvm, a library for support vector machines by Chih-Chung Chang and Chih-Jen Lin (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) with some modifications.

from std.memory import unsafe_memcpy, unsafe_memset_zero, unsafe_memset
from .svm_node import svm_node
from .svm_parameter import svm_parameter
from .svm_problem import svm_problem
from .svm_model import svm_model
from std.sys import size_of
import std.math as math
from mojmelo.utils.algorithm import parallelize, reduction
from mojmelo.utils.utils import fill_indices
import std.random as random

comptime TAU = 1e-12

@always_inline
def powi(base: Float64, times: Int) -> Float64:
    var tmp = base
    var ret = 1.0

    var t = times
    while t>0:
        if t%2==1:
            ret *= tmp
        tmp = tmp * tmp
        t//=2
    return ret

@always_inline
def dot(var px: Pointer[svm_node, MutUntrackedOrigin], var py: Pointer[svm_node, MutUntrackedOrigin]) -> Float64:
    var sum = 0.0
    while px[].index != -1 and py[].index != -1:
        if px[].index == py[].index:
            sum += px[].value * py[].value
            px = px.unsafe_offset(1)
            py = py.unsafe_offset(1)
        else:
            if px[].index > py[].index:
                py = py.unsafe_offset(1)
            else:
                px = px.unsafe_offset(1)

    return sum

@fieldwise_init
struct kernel_params(RegisterPassable):
    var x: Pointer[Pointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]
    var x_square: Pointer[Float64, MutUntrackedOrigin]
    # svm_parameter
    var kernel_type: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64

def k_function(var x: Pointer[svm_node, MutUntrackedOrigin], var y: Pointer[svm_node, MutUntrackedOrigin], param: svm_parameter) -> Float64:
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
                x = x.unsafe_offset(1)
                y = y.unsafe_offset(1)
            else:
                if x[].index > y[].index:
                    sum += y[].value * y[].value
                    y = y.unsafe_offset(1)
                else:
                    sum += x[].value * x[].value
                    x = x.unsafe_offset(1)

        while x[].index != -1:
            sum += x[].value * x[].value
            x = x.unsafe_offset(1)

        while y[].index != -1:
            sum += y[].value * y[].value
            y = y.unsafe_offset(1)

        return math.exp(-param.gamma*sum)
    if param.kernel_type == svm_parameter.SIGMOID:
        return math.tanh(param.gamma*dot(x,y)+param.coef0)
    if param.kernel_type == svm_parameter.PRECOMPUTED:  # x: test (validation), y: SV
        return x[unsafe_offset=Int(y[].value)].value
    else:
        return 0  # Unreachable

@always_inline
def kernel_linear(k: kernel_params, i: Int, j: Int) -> Float64:
    return dot(k.x[unsafe_offset=i],k.x[unsafe_offset=j])
@always_inline
def kernel_poly(k: kernel_params, i: Int, j: Int) -> Float64:
    return powi(k.gamma*dot(k.x[unsafe_offset=i],k.x[unsafe_offset=j])+k.coef0,k.degree)
@always_inline
def kernel_rbf(k: kernel_params, i: Int, j: Int) -> Float64:
    return math.exp(-k.gamma*(k.x_square[unsafe_offset=i]+k.x_square[unsafe_offset=j]-2*dot(k.x[unsafe_offset=i],k.x[unsafe_offset=j])))
@always_inline
def kernel_sigmoid(k: kernel_params, i: Int, j: Int) -> Float64:
    return math.tanh(k.gamma*dot(k.x[unsafe_offset=i],k.x[unsafe_offset=j])+k.coef0)
@always_inline
def kernel_precomputed(k: kernel_params, i: Int, j: Int) -> Float64:
    return k.x[unsafe_offset=i][unsafe_offset=Int(k.x[unsafe_offset=j][unsafe_offset=0].value)].value

struct head_t(RegisterPassable):
    var prev: OptionalPointer[head_t, MutUntrackedOrigin]
    var next: OptionalPointer[head_t, MutUntrackedOrigin]	# a cicular list
    var data: OptionalPointer[Float32, MutUntrackedOrigin]
    var _len: Int		# data[0,len) is cached in this entry

    @always_inline
    def __init__(out self):
        self.prev = None
        self.next = None
        self.data = None
        self._len = 0

# Kernel Cache
#
# l is the number of total data items
# size is the cache size limit in bytes
struct Cache:
    var l: Int
    var size: UInt
    var head: OptionalPointer[head_t, MutUntrackedOrigin]
    var lru_head: head_t

    @always_inline
    def __init__(out self, l_: Int, size_: UInt):
        self.l = l_
        self.size = (size_ - UInt(self.l * size_of[head_t]())) // 4
        self.head = alloc[head_t](self.l)
        unsafe_memset_zero(self.head.value(), self.l) # initialized to 0
        self.size = max(self.size, UInt(2) * UInt(self.l))  # cache must be large enough for two columns
        self.lru_head = head_t()
        self.lru_head.next = self.lru_head.prev = Pointer[head_t, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.lru_head)))

    def __deinit__(deinit self):
        var h = self.lru_head.next
        while h != Pointer[head_t, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.lru_head))):
            var _h = h.value()
            if _h[].data:
                _h[].data.value().unsafe_free()
            h = _h[].next
        if self.head:
            self.head.value().unsafe_free()

    def lru_delete(self, h: Pointer[head_t, MutUntrackedOrigin]):
        # delete from current location
        h[].prev.value()[].next = h[].next
        h[].next.value()[].prev = h[].prev

    def lru_insert(mut self, h: Pointer[head_t, MutUntrackedOrigin]):
        # insert to last position
        h[].next = Pointer[head_t, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.lru_head)))
        h[].prev = self.lru_head.prev
        h[].prev.value()[].next = h
        h[].next.value()[].prev = h

    @always_inline
    def get_data(mut self, index: Int, data: Pointer[OptionalPointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin], var _len: Int) -> Int:
        var h = self.head.value().unsafe_offset(index)
        if h[]._len:
            self.lru_delete(h)
        var more = _len - h[]._len

        if more > 0:
            # free old space
            while self.size < UInt(more):
                var old = self.lru_head.next.value()
                self.lru_delete(old)
                old[].data.value().unsafe_free()
                self.size += UInt(old[]._len)
                old[].data = OptionalPointer[Float32, MutUntrackedOrigin]()
                old[]._len = 0

            # allocate new space
            var new = alloc[Float32](_len)
            if h[].data:
                unsafe_memcpy(dest=new, src=h[].data.value(), count=h[]._len)
                h[].data.value().unsafe_free()
            h[].data = new
            self.size -= UInt(more)  # previous while loop guarantees size >= more and subtraction of size_t variable will not underflow
            swap(h[]._len, _len)

        self.lru_insert(h)
        data[] = h[].data
        return _len

    @always_inline
    def swap_index(mut self, var i: Int, var j: Int):
        if i==j:
            return

        var head = self.head.value()
        if head[unsafe_offset=i]._len:
            self.lru_delete(head.unsafe_offset(i))
        if head[unsafe_offset=j]._len:
            self.lru_delete(head.unsafe_offset(j))
        swap(head[unsafe_offset=i].data,head[unsafe_offset=j].data)
        swap(head[unsafe_offset=i]._len,head[unsafe_offset=j]._len)
        if head[unsafe_offset=i]._len:
            self.lru_insert(head.unsafe_offset(i))
        if head[unsafe_offset=j]._len:
            self.lru_insert(head.unsafe_offset(j))

        if i>j:
            swap(i,j)

        var h = self.lru_head.next.value()
        while h != Pointer(to=self.lru_head):
            if h[]._len > i:
                if(h[]._len > j):
                    swap(h[].data.value()[unsafe_offset=i],h[].data.value()[unsafe_offset=j])
                else:
                    # give up
                    self.lru_delete(h)
                    h[].data.value().unsafe_free()
                    self.size += UInt(h[]._len)
                    h[].data = OptionalPointer[Float32, MutUntrackedOrigin]()
                    h[]._len = 0
            h=h[].next.value()

# Kernel evaluation
#
# the static method k_function is for doing single kernel evaluation
# the constructor of Kernel prepares to calculate the l*l kernel matrix
# the member function get_Q is for getting one column from the Q Matrix
#
trait QMatrix:
    def get_Q(mut self, column: Int, _len: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        ...
    def get_QD(self) -> Pointer[Float64, MutUntrackedOrigin]:
        ...
    def swap_index(mut self, i: Int, j: Int):
        ...

#struct Kernel:
#    var _self: kernel_params
#
#    var kernel_function: def(kernel_params, Int, Int) -> Float64
#
#    @always_inline
#    def __init__(out self, l: Int, x_: OptionalPointer[OptionalPointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin], param: svm_parameter):
#        var x = alloc[OptionalPointer[svm_node, MutUntrackedOrigin]](l)
#        unsafe_memcpy(dest=x, src=x_, count=l)
#
#        var x_square: OptionalPointer[Float64, MutUntrackedOrigin]
#        if param.kernel_type == svm_parameter.RBF:
#            x_square = alloc[Float64](l)
#            for i in range(l):
#                x_square[i] = dot(x[i], x[i])
#        else:
#            x_square = OptionalPointer[Float64, MutUntrackedOrigin]()
#
#        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)
#
#        if self._self.kernel_type == svm_parameter.LINEAR:
#            self.kernel_function = kernel_linear
#        elif self._self.kernel_type == svm_parameter.POLY:
#            self.kernel_function = kernel_poly
#        elif self._self.kernel_type == svm_parameter.RBF:
#            self.kernel_function = kernel_rbf
#        elif self._self.kernel_type == svm_parameter.SIGMOID:
#            self.kernel_function = kernel_sigmoid
#        elif self._self.kernel_type == svm_parameter.PRECOMPUTED:
#            self.kernel_function = kernel_precomputed
#        else:
#            self.kernel_function = kernel_linear
#
#    def swap_index(self, i: Int, j: Int):
#        swap(self._self.x[i],self._self.x[j])
#        if self._self.x_square:
#            swap(self._self.x_square[i],self._self.x_square[j])
#
#    def __del__(deinit self):
#        if self._self.x:
#            self._self.x.free()
#        if self._self.x_square:
#            self._self.x_square.free()

struct SolutionInfo(TrivialRegisterPassable):
    var obj: Float64
    var rho: Float64
    var upper_bound_p: Float64
    var upper_bound_n: Float64
    var r: Float64	# for Solver_NU

    @always_inline
    def __init__(out self):
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
    var y: Pointer[Int8, MutUntrackedOrigin]
    var G: Pointer[Float64, MutUntrackedOrigin]	# gradient of objective function
    comptime LOWER_BOUND: Int8 = 0
    comptime UPPER_BOUND: Int8 = 1
    comptime FREE: Int8 = 2
    var alpha_status: Pointer[Int8, MutUntrackedOrigin]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: Pointer[Float64, MutUntrackedOrigin]
    var QD: Pointer[Float64, MutUntrackedOrigin]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: Pointer[Float64, MutUntrackedOrigin]
    var active_set: Pointer[Int, MutUntrackedOrigin]
    var G_bar: Pointer[Float64, MutUntrackedOrigin]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    @always_inline
    def __init__(out self):
        self.active_size = 0
        self.y = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling()
        self.G = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.alpha_status = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling()
        self.alpha = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.QD = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.eps = 0.0
        self.Cp = 0.0
        self.Cn = 0.0
        self.p = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.active_set = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()
        self.G_bar = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.l = 0
        self.unshrink = False

    def get_C(self, i: Int) -> Float64:
        return self.Cp if self.y[unsafe_offset=i] > 0 else self.Cn

    def update_alpha_status(self, i: Int):
        if self.alpha[unsafe_offset=i] >= self.get_C(i):
            self.alpha_status[unsafe_offset=i] = self.UPPER_BOUND
        elif self.alpha[unsafe_offset=i] <= 0:
            self.alpha_status[unsafe_offset=i] = self.LOWER_BOUND
        else:
            self.alpha_status[unsafe_offset=i] = self.FREE

    def is_upper_bound(self, i: Int) -> Bool:
        return self.alpha_status[unsafe_offset=i] == self.UPPER_BOUND
    def is_lower_bound(self, i: Int) -> Bool:
        return self.alpha_status[unsafe_offset=i] == self.LOWER_BOUND
    def is_free(self, i: Int) -> Bool:
        return self.alpha_status[unsafe_offset=i] == self.FREE

    def swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int):
        Q.swap_index(i,j)
        swap(self.y[unsafe_offset=i], self.y[unsafe_offset=j])
        swap(self.G[unsafe_offset=i], self.G[unsafe_offset=j])
        swap(self.alpha_status[unsafe_offset=i], self.alpha_status[unsafe_offset=j])
        swap(self.alpha[unsafe_offset=i], self.alpha[unsafe_offset=j])
        swap(self.p[unsafe_offset=i], self.p[unsafe_offset=j])
        swap(self.active_set[unsafe_offset=i], self.active_set[unsafe_offset=j])
        swap(self.G_bar[unsafe_offset=i], self.G_bar[unsafe_offset=j])

    def reconstruct_gradient[QM: QMatrix](self, mut Q: QM):
        # reconstruct inactive elements of G from G_bar and free variables

        if self.active_size == self.l:
            return

        var nr_free = 0

        for j in range(self.active_size, self.l):
            self.G[unsafe_offset=j] = self.G_bar[unsafe_offset=j] + self.p[unsafe_offset=j]

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
                        self.G[unsafe_offset=i] += self.alpha[unsafe_offset=j] * Q_i[unsafe_offset=j].cast[DType.float64]()
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha[unsafe_offset=i]
                    for j in range(self.active_size, self.l):
                        self.G[unsafe_offset=j] += alpha_i * Q_i[unsafe_offset=j].cast[DType.float64]()

    def Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: OptionalPointer[Float64, MutUntrackedOrigin], y_: OptionalPointer[Int8, MutUntrackedOrigin],
                alpha_: Pointer[Float64, MutUntrackedOrigin], Cp: Float64, Cn: Float64, eps: Float64, mut si: SolutionInfo, shrinking: Int):
        self.l = l
        self.QD = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.QD = Q.get_QD()
        self.p = alloc[Float64](self.l)
        unsafe_memcpy(dest=self.p, src=p_.value(), count=self.l)
        self.y = alloc[Int8](self.l)
        unsafe_memcpy(dest=self.y, src=y_.value(), count=self.l)
        self.alpha = alloc[Float64](self.l)
        unsafe_memcpy(dest=self.alpha, src=alpha_, count=self.l)
        self.Cp = Cp
        self.Cn = Cn
        self.eps = eps
        self.unshrink = False

        # initialize alpha_status
        self.alpha_status = alloc[Int8](self.l)
        for i in range(self.l):
            if self.alpha[unsafe_offset=i] >= (self.Cp if self.y[unsafe_offset=i] > 0 else self.Cn):
                self.alpha_status[unsafe_offset=i] = self.UPPER_BOUND
            elif self.alpha[unsafe_offset=i] <= 0:
                self.alpha_status[unsafe_offset=i] = self.LOWER_BOUND
            else:
                self.alpha_status[unsafe_offset=i] = self.FREE

        # initialize active set (for shrinking)
        try:
            self.active_set = fill_indices(self.l)
        except:
            self.active_set = alloc[Int](self.l)
            for i in range(self.l):
                self.active_set[unsafe_offset=i] = i
        self.active_size = self.l

        # initialize gradient
        self.G = alloc[Float64](self.l)
        self.G_bar = alloc[Float64](self.l)
        unsafe_memcpy(dest=self.G, src=self.p, count=self.l)
        unsafe_memset_zero(self.G_bar, self.l)

        for i in range(self.l):
            if not self.is_lower_bound(i):
                var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha[unsafe_offset=i]
                for j in range(self.l):
                    self.G[unsafe_offset=j] += alpha_i*Q_i[unsafe_offset=j].cast[DType.float64]()
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        self.G_bar[unsafe_offset=j] += self.get_C(i) * Q_i[unsafe_offset=j].cast[DType.float64]()

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

            var old_alpha_i = self.alpha[unsafe_offset=i]
            var old_alpha_j = self.alpha[unsafe_offset=j]

            if self.y[unsafe_offset=i]!=self.y[unsafe_offset=j]:
                var quad_coef = self.QD[unsafe_offset=i]+self.QD[unsafe_offset=j]+2*Q_i[unsafe_offset=j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-self.G[unsafe_offset=i]-self.G[unsafe_offset=j])/quad_coef
                var diff = self.alpha[unsafe_offset=i] - self.alpha[unsafe_offset=j]
                self.alpha[unsafe_offset=i] += delta
                self.alpha[unsafe_offset=j] += delta

                if(diff > 0):
                    if self.alpha[unsafe_offset=j] < 0:
                        self.alpha[unsafe_offset=j] = 0
                        self.alpha[unsafe_offset=i] = diff
                else:
                    if self.alpha[unsafe_offset=i] < 0:
                        self.alpha[unsafe_offset=i] = 0
                        self.alpha[unsafe_offset=j] = -diff
                if diff > C_i - C_j:
                    if self.alpha[unsafe_offset=i] > C_i:
                        self.alpha[unsafe_offset=i] = C_i
                        self.alpha[unsafe_offset=j] = C_i - diff
                else:
                    if self.alpha[unsafe_offset=j] > C_j:
                        self.alpha[unsafe_offset=j] = C_j
                        self.alpha[unsafe_offset=i] = C_j + diff
            else:
                var quad_coef = self.QD[unsafe_offset=i]+self.QD[unsafe_offset=j]-2*Q_i[unsafe_offset=j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (self.G[unsafe_offset=i]-self.G[unsafe_offset=j])/quad_coef
                var sum = self.alpha[unsafe_offset=i] + self.alpha[unsafe_offset=j]
                self.alpha[unsafe_offset=i] -= delta
                self.alpha[unsafe_offset=j] += delta

                if sum > C_i:
                    if self.alpha[unsafe_offset=i] > C_i:
                        self.alpha[unsafe_offset=i] = C_i
                        self.alpha[unsafe_offset=j] = sum - C_i
                else:
                    if self.alpha[unsafe_offset=j] < 0:
                        self.alpha[unsafe_offset=j] = 0
                        self.alpha[unsafe_offset=i] = sum
                if sum > C_j:
                    if self.alpha[unsafe_offset=j] > C_j:
                        self.alpha[unsafe_offset=j] = C_j
                        self.alpha[unsafe_offset=i] = sum - C_j
                else:
                    if self.alpha[unsafe_offset=i] < 0:
                        self.alpha[unsafe_offset=i] = 0
                        self.alpha[unsafe_offset=j] = sum

            # update G

            var delta_alpha_i = self.alpha[unsafe_offset=i] - old_alpha_i
            var delta_alpha_j = self.alpha[unsafe_offset=j] - old_alpha_j

            for k in range(self.active_size):
                self.G[unsafe_offset=k] += Q_i[unsafe_offset=k].cast[DType.float64]()*delta_alpha_i + Q_j[unsafe_offset=k].cast[DType.float64]()*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] -= C_i * Q_i[unsafe_offset=k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] += C_i * Q_i[unsafe_offset=k].cast[DType.float64]()

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] -= C_j * Q_j[unsafe_offset=k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] += C_j * Q_j[unsafe_offset=k].cast[DType.float64]()

        if iter >= max_iter:
            if(self.active_size < self.l):
                # reconstruct the whole gradient to calculate objective value
                self.reconstruct_gradient(Q)
                self.active_size = self.l
            print("\nWARNING: reaching max number of iterations\n")

        # calculate rho

        si.rho = self.calculate_rho()

        # calculate objective value
        var v = 0.0
        for i in range(self.l):
            v += self.alpha[unsafe_offset=i] * (self.G[unsafe_offset=i] + self.p[unsafe_offset=i])

        si.obj = v/2

        # put back the solution

        for i in range(self.l):
            alpha_[unsafe_offset=self.active_set[unsafe_offset=i]] = self.alpha[unsafe_offset=i]

        # juggle everything back

        #for i in range(self.l):
        #    while self.active_set[i] != i:
        #        self.swap_index(i,self.active_set[i])
        #       # or Q.swap_index(i,self.active_set[i])


        si.upper_bound_p = Cp
        si.upper_bound_n = Cn

        self.p.unsafe_free()
        self.y.unsafe_free()
        self.alpha.unsafe_free()
        self.alpha_status.unsafe_free()
        self.active_set.unsafe_free()
        self.G.unsafe_free()
        self.G_bar.unsafe_free()

    # return 1 if already optimal, return 0 otherwise
    def select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int:
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
            if self.y[unsafe_offset=t]== 1:
                if not self.is_upper_bound(t):
                    if -self.G[unsafe_offset=t] >= Gmax:
                        Gmax = -self.G[unsafe_offset=t]
                        Gmax_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G[unsafe_offset=t] >= Gmax:
                        Gmax = self.G[unsafe_offset=t]
                        Gmax_idx = t

        var i = Gmax_idx
        var Q_i = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        if i != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_i = Q.get_Q(i,self.active_size)

        for j in range(self.active_size):
            if self.y[unsafe_offset=j]==1:
                if not self.is_lower_bound(j):
                    var grad_diff=Gmax+self.G[unsafe_offset=j]
                    if self.G[unsafe_offset=j] >= Gmax2:
                        Gmax2 = self.G[unsafe_offset=j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[unsafe_offset=i]+self.QD[unsafe_offset=j]-2.0*self.y[unsafe_offset=i].cast[DType.float64]()*Q_i[unsafe_offset=j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    var grad_diff= Gmax-self.G[unsafe_offset=j]
                    if -self.G[unsafe_offset=j] >= Gmax2:
                        Gmax2 = -self.G[unsafe_offset=j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[unsafe_offset=i]+self.QD[unsafe_offset=j]+2.0*self.y[unsafe_offset=i].cast[DType.float64]()*Q_i[unsafe_offset=j].cast[DType.float64]()
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

    def be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64) -> Bool:
        if self.is_upper_bound(i):
            if self.y[unsafe_offset=i]==1:
                return -self.G[unsafe_offset=i] > Gmax1
            else:
                return -self.G[unsafe_offset=i] > Gmax2
        elif self.is_lower_bound(i):
            if self.y[unsafe_offset=i]==1:
                return self.G[unsafe_offset=i] > Gmax2
            else:
                return self.G[unsafe_offset=i] > Gmax1
        else:
            return False

    def do_shrinking[QM: QMatrix](mut self, mut Q: QM):
        var Gmax1 = -math.inf[DType.float64]()		# max { -y_i * grad(f)_i | i in I_up(\alpha) }
        var Gmax2 = -math.inf[DType.float64]()		# max { y_i * grad(f)_i | i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if self.y[unsafe_offset=i]==1:
                if not self.is_upper_bound(i):
                    if -self.G[unsafe_offset=i] >= Gmax1:
                        Gmax1 = -self.G[unsafe_offset=i]
                if not self.is_lower_bound(i):
                    if self.G[unsafe_offset=i] >= Gmax2:
                        Gmax2 = self.G[unsafe_offset=i]
            else:
                if not self.is_upper_bound(i):
                    if -self.G[unsafe_offset=i] >= Gmax2:
                        Gmax2 = -self.G[unsafe_offset=i]
                if not self.is_lower_bound(i):
                    if self.G[unsafe_offset=i] >= Gmax1:
                        Gmax1 = self.G[unsafe_offset=i]

        if self.unshrink == False and Gmax1 + Gmax2 <= self.eps*10:
            self.unshrink = True
            self.reconstruct_gradient(Q)
            self.active_size = self.l

        var i = 0
        while i < self.active_size:
            if self.be_shrunk(i, Gmax1, Gmax2):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2):
                        self.swap_index(Q, i,self.active_size)
                        break
                    self.active_size -= 1
            i += 1

    def calculate_rho(self) -> Float64:
        var r: Float64
        var nr_free = 0
        var ub = math.inf[DType.float64]()
        var lb = -math.inf[DType.float64]()
        var sum_free = 0.0
        for i in range(self.active_size):
            var yG = self.y[unsafe_offset=i].cast[DType.float64]()*self.G[unsafe_offset=i]

            if self.is_upper_bound(i):
                if self.y[unsafe_offset=i]==-1:
                    ub = min(ub,yG)
                else:
                    lb = max(lb,yG)
            elif self.is_lower_bound(i):
                if self.y[unsafe_offset=i]==1:
                    ub = min(ub,yG)
                else:
                    lb = max(lb,yG)
            else:
                nr_free += 1
                sum_free += yG

        if nr_free>0:
            r = sum_free/Float64(nr_free)
        else:
            r = (ub+lb)/2

        return r

#
# Solver for nu-svm classification and regression
#
# additional constraint: e^T \alpha = constant
#
struct Solver_NU:
    var si: SolutionInfo

    var active_size: Int
    var y: Pointer[Int8, MutUntrackedOrigin]
    var G: Pointer[Float64, MutUntrackedOrigin]	# gradient of objective function
    comptime LOWER_BOUND: Int8 = 0
    comptime UPPER_BOUND: Int8 = 1
    comptime FREE: Int8 = 2
    var alpha_status: Pointer[Int8, MutUntrackedOrigin]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: Pointer[Float64, MutUntrackedOrigin]
    var QD: Pointer[Float64, MutUntrackedOrigin]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: Pointer[Float64, MutUntrackedOrigin]
    var active_set: Pointer[Int, MutUntrackedOrigin]
    var G_bar: Pointer[Float64, MutUntrackedOrigin]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    @always_inline
    def __init__(out self):
        self.si = SolutionInfo()
        self.active_size = 0
        self.y = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling()
        self.G = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.alpha_status = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling()
        self.alpha = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.QD = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.eps = 0.0
        self.Cp = 0.0
        self.Cn = 0.0
        self.p = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.active_set = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()
        self.G_bar = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.l = 0
        self.unshrink = False

    def get_C(self, i: Int) -> Float64:
        return self.Cp if self.y[unsafe_offset=i] > 0 else self.Cn

    def update_alpha_status(self, i: Int):
        if self.alpha[unsafe_offset=i] >= self.get_C(i):
            self.alpha_status[unsafe_offset=i] = self.UPPER_BOUND
        elif self.alpha[unsafe_offset=i] <= 0:
            self.alpha_status[unsafe_offset=i] = self.LOWER_BOUND
        else:
            self.alpha_status[unsafe_offset=i] = self.FREE

    def is_upper_bound(self, i: Int) -> Bool:
        return self.alpha_status[unsafe_offset=i] == self.UPPER_BOUND
    def is_lower_bound(self, i: Int) -> Bool:
        return self.alpha_status[unsafe_offset=i] == self.LOWER_BOUND
    def is_free(self, i: Int) -> Bool:
        return self.alpha_status[unsafe_offset=i] == self.FREE

    def swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int):
        Q.swap_index(i,j)
        swap(self.y[unsafe_offset=i], self.y[unsafe_offset=j])
        swap(self.G[unsafe_offset=i], self.G[unsafe_offset=j])
        swap(self.alpha_status[unsafe_offset=i], self.alpha_status[unsafe_offset=j])
        swap(self.alpha[unsafe_offset=i], self.alpha[unsafe_offset=j])
        swap(self.p[unsafe_offset=i], self.p[unsafe_offset=j])
        swap(self.active_set[unsafe_offset=i], self.active_set[unsafe_offset=j])
        swap(self.G_bar[unsafe_offset=i], self.G_bar[unsafe_offset=j])

    def reconstruct_gradient[QM: QMatrix](self, mut Q: QM):
        # reconstruct inactive elements of G from G_bar and free variables

        if self.active_size == self.l:
            return

        var nr_free = 0

        for j in range(self.active_size, self.l):
            self.G[unsafe_offset=j] = self.G_bar[unsafe_offset=j] + self.p[unsafe_offset=j]

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
                        self.G[unsafe_offset=i] += self.alpha[unsafe_offset=j] * Q_i[unsafe_offset=j].cast[DType.float64]()
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha[unsafe_offset=i]
                    for j in range(self.active_size, self.l):
                        self.G[unsafe_offset=j] += alpha_i * Q_i[unsafe_offset=j].cast[DType.float64]()

    def Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: OptionalPointer[Float64, MutUntrackedOrigin], y_: OptionalPointer[Int8, MutUntrackedOrigin],
                alpha_: Pointer[Float64, MutUntrackedOrigin], Cp: Float64, Cn: Float64, eps: Float64, si: SolutionInfo, shrinking: Int):
        self.si = si
        # Solve
        self.l = l
        self.QD = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()
        self.QD = Q.get_QD()
        self.p = alloc[Float64](self.l)
        unsafe_memcpy(dest=self.p, src=p_.value(), count=self.l)
        self.y = alloc[Int8](self.l)
        unsafe_memcpy(dest=self.y, src=y_.value(), count=self.l)
        self.alpha = alloc[Float64](self.l)
        unsafe_memcpy(dest=self.alpha, src=alpha_, count=self.l)
        self.Cp = Cp
        self.Cn = Cn
        self.eps = eps
        self.unshrink = False

        # initialize alpha_status
        self.alpha_status = alloc[Int8](self.l)
        for i in range(self.l):
            if self.alpha[unsafe_offset=i] >= (self.Cp if self.y[unsafe_offset=i] > 0 else self.Cn):
                self.alpha_status[unsafe_offset=i] = self.UPPER_BOUND
            elif self.alpha[unsafe_offset=i] <= 0:
                self.alpha_status[unsafe_offset=i] = self.LOWER_BOUND
            else:
                self.alpha_status[unsafe_offset=i] = self.FREE

        # initialize active set (for shrinking)
        try:
            self.active_set = fill_indices(self.l)
        except:
            self.active_set = alloc[Int](self.l)
            for i in range(self.l):
                self.active_set[unsafe_offset=i] = i
        self.active_size = self.l

        # initialize gradient
        self.G = alloc[Float64](self.l)
        self.G_bar = alloc[Float64](self.l)
        unsafe_memcpy(dest=self.G, src=self.p, count=self.l)
        unsafe_memset_zero(self.G_bar, self.l)

        for i in range(self.l):
            if not self.is_lower_bound(i):
                var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha[unsafe_offset=i]
                for j in range(self.l):
                    self.G[unsafe_offset=j] += alpha_i*Q_i[unsafe_offset=j].cast[DType.float64]()
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        self.G_bar[unsafe_offset=j] += self.get_C(i) * Q_i[unsafe_offset=j].cast[DType.float64]()

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

            var old_alpha_i = self.alpha[unsafe_offset=i]
            var old_alpha_j = self.alpha[unsafe_offset=j]

            if self.y[unsafe_offset=i]!=self.y[unsafe_offset=j]:
                var quad_coef = self.QD[unsafe_offset=i]+self.QD[unsafe_offset=j]+2*Q_i[unsafe_offset=j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-self.G[unsafe_offset=i]-self.G[unsafe_offset=j])/quad_coef
                var diff = self.alpha[unsafe_offset=i] - self.alpha[unsafe_offset=j]
                self.alpha[unsafe_offset=i] += delta
                self.alpha[unsafe_offset=j] += delta

                if(diff > 0):
                    if self.alpha[unsafe_offset=j] < 0:
                        self.alpha[unsafe_offset=j] = 0
                        self.alpha[unsafe_offset=i] = diff
                else:
                    if self.alpha[unsafe_offset=i] < 0:
                        self.alpha[unsafe_offset=i] = 0
                        self.alpha[unsafe_offset=j] = -diff
                if diff > C_i - C_j:
                    if self.alpha[unsafe_offset=i] > C_i:
                        self.alpha[unsafe_offset=i] = C_i
                        self.alpha[unsafe_offset=j] = C_i - diff
                else:
                    if self.alpha[unsafe_offset=j] > C_j:
                        self.alpha[unsafe_offset=j] = C_j
                        self.alpha[unsafe_offset=i] = C_j + diff
            else:
                var quad_coef = self.QD[unsafe_offset=i]+self.QD[unsafe_offset=j]-2*Q_i[unsafe_offset=j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (self.G[unsafe_offset=i]-self.G[unsafe_offset=j])/quad_coef
                var sum = self.alpha[unsafe_offset=i] + self.alpha[unsafe_offset=j]
                self.alpha[unsafe_offset=i] -= delta
                self.alpha[unsafe_offset=j] += delta

                if sum > C_i:
                    if self.alpha[unsafe_offset=i] > C_i:
                        self.alpha[unsafe_offset=i] = C_i
                        self.alpha[unsafe_offset=j] = sum - C_i
                else:
                    if self.alpha[unsafe_offset=j] < 0:
                        self.alpha[unsafe_offset=j] = 0
                        self.alpha[unsafe_offset=i] = sum
                if sum > C_j:
                    if self.alpha[unsafe_offset=j] > C_j:
                        self.alpha[unsafe_offset=j] = C_j
                        self.alpha[unsafe_offset=i] = sum - C_j
                else:
                    if self.alpha[unsafe_offset=i] < 0:
                        self.alpha[unsafe_offset=i] = 0
                        self.alpha[unsafe_offset=j] = sum

            # update G

            var delta_alpha_i = self.alpha[unsafe_offset=i] - old_alpha_i
            var delta_alpha_j = self.alpha[unsafe_offset=j] - old_alpha_j

            for k in range(self.active_size):
                self.G[unsafe_offset=k] += Q_i[unsafe_offset=k].cast[DType.float64]()*delta_alpha_i + Q_j[unsafe_offset=k].cast[DType.float64]()*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] -= C_i * Q_i[unsafe_offset=k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] += C_i * Q_i[unsafe_offset=k].cast[DType.float64]()

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] -= C_j * Q_j[unsafe_offset=k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar[unsafe_offset=k] += C_j * Q_j[unsafe_offset=k].cast[DType.float64]()

        if iter >= max_iter:
            if(self.active_size < self.l):
                # reconstruct the whole gradient to calculate objective value
                self.reconstruct_gradient(Q)
                self.active_size = self.l
            print("\nWARNING: reaching max number of iterations\n")

        # calculate rho

        self.si.rho = self.calculate_rho()

        # calculate objective value
        var v = 0.0
        for i in range(self.l):
            v += self.alpha[unsafe_offset=i] * (self.G[unsafe_offset=i] + self.p[unsafe_offset=i])

        self.si.obj = v/2

        # put back the solution

        for i in range(self.l):
            alpha_[unsafe_offset=self.active_set[unsafe_offset=i]] = self.alpha[unsafe_offset=i]

        # juggle everything back

        #for i in range(self.l):
        #   while self.active_set[i] != i:
        #       self.swap_index(i,self.active_set[i])
        #       # or Q.swap_index(i,self.active_set[i])


        self.si.upper_bound_p = Cp
        self.si.upper_bound_n = Cn

        self.p.unsafe_free()
        self.y.unsafe_free()
        self.alpha.unsafe_free()
        self.alpha_status.unsafe_free()
        self.active_set.unsafe_free()
        self.G.unsafe_free()
        self.G_bar.unsafe_free()

    # return 1 if already optimal, return 0 otherwise
    def select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int:
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
            if self.y[unsafe_offset=t]== 1:
                if not self.is_upper_bound(t):
                    if -self.G[unsafe_offset=t] >= Gmaxp:
                        Gmaxp = -self.G[unsafe_offset=t]
                        Gmaxp_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G[unsafe_offset=t] >= Gmaxn:
                        Gmaxn = self.G[unsafe_offset=t]
                        Gmaxn_idx = t

        var i_p = Gmaxp_idx
        var i_n = Gmaxn_idx
        var Q_ip = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        var Q_in = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        if i_p != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_ip = Q.get_Q(i_p,self.active_size)
        if i_n != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_in = Q.get_Q(i_n,self.active_size)

        for j in range(self.active_size):
            if self.y[unsafe_offset=j]==1:
                if not self.is_lower_bound(j):
                    var grad_diff=Gmaxp+self.G[unsafe_offset=j]
                    if self.G[unsafe_offset=j] >= Gmaxp2:
                        Gmaxp2 = self.G[unsafe_offset=j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[unsafe_offset=i_p]+self.QD[unsafe_offset=j]-2.0*Q_ip[unsafe_offset=j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    var grad_diff= Gmaxn-self.G[unsafe_offset=j]
                    if -self.G[unsafe_offset=j] >= Gmaxn2:
                        Gmaxn2 = -self.G[unsafe_offset=j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD[unsafe_offset=i_n]+self.QD[unsafe_offset=j]+2.0*Q_in[unsafe_offset=j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff

        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < self.eps or Gmin_idx == -1:
            return 1

        if self.y[unsafe_offset=Gmin_idx] == 1:
            out_i = Gmaxp_idx
        else:
            out_i = Gmaxn_idx
        out_j = Gmin_idx
        return 0

    def be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64, Gmax3: Float64, Gmax4: Float64) -> Bool:
        if self.is_upper_bound(i):
            if self.y[unsafe_offset=i]==1:
                return -self.G[unsafe_offset=i] > Gmax1
            else:
                return -self.G[unsafe_offset=i] > Gmax4
        elif self.is_lower_bound(i):
            if self.y[unsafe_offset=i]==1:
                return self.G[unsafe_offset=i] > Gmax2
            else:
                return self.G[unsafe_offset=i] > Gmax3
        else:
            return False

    def do_shrinking[QM: QMatrix](mut self, mut Q: QM):
        var Gmax1 = -math.inf[DType.float64]()		# max { -y_i * grad(f)_i | i in I_up(\alpha) }
        var Gmax2 = -math.inf[DType.float64]()		# max { y_i * grad(f)_i | i in I_low(\alpha) }
        var Gmax3 = -math.inf[DType.float64]()	    # max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
        var Gmax4 = -math.inf[DType.float64]()	    # max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if not self.is_upper_bound(i):
                if self.y[unsafe_offset=i]==1:
                    if -self.G[unsafe_offset=i] > Gmax1:
                        Gmax1 = -self.G[unsafe_offset=i]
                else:
                    if -self.G[unsafe_offset=i] > Gmax4:
                        Gmax4 = -self.G[unsafe_offset=i]
            if not self.is_lower_bound(i):
                if self.y[unsafe_offset=i]==1:
                    if self.G[unsafe_offset=i] > Gmax2:
                        Gmax2 = self.G[unsafe_offset=i]
                else:
                    if self.G[unsafe_offset=i] > Gmax3:
                        Gmax3 = self.G[unsafe_offset=i]

        if self.unshrink == False and max(Gmax1+Gmax2,Gmax3+Gmax4) <= self.eps*10:
            self.unshrink = True
            self.reconstruct_gradient(Q)
            self.active_size = self.l

        var i = 0
        while i < self.active_size:
            if self.be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2, Gmax3, Gmax4):
                        self.swap_index(Q, i,self.active_size)
                        break
                    self.active_size -= 1
            i += 1

    def calculate_rho(mut self) -> Float64:
        var nr_free1 = 0
        var nr_free2 = 0
        var ub1 = math.inf[DType.float64]()
        var ub2 = math.inf[DType.float64]()
        var lb1 = -math.inf[DType.float64]()
        var lb2 = -math.inf[DType.float64]()
        var sum_free1 = 0.0
        var sum_free2 = 0.0

        for i in range(self.active_size):
            if self.y[unsafe_offset=i]==1:
                if self.is_upper_bound(i):
                    lb1 = max(lb1,self.G[unsafe_offset=i])
                elif self.is_lower_bound(i):
                    ub1 = min(ub1,self.G[unsafe_offset=i])
                else:
                    nr_free1 += 1
                    sum_free1 += self.G[unsafe_offset=i]
            else:
                if self.is_upper_bound(i):
                    lb2 = max(lb2,self.G[unsafe_offset=i])
                elif self.is_lower_bound(i):
                    ub2 = min(ub2,self.G[unsafe_offset=i])
                else:
                    nr_free2 += 1
                    sum_free2 += self.G[unsafe_offset=i]

        var r1: Float64
        var r2: Float64
        if nr_free1 > 0:
            r1 = sum_free1/Float64(nr_free1)
        else:
            r1 = (ub1+lb1)/2

        if nr_free2 > 0:
            r2 = sum_free2/Float64(nr_free2)
        else:
            r2 = (ub2+lb2)/2

        self.si.r = (r1+r2)/2
        return (r1-r2)/2

#
# Q matrices for various formulations
#
struct SVC_Q(QMatrix):
    var y: Pointer[Int8, MutUntrackedOrigin]
    var cache: Cache
    var QD: Pointer[Float64, MutUntrackedOrigin]

    var _self: kernel_params

    var kernel_function: def(kernel_params, Int, Int) thin -> Float64

    @always_inline
    def __init__(out self, prob: svm_problem, param: svm_parameter, y_: OptionalPointer[Int8, MutUntrackedOrigin]):
        # Kernel
        var x = alloc[Pointer[svm_node, MutUntrackedOrigin]](prob.l)
        unsafe_memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: Pointer[Float64, MutUntrackedOrigin]
        if param.kernel_type == svm_parameter.RBF:
            x_square = alloc[Float64](prob.l)
            for i in range(prob.l):
                x_square[unsafe_offset=i] = dot(x[unsafe_offset=i], x[unsafe_offset=i])
        else:
            x_square = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        elif self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        elif self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        elif self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        elif self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear
        ##
        self.y = alloc[Int8](prob.l)
        unsafe_memcpy(dest=self.y, src=y_.value(), count=prob.l)

        self.cache = Cache(prob.l, UInt(Int(param.cache_size*(1<<20))))

        self.QD = alloc[Float64](prob.l)
        for i in range(prob.l):
            self.QD[unsafe_offset=i] = self.kernel_function(self._self, i,i)

    def get_Q(mut self, i: Int, _len: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        var data = OptionalPointer[Float32, MutUntrackedOrigin]()
        var start = self.cache.get_data(i, Pointer[OptionalPointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=data))),_len)
        if start < _len:
            @parameter
            def p(j: Int):
                data.value()[unsafe_offset=j+start] = ((self.y[unsafe_offset=i]*self.y[unsafe_offset=j+start]).cast[DType.float64]()*self.kernel_function(self._self, i,j+start)).cast[DType.float32]()
            parallelize[p](_len - start)
        return data.value()

    def get_QD(self) -> Pointer[Float64, MutUntrackedOrigin]:
        return self.QD

    def swap_index(mut self, i: Int, j: Int):
        self.cache.swap_index(i,j)

        swap(self._self.x[unsafe_offset=i],self._self.x[unsafe_offset=j])
        if self._self.kernel_type == svm_parameter.RBF:
            swap(self._self.x_square[unsafe_offset=i],self._self.x_square[unsafe_offset=j])

        swap(self.y[unsafe_offset=i],self.y[unsafe_offset=j])
        swap(self.QD[unsafe_offset=i],self.QD[unsafe_offset=j])

    def __deinit__(deinit self):
        self._self.x.unsafe_free()
        if self._self.kernel_type == svm_parameter.RBF:
            self._self.x_square.unsafe_free()

        self.y.unsafe_free()
        self.QD.unsafe_free()

struct ONE_CLASS_Q(QMatrix):
    var cache: Cache
    var QD: Pointer[Float64, MutUntrackedOrigin]

    var _self: kernel_params

    var kernel_function: def(kernel_params, Int, Int) thin -> Float64

    @always_inline
    def __init__(out self, prob: svm_problem, param: svm_parameter):
        # Kernel
        var x = alloc[Pointer[svm_node, MutUntrackedOrigin]](prob.l)
        unsafe_memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: Pointer[Float64, MutUntrackedOrigin]
        if param.kernel_type == svm_parameter.RBF:
            x_square = alloc[Float64](prob.l)
            for i in range(prob.l):
                x_square[unsafe_offset=i] = dot(x[unsafe_offset=i], x[unsafe_offset=i])
        else:
            x_square = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        elif self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        elif self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        elif self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        elif self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear
        ##
        self.cache = Cache(prob.l, UInt(Int(param.cache_size*(1<<20))))

        self.QD = alloc[Float64](prob.l)
        for i in range(prob.l):
            self.QD[unsafe_offset=i] = self.kernel_function(self._self, i,i)

    def get_Q(mut self, i: Int, _len: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        var data = OptionalPointer[Float32, MutUntrackedOrigin]()
        var start = self.cache.get_data(i, Pointer[OptionalPointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=data))),_len)
        if start < _len:
            for j in range(start, _len):
                data.value()[unsafe_offset=j] = self.kernel_function(self._self, i,j).cast[DType.float32]()
        return data.value()

    def get_QD(self) -> Pointer[Float64, MutUntrackedOrigin]:
        return self.QD

    def swap_index(mut self, i: Int, j: Int):
        self.cache.swap_index(i,j)

        swap(self._self.x[unsafe_offset=i],self._self.x[unsafe_offset=j])
        if self._self.kernel_type == svm_parameter.RBF:
            swap(self._self.x_square[unsafe_offset=i],self._self.x_square[unsafe_offset=j])

        swap(self.QD[unsafe_offset=i],self.QD[unsafe_offset=j])

    def __deinit__(deinit self):
        self._self.x.unsafe_free()
        if self._self.kernel_type == svm_parameter.RBF:
            self._self.x_square.unsafe_free()

        self.QD.unsafe_free()

struct SVR_Q(QMatrix):
    var l: Int
    var cache: Cache
    var sign: Pointer[Int8, MutUntrackedOrigin]
    var index: Pointer[Int, MutUntrackedOrigin]
    var next_buffer: Int
    var buffer: Array[OptionalPointer[Float32, MutUntrackedOrigin], 2]
    var QD: Pointer[Float64, MutUntrackedOrigin]

    var _self: kernel_params

    var kernel_function: def(kernel_params, Int, Int) thin -> Float64

    @always_inline
    def __init__(out self, prob: svm_problem, param: svm_parameter):
        # Kernel
        var x = alloc[Pointer[svm_node, MutUntrackedOrigin]](prob.l)
        unsafe_memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: Pointer[Float64, MutUntrackedOrigin]
        if param.kernel_type == svm_parameter.RBF:
            x_square = alloc[Float64](prob.l)
            for i in range(prob.l):
                x_square[unsafe_offset=i] = dot(x[unsafe_offset=i], x[unsafe_offset=i])
        else:
            x_square = Pointer[Float64, MutUntrackedOrigin].unsafe_dangling()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)

        if self._self.kernel_type == svm_parameter.LINEAR:
            self.kernel_function = kernel_linear
        elif self._self.kernel_type == svm_parameter.POLY:
            self.kernel_function = kernel_poly
        elif self._self.kernel_type == svm_parameter.RBF:
            self.kernel_function = kernel_rbf
        elif self._self.kernel_type == svm_parameter.SIGMOID:
            self.kernel_function = kernel_sigmoid
        elif self._self.kernel_type == svm_parameter.PRECOMPUTED:
            self.kernel_function = kernel_precomputed
        else:
            self.kernel_function = kernel_linear
        ##
        self.l = prob.l
        self.cache = Cache(self.l, UInt(Int(param.cache_size*(1<<20))))
        self.QD = alloc[Float64](2*self.l)
        self.sign = alloc[Int8](2*self.l)
        self.index = alloc[Int](2*self.l)
        for k in range(self.l):
            self.sign[unsafe_offset=k] = 1
            self.sign[unsafe_offset=k+self.l] = -1
            self.index[unsafe_offset=k] = k
            self.index[unsafe_offset=k+self.l] = k
            self.QD[unsafe_offset=k] = self.kernel_function(self._self, k,k)
            self.QD[unsafe_offset=k+self.l] = self.QD[unsafe_offset=k]
        self.buffer: Array[OptionalPointer[Float32, MutUntrackedOrigin], 2] = [alloc[Float32](2*self.l), alloc[Float32](2*self.l)]
        self.next_buffer = 0

    def swap_index(self, i: Int, j: Int):
        swap(self.sign[unsafe_offset=i],self.sign[unsafe_offset=j])
        swap(self.index[unsafe_offset=i],self.index[unsafe_offset=j])
        swap(self.QD[unsafe_offset=i],self.QD[unsafe_offset=j])

    def get_Q(mut self, i: Int, _len: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        var data = OptionalPointer[Float32, MutUntrackedOrigin]()
        var real_i = self.index[unsafe_offset=i]
        if self.cache.get_data(real_i, Pointer[OptionalPointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=data))),self.l) < self.l:
            @parameter
            def p(j: Int):
                data.value()[unsafe_offset=j] = self.kernel_function(self._self, real_i,j).cast[DType.float32]()
            parallelize[p](self.l)
        # reorder and copy
        var buf = self.buffer[self.next_buffer]
        self.next_buffer = 1 - self.next_buffer
        var si = self.sign[unsafe_offset=i]
        for j in range(_len):
            buf.value()[unsafe_offset=j] = si.cast[DType.float32]() * self.sign[unsafe_offset=j].cast[DType.float32]() * data.value()[unsafe_offset=self.index[unsafe_offset=j]]
        return buf.value()

    def get_QD(self) -> Pointer[Float64, MutUntrackedOrigin]:
        return self.QD

    def __deinit__(deinit self):
        self._self.x.unsafe_free()
        if self._self.kernel_type == svm_parameter.RBF:
            self._self.x_square.unsafe_free()

        self.QD.unsafe_free()
        self.sign.unsafe_free()
        self.index.unsafe_free()
        if self.buffer[0]:
            self.buffer[0].value().unsafe_free()
        if self.buffer[1]:
            self.buffer[1].value().unsafe_free()

#
# construct and solve various formulations
#
def solve_c_svc(
    prob: svm_problem, param: svm_parameter,
    alpha: Pointer[Float64, MutUntrackedOrigin], mut si: SolutionInfo, Cp: Float64, Cn: Float64):
    var l = prob.l
    var minus_ones = alloc[Float64](l)
    var y = alloc[Int8](l)

    unsafe_memset_zero(alpha, l)
    for i in range(l):
        minus_ones[unsafe_offset=i] = -1
        if prob.y[unsafe_offset=i] > 0:
            y[unsafe_offset=i] = 1
        else:
            y[unsafe_offset=i] = -1

    var s = Solver()
    var q = SVC_Q(prob,param,y)
    s.Solve(l, q, minus_ones, y,
        alpha, Cp, Cn, param.eps, si, param.shrinking)

    var sum_alpha=0.0
    for i in range(l):
        sum_alpha += alpha[unsafe_offset=i]

    for i in range(l):
        alpha[unsafe_offset=i] *= y[unsafe_offset=i].cast[DType.float64]()

    minus_ones.unsafe_free()
    y.unsafe_free()

def solve_nu_svc(
    prob: svm_problem, param: svm_parameter,
    alpha: Pointer[Float64, MutUntrackedOrigin], mut si: SolutionInfo):
    var l = prob.l
    var nu = param.nu

    var y = alloc[Int8](l)

    for i in range(l):
        if prob.y[unsafe_offset=i]>0:
            y[unsafe_offset=i] = 1
        else:
            y[unsafe_offset=i] = -1

    var sum_pos = nu*Float64(l)/2
    var sum_neg = nu*Float64(l)/2

    for i in range(l):
        if y[unsafe_offset=i] == 1:
            alpha[unsafe_offset=i] = min(1.0,sum_pos)
            sum_pos -= alpha[unsafe_offset=i]
        else:
            alpha[unsafe_offset=i] = min(1.0,sum_neg)
            sum_neg -= alpha[unsafe_offset=i]

    var zeros = alloc[Float64](l)
    unsafe_memset_zero(zeros, l)

    var s = Solver_NU()
    var q = SVC_Q(prob,param,y)
    s.Solve(l, q, zeros, y,
        alpha, 1.0, 1.0, param.eps, si, param.shrinking)
    var r = si.r

    for i in range(l):
        alpha[unsafe_offset=i] *= y[unsafe_offset=i].cast[DType.float64]()/r

    si.rho /= r
    si.obj /= (r*r)
    si.upper_bound_p = 1/r
    si.upper_bound_n = 1/r 

    y.unsafe_free()
    zeros.unsafe_free()

def solve_one_class(
    prob: svm_problem, param: svm_parameter,
    alpha: Pointer[Float64, MutUntrackedOrigin], mut si: SolutionInfo):
    var l = prob.l
    var zeros = alloc[Float64](l)
    var ones = alloc[Int8](l)

    var n = Int(param.nu*Float64(prob.l))	# # of alpha's at upper bound

    for i in range(n):
        alpha[unsafe_offset=i] = 1
    if n<prob.l:
        alpha[unsafe_offset=n] = param.nu * Float64(prob.l) - Float64(n)
    unsafe_memset_zero(alpha.unsafe_offset(n+1), l - (n+1))

    unsafe_memset_zero(zeros, l)
    unsafe_memset(ones, 1, l)

    var s = Solver()
    var q = ONE_CLASS_Q(prob,param)
    s.Solve(l, q, zeros, ones,
        alpha, 1.0, 1.0, param.eps, si, param.shrinking)

    zeros.unsafe_free()
    ones.unsafe_free()

def solve_epsilon_svr(
    prob: svm_problem, param: svm_parameter,
    alpha: Pointer[Float64, MutUntrackedOrigin], mut si: SolutionInfo):
    var l = prob.l
    var alpha2 = alloc[Float64](2*l)
    var linear_term = alloc[Float64](2*l)
    var y = alloc[Int8](2*l)

    for i in range(l):
        alpha2[unsafe_offset=i] = 0
        linear_term[unsafe_offset=i] = param.p - prob.y[unsafe_offset=i]
        y[unsafe_offset=i] = 1

        alpha2[unsafe_offset=i+l] = 0
        linear_term[unsafe_offset=i+l] = param.p + prob.y[unsafe_offset=i]
        y[unsafe_offset=i+l] = -1

    var s = Solver()
    var q = SVR_Q(prob,param)
    s.Solve(2*l, q, linear_term, y,
        alpha2, param.C, param.C, param.eps, si, param.shrinking)

    var sum_alpha = 0.0
    for i in range(l):
        alpha[unsafe_offset=i] = alpha2[unsafe_offset=i] - alpha2[unsafe_offset=i+l]
        sum_alpha += abs(alpha[unsafe_offset=i])

    alpha2.unsafe_free()
    linear_term.unsafe_free()
    y.unsafe_free()

def solve_nu_svr(
    prob: svm_problem, param: svm_parameter,
    alpha: Pointer[Float64, MutUntrackedOrigin], mut si: SolutionInfo):
    var l = prob.l
    var C = param.C
    var alpha2 = alloc[Float64](2*l)
    var linear_term = alloc[Float64](2*l)
    var y = alloc[Int8](2*l)

    var sum = C * param.nu * Float64(l) / 2
    for i in range(l):
        alpha2[unsafe_offset=i] = alpha2[unsafe_offset=i+l] = min(sum,C)
        sum -= alpha2[unsafe_offset=i]

        linear_term[unsafe_offset=i] = - prob.y[unsafe_offset=i]
        y[unsafe_offset=i] = 1

        linear_term[unsafe_offset=i+l] = prob.y[unsafe_offset=i]
        y[unsafe_offset=i+l] = -1

    var s = Solver_NU()
    var q = SVR_Q(prob,param)
    s.Solve(2*l, q, linear_term, y,
        alpha2, C, C, param.eps, si, param.shrinking)

    for i in range(l):
        alpha[unsafe_offset=i] = alpha2[unsafe_offset=i] - alpha2[unsafe_offset=i+l]

    alpha2.unsafe_free()
    linear_term.unsafe_free()
    y.unsafe_free()

#
# decision_function
#
@fieldwise_init
struct decision_function(RegisterPassable, Copyable):
    var alpha: OptionalPointer[Float64, MutUntrackedOrigin]
    var rho: Float64

def svm_train_one(
    prob: svm_problem, param: svm_parameter,
    Cp: Float64, Cn: Float64) -> decision_function:
    var alpha = alloc[Float64](prob.l)
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
        if abs(alpha[unsafe_offset=i]) > 0:
            nSV += 1
            if prob.y[unsafe_offset=i] > 0:
                if abs(alpha[unsafe_offset=i]) >= si.upper_bound_p:
                    nBSV += 1
            else:
                if abs(alpha[unsafe_offset=i]) >= si.upper_bound_n:
                    nBSV += 1

    return decision_function(alpha=alpha, rho=si.rho)

# Platt's binary SVM Probablistic Output: an improvement from Lin et al.
def sigmoid_train(
    l: Int, dec_values: Pointer[Float64, MutUntrackedOrigin], labels: Pointer[Float64, MutUntrackedOrigin],
    mut A: Float64, mut B: Float64):
    var prior1 = 0.0
    var prior0 = 0.0

    for i in range(l):
        if labels[unsafe_offset=i] > 0:
            prior1 += 1
        else:
            prior0 += 1

    var max_iter=100	# Maximal number of iterations
    var min_step=1e-10	# Minimal step taken in line search
    var sigma=1e-12	# For numerically strict PD of Hessian
    var eps=1e-5
    var hiTarget=(prior1+1.0)/(prior1+2.0)
    var loTarget=1/(prior0+2.0)
    var t= alloc[Float64](l)
    var fApB: Float64; p: Float64; q: Float64; h11: Float64; h22: Float64; h21: Float64; g1: Float64; g2: Float64; det: Float64; dA: Float64; dB: Float64; gd: Float64; stepsize: Float64
    var newA: Float64; newB: Float64; newf: Float64; d1: Float64; d2: Float64
    var iter: Int

    # Initial Point and Initial Fun Value
    A=0.0
    B=math.log((prior0+1.0)/(prior1+1.0))
    var fval = 0.0

    for i in range(l):
        if (labels[unsafe_offset=i]>0):
            t[unsafe_offset=i]=hiTarget
        else:
            t[unsafe_offset=i]=loTarget
        fApB = dec_values[unsafe_offset=i]*A+B
        if fApB>=0:
            fval += t[unsafe_offset=i]*fApB + math.log(1+math.exp(-fApB))
        else:
            fval += (t[unsafe_offset=i] - 1)*fApB +math.log(1+math.exp(fApB))

    iter = 0
    while iter<max_iter:
        # Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma # numerically ensures strict PD
        h22=sigma
        h21=0.0; g1=0.0; g2=0.0
        for i in range(l):
            fApB = dec_values[unsafe_offset=i]*A+B
            if (fApB >= 0):
                p=math.exp(-fApB)/(1.0+math.exp(-fApB))
                q=1.0/(1.0+math.exp(-fApB))
            else:
                p=1.0/(1.0+math.exp(fApB))
                q=math.exp(fApB)/(1.0+math.exp(fApB))

            d2=p*q
            h11+=dec_values[unsafe_offset=i]*dec_values[unsafe_offset=i]*d2
            h22+=d2
            h21+=dec_values[unsafe_offset=i]*d2
            d1=t[unsafe_offset=i]-p
            g1+=dec_values[unsafe_offset=i]*d1
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
                fApB = dec_values[unsafe_offset=i]*newA+newB
                if fApB >= 0:
                    newf += t[unsafe_offset=i]*fApB + math.log(1+math.exp(-fApB))
                else:
                    newf += (t[unsafe_offset=i] - 1)*fApB +math.log(1+math.exp(fApB))

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
    t.unsafe_free()

def sigmoid_predict(decision_value: Float64, A: Float64, B: Float64) -> Float64:
    var fApB = decision_value*A+B
    # 1-p used later; avoid catastrophic cancellation
    if fApB >= 0:
        return math.exp(-fApB)/(1.0+math.exp(-fApB))
    else:
        return 1.0/(1+math.exp(fApB))

# Method 2 from the multiclass_prob paper by Wu, Lin, and Weng to predict probabilities
def multiclass_probability(k: Int, r: Pointer[Pointer[Float64, MutUntrackedOrigin], MutUntrackedOrigin], p: Pointer[Float64, MutUntrackedOrigin]):
    var max_iter=max(100,k)
    var Q=alloc[Pointer[Float64, MutUntrackedOrigin]](k)
    var Qp=alloc[Float64](k)
    var pQp: Float64
    var eps=0.005/Float64(k)

    for t in range(k):
        p[unsafe_offset=t]=1.0/Float64(k)  # Valid if k = 1
        Q[unsafe_offset=t]=alloc[Float64](k)
        Q[unsafe_offset=t][unsafe_offset=t]=0
        for j in range(t):
            Q[unsafe_offset=t][unsafe_offset=t]+=r[unsafe_offset=j][unsafe_offset=t]*r[unsafe_offset=j][unsafe_offset=t]
            Q[unsafe_offset=t][unsafe_offset=j]=Q[unsafe_offset=j][unsafe_offset=t]
        for j in range(t+1,k):
            Q[unsafe_offset=t][unsafe_offset=t]+=r[unsafe_offset=j][unsafe_offset=t]*r[unsafe_offset=j][unsafe_offset=t]
            Q[unsafe_offset=t][unsafe_offset=j]=-r[unsafe_offset=j][unsafe_offset=t]*r[unsafe_offset=t][unsafe_offset=j]
    var iter = 0
    while iter<max_iter:
        # stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0.0
        for t in range(k):
            Qp[unsafe_offset=t]=0
            for j in range(k):
                Qp[unsafe_offset=t]+=Q[unsafe_offset=t][unsafe_offset=j]*p[unsafe_offset=j]
            pQp+=p[unsafe_offset=t]*Qp[unsafe_offset=t]

        var max_error=0.0
        for t in range(k):
            var error=abs(Qp[unsafe_offset=t]-pQp)
            if error>max_error:
                max_error=error

        if max_error<eps:
            break

        for t in range(k):
            var diff=(-Qp[unsafe_offset=t]+pQp)/Q[unsafe_offset=t][unsafe_offset=t]
            p[unsafe_offset=t]+=diff
            pQp=(pQp+diff*(diff*Q[unsafe_offset=t][unsafe_offset=t]+2*Qp[unsafe_offset=t]))/(1+diff)/(1+diff)
            for j in range(k):
                Qp[unsafe_offset=j]=(Qp[unsafe_offset=j]+diff*Q[unsafe_offset=t][unsafe_offset=j])/(1+diff)
                p[unsafe_offset=j]/=(1+diff)

        iter += 1

    if iter>=max_iter:
        print("Exceeds max_iter in multiclass_prob\n")
    for t in range(k):
        Q[unsafe_offset=t].unsafe_free()
    Q.unsafe_free()
    Qp.unsafe_free()

# Using cross-validation decision values to get parameters for SVC probability estimates
def svm_binary_svc_probability(
    prob: svm_problem, param: svm_parameter,
    Cp: Float64, Cn: Float64, mut probA: Float64, mut probB: Float64):
    var nr_fold = 5
    var perm: Pointer[Int, MutUntrackedOrigin]
    var dec_values = alloc[Float64](prob.l)

    # random shuffle
    try:
        perm = fill_indices(prob.l)
    except:
        perm = alloc[Int](prob.l)
        for i in range(prob.l):
            perm[unsafe_offset=i]=i

    for i in range(prob.l - 1, 0, -1):
        var j = Int(random.random_ui64(0, UInt64(i)))
        swap(perm[unsafe_offset=i],perm[unsafe_offset=j])

    for i in range(nr_fold):
        var begin = i*prob.l//nr_fold
        var end = (i+1)*prob.l//nr_fold
        var k = 0
        var subprob = svm_problem()

        subprob.l = prob.l-(end-begin)
        subprob.x = alloc[Pointer[svm_node, MutUntrackedOrigin]](subprob.l)
        subprob.y = alloc[Float64](subprob.l)

        for j in range(begin):
            subprob.x[unsafe_offset=k] = prob.x[unsafe_offset=perm[unsafe_offset=j]]
            subprob.y[unsafe_offset=k] = prob.y[unsafe_offset=perm[unsafe_offset=j]]
            k += 1

        for j in range(end, prob.l):
            subprob.x[unsafe_offset=k] = prob.x[unsafe_offset=perm[unsafe_offset=j]]
            subprob.y[unsafe_offset=k] = prob.y[unsafe_offset=perm[unsafe_offset=j]]
            k += 1

        var p_count, n_count = 0, 0
        for j in range(k):
            if subprob.y[unsafe_offset=j]>0:
                p_count += 1
            else:
                n_count += 1

        if p_count==0 and n_count==0:
            for j in range(begin, end):
                dec_values[unsafe_offset=perm[unsafe_offset=j]] = 0
        elif p_count > 0 and n_count == 0:
            for j in range(begin, end):
                dec_values[unsafe_offset=perm[unsafe_offset=j]] = 1
        elif p_count == 0 and n_count > 0:
            for j in range(begin, end):
                dec_values[unsafe_offset=perm[unsafe_offset=j]] = -1
        else:
            var subparam = param.copy()
            subparam.probability=0
            subparam.C=1.0
            subparam.nr_weight=2
            subparam.weight_label = alloc[Int](2)
            subparam.weight = alloc[Float64](2)
            subparam.weight_label.value()[unsafe_offset=0]=+1
            subparam.weight_label.value()[unsafe_offset=1]=-1
            subparam.weight.value()[unsafe_offset=0]=Cp
            subparam.weight.value()[unsafe_offset=1]=Cn
            var submodel = svm_train(subprob,subparam)
            for j in range(begin, end):
                _ = svm_predict_values(submodel.value()[],prob.x[unsafe_offset=perm[unsafe_offset=j]],dec_values.unsafe_offset(perm[unsafe_offset=j]))
                # ensure +1 -1 order; reason not using CV subroutine
                dec_values[unsafe_offset=perm[unsafe_offset=j]] *= Float64(submodel.value()[].label.value()[unsafe_offset=0])

            svm_free_and_destroy_model(submodel)
            svm_destroy_param(subparam)

        subprob.x.unsafe_free()
        subprob.y.unsafe_free()

    sigmoid_train(prob.l,dec_values,prob.y,probA,probB)
    dec_values.unsafe_free()
    perm.unsafe_free()

# Binning method from the oneclass_prob paper by Que and Lin to predict the probability as a normal instance (i.e., not an outlier)
def predict_one_class_probability(model: svm_model, dec_value: Float64) -> Float64:
    var prob_estimate = 0.0
    var nr_marks = 10

    if dec_value < model.prob_density_marks.value()[unsafe_offset=0]:
        prob_estimate = 0.001
    elif dec_value > model.prob_density_marks.value()[unsafe_offset=nr_marks-1]:
        prob_estimate = 0.999
    else:
        for i in range(1,nr_marks):
            if dec_value < model.prob_density_marks.value()[unsafe_offset=i]:
                prob_estimate = Float64(i)/Float64(nr_marks)
                break

    return prob_estimate

# Get parameters for one-class SVM probability estimates
def svm_one_class_probability(prob: svm_problem, model: svm_model, prob_density_marks: OptionalPointer[Float64, MutUntrackedOrigin]) -> Int:
    var dec_values = alloc[Float64](prob.l)
    var pred_results = alloc[Float64](prob.l)
    var ret = 0
    var nr_marks = 10

    for i in range(prob.l):
        pred_results[unsafe_offset=i] = svm_predict_values(model,prob.x[unsafe_offset=i], dec_values.unsafe_offset(i))
    @parameter
    def cmp_fn(a: Float64, b: Float64) -> Bool:
        return a < b

    sort[cmp_fn](
        Span(unsafe_ptr=dec_values, length=prob.l)
    )

    var neg_counter=0
    for i in range(prob.l):
        if dec_values[unsafe_offset=i]>=0:
            neg_counter = i
            break

    var pos_counter = prob.l-neg_counter
    if neg_counter<nr_marks//2 or pos_counter<nr_marks//2:
        print("WARNING: number of positive or negative decision values <" + String(nr_marks/2) + "; too few to do a probability estimation.\n")
        ret = -1
    else:
        # Binning by density
        var tmp_marks = alloc[Float64](nr_marks+1)
        var mid = nr_marks//2
        for i in range(mid):
            tmp_marks[unsafe_offset=i] = dec_values[unsafe_offset=i*neg_counter//mid]
        tmp_marks[unsafe_offset=mid] = 0
        for i in range(mid+1, nr_marks+1):
            tmp_marks[unsafe_offset=i] = dec_values[unsafe_offset=neg_counter-1+(i-mid)*pos_counter//mid]

        for i in range(nr_marks):
            prob_density_marks.value()[unsafe_offset=i] = (tmp_marks[unsafe_offset=i]+tmp_marks[unsafe_offset=i+1])/2
        tmp_marks.unsafe_free()

    dec_values.unsafe_free()
    pred_results.unsafe_free()
    return ret

# Return parameter of a Laplace distribution
def svm_svr_probability(prob: svm_problem, param: svm_parameter) -> Float64:
    var nr_fold = 5
    var ymv = alloc[Float64](prob.l)
    var mae = 0.0

    var newparam = param.copy()
    newparam.probability = 0
    svm_cross_validation(prob, newparam, nr_fold, ymv)
    for i in range(prob.l):
        ymv[unsafe_offset=i]=prob.y[unsafe_offset=i]-ymv[unsafe_offset=i]
        mae += abs(ymv[unsafe_offset=i])
    mae /= Float64(prob.l)
    var std=math.sqrt(2*mae*mae)
    var count=0
    mae=0.0
    for i in range(prob.l):
        if abs(ymv[unsafe_offset=i]) > 5*std:
            count=count+1
        else:
            mae+=abs(ymv[unsafe_offset=i])
    mae /= Float64(prob.l-count)

    ymv.unsafe_free()
    return mae

# label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
# perm, length l, must be allocated before calling this subroutine
def svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: OptionalPointer[Int, MutUntrackedOrigin], mut start_ret: OptionalPointer[Int, MutUntrackedOrigin], mut count_ret: OptionalPointer[Int, MutUntrackedOrigin], perm: Pointer[Int, MutUntrackedOrigin]):
    var l = prob.l
    var max_nr_class = 16
    var nr_class = 0
    var label = alloc[Int](max_nr_class)
    var count = alloc[Int](max_nr_class)
    var data_label = alloc[Int](l)

    for i in range(l):
        var this_label = Int(prob.y[unsafe_offset=i])
        var j = 0
        while j<nr_class:
            if this_label == label[unsafe_offset=j]:
                count[unsafe_offset=j] += 1
                break
            j += 1

        data_label[unsafe_offset=i] = j
        if j == nr_class:
            if nr_class == max_nr_class:
                var new = alloc[Int](max_nr_class*2)
                unsafe_memcpy(dest=new, src=label, count=max_nr_class)
                label.unsafe_free()
                label = new
                new = alloc[Int](max_nr_class*2)
                unsafe_memcpy(dest=new, src=count, count=max_nr_class)
                count.unsafe_free()
                count = new
            label[unsafe_offset=nr_class] = this_label
            count[unsafe_offset=nr_class] = 1
            nr_class += 1

    #
    # Labels are ordered by their first occurrence in the training set.
    # However, for two-class sets with -1/+1 labels and -1 appears first,
    # we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
    #
    if nr_class == 2 and label[unsafe_offset=0] == -1 and label[unsafe_offset=1] == 1:
        swap(label[unsafe_offset=0],label[unsafe_offset=1])
        swap(count[unsafe_offset=0],count[unsafe_offset=1])
        for i in range(l):
            if data_label[unsafe_offset=i] == 0:
                data_label[unsafe_offset=i] = 1
            else:
                data_label[unsafe_offset=i] = 0

    var start = alloc[Int](nr_class)
    start[unsafe_offset=0] = 0
    for i in range(1,nr_class):
        start[unsafe_offset=i] = start[unsafe_offset=i-1]+count[unsafe_offset=i-1]
    for i in range(l):
        perm[unsafe_offset=start[unsafe_offset=data_label[unsafe_offset=i]]] = i
        start[unsafe_offset=data_label[unsafe_offset=i]] += 1
    start[unsafe_offset=0] = 0
    for i in range(1,nr_class):
        start[unsafe_offset=i] = start[unsafe_offset=i-1]+count[unsafe_offset=i-1]

    nr_class_ret = nr_class
    label_ret = label
    start_ret = start
    count_ret = count
    data_label.unsafe_free()

#
# Interface functions
#
def svm_train(prob: svm_problem, param: svm_parameter) -> OptionalPointer[svm_model, MutUntrackedOrigin]:
    var model = alloc[svm_model](1)
    model[].param = param.copy()
    model[].free_sv = 0

    if param.svm_type == svm_parameter.ONE_CLASS or param.svm_type == svm_parameter.EPSILON_SVR or param.svm_type == svm_parameter.NU_SVR:
        # regression or one-class-svm
        model[].nr_class = 2
        model[].label = OptionalPointer[Int, MutUntrackedOrigin]()
        model[].nSV = OptionalPointer[Int, MutUntrackedOrigin]()
        model[].probA = OptionalPointer[Float64, MutUntrackedOrigin]()
        model[].probB = OptionalPointer[Float64, MutUntrackedOrigin]()
        model[].prob_density_marks = OptionalPointer[Float64, MutUntrackedOrigin]()
        model[].sv_coef = alloc[OptionalPointer[Float64, MutUntrackedOrigin]](1)

        var f = svm_train_one(prob,param,0,0)
        model[].rho = alloc[Float64](1)
        model[].rho.value()[unsafe_offset=0] = f.rho

        var nSV = 0
        for i in range(prob.l):
            if abs(f.alpha.value()[unsafe_offset=i]) > 0:
                nSV += 1
        model[].l = nSV
        model[].SV = alloc[Pointer[svm_node, MutUntrackedOrigin]](nSV)
        model[].sv_coef.value()[unsafe_offset=0] = alloc[Float64](nSV)
        model[].sv_indices = alloc[Int](nSV)
        var j = 0
        for i in range(prob.l):
            if abs(f.alpha.value()[unsafe_offset=i]) > 0:
                model[].SV.value()[unsafe_offset=j] = prob.x[unsafe_offset=i]
                model[].sv_coef.value()[unsafe_offset=0].value()[unsafe_offset=j] = f.alpha.value()[unsafe_offset=i]
                model[].sv_indices.value()[unsafe_offset=j] = i+1
                j += 1

        if param.probability and (param.svm_type == svm_parameter.EPSILON_SVR or param.svm_type == svm_parameter.NU_SVR):
            model[].probA = alloc[Float64](1)
            model[].probA.value()[unsafe_offset=0] = svm_svr_probability(prob,param)
        elif param.probability and param.svm_type == svm_parameter.ONE_CLASS:
            var nr_marks = 10
            var prob_density_marks = alloc[Float64](nr_marks)

            if svm_one_class_probability(prob,model[],prob_density_marks) == 0:
                model[].prob_density_marks = prob_density_marks
            else:
                prob_density_marks.unsafe_free()

        f.alpha.value().unsafe_free()
    else:
        # classification
        var l = prob.l
        var nr_class = 0
        var label = OptionalPointer[Int, MutUntrackedOrigin]()
        var start = OptionalPointer[Int, MutUntrackedOrigin]()
        var count = OptionalPointer[Int, MutUntrackedOrigin]()
        var perm = alloc[Int](l)

        # group training data of the same class
        svm_group_classes(prob,nr_class,label,start,count,perm)

        var x = alloc[Pointer[svm_node, MutUntrackedOrigin]](l)
        for i in range(l):
            x[unsafe_offset=i] = prob.x[unsafe_offset=perm[unsafe_offset=i]]

        # calculate weighted C
        var weighted_C = alloc[Float64](nr_class)
        for i in range(nr_class):
            weighted_C[unsafe_offset=i] = param.C
        for i in range(param.nr_weight):
            var j = 0
            while j<nr_class:
                if param.weight_label.value()[unsafe_offset=i] == label.value()[unsafe_offset=j]:
                    break
                j += 1
            if j == nr_class:
                print("WARNING: class label", param.weight_label.value()[unsafe_offset=i], "specified in weight is not found\n")
            else:
                weighted_C[unsafe_offset=j] *= param.weight.value()[unsafe_offset=i]

        # train k*(k-1)/2 models

        var nonzero = alloc[Bool](l)
        unsafe_memset_zero(nonzero, l)
        var f = alloc[decision_function](nr_class*(nr_class-1)//2)

        var probA = OptionalPointer[Float64, MutUntrackedOrigin]()
        var probB = OptionalPointer[Float64, MutUntrackedOrigin]()
        if param.probability:
            probA = alloc[Float64](nr_class*(nr_class-1)//2)
            probB = alloc[Float64](nr_class*(nr_class-1)//2)

        var p = 0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                var sub_prob = svm_problem()
                var si = start.value()[unsafe_offset=i]
                var sj = start.value()[unsafe_offset=j]
                var ci = count.value()[unsafe_offset=i]
                var cj = count.value()[unsafe_offset=j]
                sub_prob.l = ci+cj
                sub_prob.x = alloc[Pointer[svm_node, MutUntrackedOrigin]](sub_prob.l)
                sub_prob.y = alloc[Float64](sub_prob.l)

                for k in range(ci):
                    sub_prob.x[unsafe_offset=k] = x[unsafe_offset=si+k]
                    sub_prob.y[unsafe_offset=k] = 1

                for k in range(cj):
                    sub_prob.x[unsafe_offset=ci+k] = x[unsafe_offset=sj+k]
                    sub_prob.y[unsafe_offset=ci+k] = -1

                if param.probability:
                    svm_binary_svc_probability(sub_prob,param,weighted_C[unsafe_offset=i],weighted_C[unsafe_offset=j],probA.value()[unsafe_offset=p],probB.value()[unsafe_offset=p])

                f[unsafe_offset=p] = svm_train_one(sub_prob,param,weighted_C[unsafe_offset=i],weighted_C[unsafe_offset=j])
                for k in range(ci):
                    if not nonzero[unsafe_offset=si+k] and abs(f[unsafe_offset=p].alpha.value()[unsafe_offset=k]) > 0:
                        nonzero[unsafe_offset=si+k] = True
                for k in range(cj):
                    if not nonzero[unsafe_offset=sj+k] and abs(f[unsafe_offset=p].alpha.value()[unsafe_offset=ci+k]) > 0:
                        nonzero[unsafe_offset=sj+k] = True
                sub_prob.x.unsafe_free()
                sub_prob.y.unsafe_free()
                p += 1

        # build output

        model[].nr_class = nr_class

        model[].label = alloc[Int](nr_class)
        for i in range(nr_class):
            model[].label.value()[unsafe_offset=i] = label.value()[unsafe_offset=i]

        model[].rho = alloc[Float64](nr_class*(nr_class-1)//2)
        for i in range(nr_class*(nr_class-1)//2):
            model[].rho.value()[unsafe_offset=i] = f[unsafe_offset=i].rho

        if param.probability:
            model[].probA = alloc[Float64](nr_class*(nr_class-1)//2)
            model[].probB = alloc[Float64](nr_class*(nr_class-1)//2)
            for i in range(nr_class*(nr_class-1)//2):
                model[].probA.value()[unsafe_offset=i] = probA.value()[unsafe_offset=i]
                model[].probB.value()[unsafe_offset=i] = probB.value()[unsafe_offset=i]
        else:
            model[].probA=OptionalPointer[Float64, MutUntrackedOrigin]()
            model[].probB=OptionalPointer[Float64, MutUntrackedOrigin]()

        model[].prob_density_marks=OptionalPointer[Float64, MutUntrackedOrigin]()	# for one-class SVM probabilistic outputs only

        var total_sv = 0
        var nz_count = alloc[Int](nr_class)
        model[].nSV = alloc[Int](nr_class)
        for i in range(nr_class):
            var nSV = 0
            for j in range(count.value()[unsafe_offset=i]):
                if nonzero[unsafe_offset=start.value()[unsafe_offset=i]+j]:
                    nSV += 1
                    total_sv += 1

            model[].nSV.value()[unsafe_offset=i] = nSV
            nz_count[unsafe_offset=i] = nSV

        model[].l = total_sv
        model[].SV = alloc[Pointer[svm_node, MutUntrackedOrigin]](total_sv)
        model[].sv_indices = alloc[Int](total_sv)
        p = 0
        for i in range(l):
            if nonzero[unsafe_offset=i]:
                model[].SV.value()[unsafe_offset=p] = x[unsafe_offset=i]
                model[].sv_indices.value()[unsafe_offset=p] = perm[unsafe_offset=i] + 1
                p += 1

        var nz_start = alloc[Int](nr_class)
        nz_start[unsafe_offset=0] = 0
        for i in range(1, nr_class):
            nz_start[unsafe_offset=i] = nz_start[unsafe_offset=i-1]+nz_count[unsafe_offset=i-1]

        model[].sv_coef = alloc[OptionalPointer[Float64, MutUntrackedOrigin]](nr_class-1)
        for i in range(nr_class-1):
            model[].sv_coef.value()[unsafe_offset=i] = alloc[Float64](total_sv)

        p = 0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                # classifier (i,j): coefficients with
                # i are in sv_coef[j-1][nz_start[i]...],
                # j are in sv_coef[i][nz_start[j]...]

                var si = start.value()[unsafe_offset=i]
                var sj = start.value()[unsafe_offset=j]
                var ci = count.value()[unsafe_offset=i]
                var cj = count.value()[unsafe_offset=j]

                var q = nz_start[unsafe_offset=i]
                for k in range(ci):
                    if nonzero[unsafe_offset=si+k]:
                        model[].sv_coef.value()[unsafe_offset=j-1].value()[unsafe_offset=q] = f[unsafe_offset=p].alpha.value()[unsafe_offset=k]
                        q += 1
                q = nz_start[unsafe_offset=j]
                for k in range(cj):
                    if nonzero[unsafe_offset=sj+k]:
                        model[].sv_coef.value()[unsafe_offset=i].value()[unsafe_offset=q] = f[unsafe_offset=p].alpha.value()[unsafe_offset=ci+k]
                        q += 1
                p += 1

        label.value().unsafe_free()
        if probA:
            probA.value().unsafe_free()
        if probB:
            probB.value().unsafe_free()
        count.value().unsafe_free()
        perm.unsafe_free()
        start.value().unsafe_free()
        x.unsafe_free()
        weighted_C.unsafe_free()
        nonzero.unsafe_free()
        for i in range(nr_class*(nr_class-1)//2):
            f[unsafe_offset=i].alpha.value().unsafe_free()
        f.unsafe_free()
        nz_count.unsafe_free()
        nz_start.unsafe_free()

    return model

# Stratified cross validation
def svm_cross_validation(prob: svm_problem, param: svm_parameter, var nr_fold: Int, target: OptionalPointer[Float64, MutUntrackedOrigin]):
    var fold_start = alloc[Int](nr_fold+1)
    var l = prob.l
    var perm = alloc[Int](l)
    var nr_class = 0
    if nr_fold > l:
        print("WARNING: # folds ("+ String(nr_fold) +") > # data ("+ String(l) +"). Will use # folds = # data instead (i.e., leave-one-out cross validation)\n")
        nr_fold = l

    # stratified cv may not give leave-one-out rate
    # Each class to l folds -> some folds may have zero elements
    if (param.svm_type == svm_parameter.C_SVC or param.svm_type == svm_parameter.NU_SVC) and nr_fold < l:
        var start = OptionalPointer[Int, MutUntrackedOrigin]()
        var label = OptionalPointer[Int, MutUntrackedOrigin]()
        var count = OptionalPointer[Int, MutUntrackedOrigin]()
        svm_group_classes(prob,nr_class,label,start,count,perm)

        # random shuffle and then data grouped by fold using the array perm
        var fold_count = alloc[Int](nr_fold)
        var index = alloc[Int](l)
        unsafe_memcpy(dest=index, src=perm, count=l)
        for c in range(nr_class):
            for i in range(count.value()[unsafe_offset=c] - 1, 0, -1):
                var j = Int(random.random_ui64(0, UInt64(i)))
                swap(index[unsafe_offset=start.value()[unsafe_offset=c]+j],index[unsafe_offset=start.value()[unsafe_offset=c]+i])

        for i in range(nr_fold):
            fold_count[unsafe_offset=i] = 0
            for c in range(nr_class):
                fold_count[unsafe_offset=i]+=(i+1)*count.value()[unsafe_offset=c]//nr_fold-i*count.value()[unsafe_offset=c]//nr_fold

        fold_start[unsafe_offset=0]=0
        for i in range(1, nr_fold+1):
            fold_start[unsafe_offset=i] = fold_start[unsafe_offset=i-1]+fold_count[unsafe_offset=i-1]
        for c in range(nr_class):
            for i in range(nr_fold):
                var begin = start.value()[unsafe_offset=c]+i*count.value()[unsafe_offset=c]//nr_fold
                var end = start.value()[unsafe_offset=c]+(i+1)*count.value()[unsafe_offset=c]//nr_fold
                for j in range(begin, end):
                    perm[unsafe_offset=fold_start[unsafe_offset=i]] = index[unsafe_offset=j]
                    fold_start[unsafe_offset=i] += 1

        fold_start[unsafe_offset=0]=0
        for i in range(1, nr_fold+1):
            fold_start[unsafe_offset=i] = fold_start[unsafe_offset=i-1]+fold_count[unsafe_offset=i-1]
        start.value().unsafe_free()
        label.value().unsafe_free()
        count.value().unsafe_free()
        index.unsafe_free()
        fold_count.unsafe_free()
    else:
        try:
            perm = fill_indices(l)
        except:
            perm = alloc[Int](l)
            for i in range(l):
                perm[unsafe_offset=i]=i
        for i in range(l - 1, 0, -1):
            var j = Int(random.random_ui64(0, UInt64(i)))
            swap(perm[unsafe_offset=i],perm[unsafe_offset=j])

        for i in range(nr_fold+1):
            fold_start[unsafe_offset=i]=i*l//nr_fold

    for i in range(nr_fold):
        var begin = fold_start[unsafe_offset=i]
        var end = fold_start[unsafe_offset=i+1]
        var k = 0
        var subprob = svm_problem()

        subprob.l = l-(end-begin)
        subprob.x = alloc[Pointer[svm_node, MutUntrackedOrigin]](subprob.l)
        subprob.y = alloc[Float64](subprob.l)

        for j in range(begin):
            subprob.x[unsafe_offset=k] = prob.x[unsafe_offset=perm[unsafe_offset=j]]
            subprob.y[unsafe_offset=k] = prob.y[unsafe_offset=perm[unsafe_offset=j]]
            k += 1

        for j in range(end,l):
            subprob.x[unsafe_offset=k] = prob.x[unsafe_offset=perm[unsafe_offset=j]]
            subprob.y[unsafe_offset=k] = prob.y[unsafe_offset=perm[unsafe_offset=j]]
            k += 1

        var submodel = svm_train(subprob,param)
        if param.probability and (param.svm_type == svm_parameter.C_SVC or param.svm_type == svm_parameter.NU_SVC):
            var prob_estimates = alloc[Float64](svm_get_nr_class(submodel.value()[]))
            for j in range(begin, end):
                target.value()[unsafe_offset=perm[unsafe_offset=j]] = svm_predict_probability(submodel.value()[],prob.x[unsafe_offset=perm[unsafe_offset=j]],prob_estimates)
            prob_estimates.unsafe_free()
        else:
            for j in range(begin, end):
                target.value()[unsafe_offset=perm[unsafe_offset=j]] = svm_predict(submodel.value()[],prob.x[unsafe_offset=perm[unsafe_offset=j]])
        svm_free_and_destroy_model(submodel)
        subprob.x.unsafe_free()
        subprob.y.unsafe_free()

    fold_start.unsafe_free()
    perm.unsafe_free()

@always_inline
def svm_get_svm_type(model: svm_model) -> Int:
    return model.param.svm_type

@always_inline
def svm_get_nr_class(model: svm_model) -> Int:
    return model.nr_class

def svm_get_labels(model: svm_model, label: OptionalPointer[Int, MutUntrackedOrigin]):
    if model.label:
        for i in range(model.nr_class):
            label.value()[unsafe_offset=i] = model.label.value()[unsafe_offset=i]

def svm_get_sv_indices(model: svm_model, indices: OptionalPointer[Int, MutUntrackedOrigin]):
    if model.sv_indices:
        unsafe_memcpy(dest=indices.value(), src=model.sv_indices.value(), count=model.l)

@always_inline
def svm_get_nr_sv(model: svm_model) -> Int:
    return model.l

def svm_get_svr_probability(model: svm_model) -> Float64:
    if (model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR) and model.probA:
        return model.probA.value()[unsafe_offset=0]
    else:
        print("Model doesn't contain information for SVR probability inference\n")
        return 0.0

def svm_predict_values(model: svm_model, x: Pointer[svm_node, MutUntrackedOrigin], dec_values: Pointer[Float64, MutUntrackedOrigin]) -> Float64:
    if model.param.svm_type == svm_parameter.ONE_CLASS or model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR:
        var sv_coef = model.sv_coef.value()[unsafe_offset=0]
        var sum = 0.0

        var values = alloc[Float64](model.l)
        @parameter
        def p(i: Int):
            values[unsafe_offset=i] = sv_coef.value()[unsafe_offset=i] * k_function(x,model.SV.value()[unsafe_offset=i],model.param)
        parallelize[p](model.l)
        try:
            sum = reduction.sum(Span(unsafe_ptr=values, length=model.l))
        except e:
            print('Error:', e)
        values.unsafe_free()
        
        sum -= model.rho.value()[unsafe_offset=0]
        dec_values[] = sum

        if model.param.svm_type == svm_parameter.ONE_CLASS:
            return 1.0 if sum>0 else -1
        else:
            return sum

    else:
        var nr_class = model.nr_class
        var l = model.l

        var kvalue = alloc[Float64](l)

        @parameter
        def pv(i: Int):
            kvalue[unsafe_offset=i] = k_function(x,model.SV.value()[unsafe_offset=i],model.param)
        parallelize[pv](l)

        var start = alloc[Int](nr_class)
        start[unsafe_offset=0] = 0
        for i in range(1, nr_class):
            start[unsafe_offset=i] = start[unsafe_offset=i-1]+model.nSV.value()[unsafe_offset=i-1]

        var vote = alloc[Int](nr_class)
        for i in range(nr_class):
            vote[unsafe_offset=i] = 0

        var p=0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                var sum = 0.0
                var si = start[unsafe_offset=i]
                var sj = start[unsafe_offset=j]
                var ci = model.nSV.value()[unsafe_offset=i]
                var cj = model.nSV.value()[unsafe_offset=j]

                var coef1 = model.sv_coef.value()[unsafe_offset=j-1]
                var coef2 = model.sv_coef.value()[unsafe_offset=i]
                for k in range(ci):
                    sum += coef1.value()[unsafe_offset=si+k] * kvalue[unsafe_offset=si+k]
                for k in range(cj):
                    sum += coef2.value()[unsafe_offset=sj+k] * kvalue[unsafe_offset=sj+k]
                sum -= model.rho.value()[unsafe_offset=p]
                dec_values[unsafe_offset=p] = sum

                if dec_values[unsafe_offset=p] > 0:
                    vote[unsafe_offset=i] += 1
                else:
                    vote[unsafe_offset=j] += 1
                p += 1

        var vote_max_idx = 0
        for i in range(1, nr_class):
            if vote[unsafe_offset=i] > vote[unsafe_offset=vote_max_idx]:
                vote_max_idx = i

        kvalue.unsafe_free()
        start.unsafe_free()
        vote.unsafe_free()
        return Float64(model.label.value()[unsafe_offset=vote_max_idx])

def svm_predict(model: svm_model, x: Pointer[svm_node, MutUntrackedOrigin]) -> Float64:
    var nr_class = model.nr_class
    var dec_values: Pointer[Float64, MutUntrackedOrigin]
    if model.param.svm_type == svm_parameter.ONE_CLASS or model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR:
        dec_values = alloc[Float64](1)
    else:
        dec_values = alloc[Float64](nr_class*(nr_class-1)//2)
    var pred_result = svm_predict_values(model, x, dec_values)
    dec_values.unsafe_free()
    return pred_result

def svm_predict_probability(model: svm_model, x: Pointer[svm_node, MutUntrackedOrigin], prob_estimates: Pointer[Float64, MutUntrackedOrigin]) -> Float64:
    if (model.param.svm_type == svm_parameter.C_SVC or model.param.svm_type == svm_parameter.NU_SVC) and model.probA and model.probB:
        var nr_class = model.nr_class
        var dec_values = alloc[Float64](nr_class*(nr_class-1)//2)
        _ = svm_predict_values(model, x, dec_values)

        var min_prob=1e-7
        var pairwise_prob=alloc[Pointer[Float64, MutUntrackedOrigin]](nr_class)
        for i in range(nr_class):
            pairwise_prob[unsafe_offset=i]=alloc[Float64](nr_class)
        var k=0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                pairwise_prob[unsafe_offset=i][unsafe_offset=j]=min(max(sigmoid_predict(dec_values[unsafe_offset=k],model.probA.value()[unsafe_offset=k],model.probB.value()[unsafe_offset=k]),min_prob),1-min_prob)
                pairwise_prob[unsafe_offset=j][unsafe_offset=i]=1-pairwise_prob[unsafe_offset=i][unsafe_offset=j]
                k += 1
        if nr_class == 2:
            prob_estimates[unsafe_offset=0] = pairwise_prob[unsafe_offset=0][unsafe_offset=1]
            prob_estimates[unsafe_offset=1] = pairwise_prob[unsafe_offset=1][unsafe_offset=0]
        else:
            multiclass_probability(nr_class,pairwise_prob,prob_estimates)

        var prob_max_idx = 0
        for i in range(1, nr_class):
            if prob_estimates[unsafe_offset=i] > prob_estimates[unsafe_offset=prob_max_idx]:
                prob_max_idx = i
        for i in range(nr_class):
            pairwise_prob[unsafe_offset=i].unsafe_free()
        dec_values.unsafe_free()
        pairwise_prob.unsafe_free()
        return Float64(model.label.value()[unsafe_offset=prob_max_idx])
    elif model.param.svm_type == svm_parameter.ONE_CLASS and model.prob_density_marks:
        var dec_value = 0.0
        var pred_result = svm_predict_values(model,x,Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=dec_value))))
        prob_estimates[unsafe_offset=0] = predict_one_class_probability(model,dec_value)
        prob_estimates[unsafe_offset=1] = 1-prob_estimates[unsafe_offset=0]
        return pred_result
    else:
        return svm_predict(model, x)

def svm_decision_function(model: svm_model, x: Pointer[svm_node, MutUntrackedOrigin]) -> Tuple[Pointer[Float64, MutUntrackedOrigin], Int]:
    var nr_class = model.nr_class
    var l: Int
    var dec_values: Pointer[Float64, MutUntrackedOrigin]
    if model.param.svm_type == svm_parameter.ONE_CLASS or model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR:
        l = 1
    else:
        l = nr_class*(nr_class-1)//2
    dec_values = alloc[Float64](l)
    _ = svm_predict_values(model, x, dec_values)
    return dec_values, l

def svm_free_model_content(mut model_ptr: svm_model):
    if model_ptr.free_sv and model_ptr.l > 0 and model_ptr.SV:
        model_ptr.SV.value()[unsafe_offset=0].unsafe_free()
    if model_ptr.sv_coef:
        for i in range(model_ptr.nr_class-1):
            model_ptr.sv_coef.value()[unsafe_offset=i].value().unsafe_free()

    model_ptr.SV.value().unsafe_free()
    model_ptr.SV = None

    model_ptr.sv_coef.value().unsafe_free()
    model_ptr.sv_coef = None

    model_ptr.rho.value().unsafe_free()
    model_ptr.rho = None

    model_ptr.label.value().unsafe_free()
    model_ptr.label = None

    if model_ptr.probA:
        model_ptr.probA.value().unsafe_free()
    model_ptr.probA = None

    if model_ptr.probB:
        model_ptr.probB.value().unsafe_free()
    model_ptr.probB = None

    if model_ptr.prob_density_marks:
        model_ptr.prob_density_marks.value().unsafe_free()
    model_ptr.prob_density_marks = None

    model_ptr.sv_indices.value().unsafe_free()
    model_ptr.sv_indices = None

    model_ptr.nSV.value().unsafe_free()
    model_ptr.nSV = None

def svm_free_and_destroy_model(mut model_ptr_ptr: OptionalPointer[svm_model, MutUntrackedOrigin]):
    if model_ptr_ptr:
        svm_free_model_content(model_ptr_ptr.value()[])
        model_ptr_ptr.value().unsafe_free()
        model_ptr_ptr = OptionalPointer[svm_model, MutUntrackedOrigin]()

def svm_destroy_param(param: svm_parameter):
    if param.weight_label:
        param.weight_label.value().unsafe_free()
    if param.weight:
        param.weight.value().unsafe_free()

def svm_check_parameter(prob: svm_problem, param: svm_parameter) -> String:
    # svm_type

    var svm_type = param.svm_type
    if svm_type != svm_parameter.C_SVC and svm_type != svm_parameter.NU_SVC and svm_type != svm_parameter.ONE_CLASS and svm_type != svm_parameter.EPSILON_SVR and svm_type != svm_parameter.NU_SVR:
        return "unknown svm type"

    # kernel_type, degree

    var kernel_type = param.kernel_type
    if kernel_type != svm_parameter.LINEAR and kernel_type != svm_parameter.POLY and kernel_type != svm_parameter.RBF and kernel_type != svm_parameter.SIGMOID and kernel_type != svm_parameter.PRECOMPUTED:
        return "unknown kernel type"

    if (kernel_type == svm_parameter.POLY or kernel_type == svm_parameter.RBF or kernel_type == svm_parameter.SIGMOID) and param.gamma < 0:
        return "gamma < 0"

    if kernel_type == svm_parameter.POLY and param.degree < 0:
        return "degree of polynomial kernel < 0"

    # cache_size,eps,C,nu,p,shrinking

    if param.cache_size <= 0:
        return "cache_size <= 0"

    if param.eps <= 0:
        return "eps <= 0"

    if svm_type == svm_parameter.C_SVC or svm_type == svm_parameter.EPSILON_SVR or svm_type == svm_parameter.NU_SVR:
        if param.C <= 0:
            return "C <= 0"

    if svm_type == svm_parameter.NU_SVC or svm_type == svm_parameter.ONE_CLASS or svm_type == svm_parameter.NU_SVR:
        if param.nu <= 0 or param.nu > 1:
            return "nu <= 0 or nu > 1"

    if svm_type == svm_parameter.EPSILON_SVR:
        if param.p < 0:
            return "p < 0"

    if param.shrinking != 0 and param.shrinking != 1:
        return "shrinking != 0 and shrinking != 1"

    if param.probability != 0 and param.probability != 1:
        return "probability != 0 and probability != 1"


    # check whether nu-svc is feasible

    if svm_type == svm_parameter.NU_SVC:
        var l = prob.l
        var max_nr_class = 16
        var nr_class = 0
        var label = alloc[Int](max_nr_class)
        var count = alloc[Int](max_nr_class)

        for i in range(l):
            var this_label = Int(prob.y[unsafe_offset=i])
            var j = 0
            while j<nr_class:
                if this_label == label[unsafe_offset=j]:
                    count[unsafe_offset=j] += 1
                    break
                j += 1
            if j == nr_class:
                if nr_class == max_nr_class:
                    var new = alloc[Int](max_nr_class*2)
                    unsafe_memcpy(dest=new, src=label, count=max_nr_class)
                    label.unsafe_free()
                    label = new
                    new = alloc[Int](max_nr_class*2)
                    unsafe_memcpy(dest=new, src=count, count=max_nr_class)
                    count.unsafe_free()
                    count = new
                label[unsafe_offset=nr_class] = this_label
                count[unsafe_offset=nr_class] = 1
                nr_class += 1

        for i in range(nr_class):
            var n1 = count[unsafe_offset=i]
            for j in range(i+1, nr_class):
                var n2 = count[unsafe_offset=j]
                if param.nu*Float64(n1+n2)/2 > Float64(min(n1,n2)):
                    label.unsafe_free()
                    count.unsafe_free()
                    return "specified nu is infeasible"

        label.unsafe_free()
        count.unsafe_free()

    return ""

def svm_check_probability_model(model: svm_model) -> Bool:
    return
        ((model.param.svm_type == svm_parameter.C_SVC or model.param.svm_type == svm_parameter.NU_SVC) and
        model.probA and model.probB) or
        (model.param.svm_type == svm_parameter.ONE_CLASS and model.prob_density_marks) or
        ((model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR) and
        model.probA)
