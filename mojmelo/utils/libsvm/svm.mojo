# Re-implementation of libsvm, a library for support vector machines by Chih-Chung Chang and Chih-Jen Lin (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) with some modifications.

from std.memory import memcpy, memset_zero
from .svm_node import svm_node
from .svm_parameter import svm_parameter
from .svm_problem import svm_problem
from .svm_model import svm_model
from std.sys import size_of
import std.math as math
from std.algorithm import parallelize, reduction
from mojmelo.utils.utils import fill_indices
from mojmelo.SVM import LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
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
def dot(var px: OptionalUnsafePointer[svm_node, MutExternalOrigin], var py: OptionalUnsafePointer[svm_node, MutExternalOrigin]) -> Float64:
    var sum = 0.0
    while px.value()[].index != -1 and py.value()[].index != -1:
        if px.value()[].index == py.value()[].index:
            sum += px.value()[].value * py.value()[].value
            px.value() += 1
            py.value() += 1
        else:
            if px.value()[].index > py.value()[].index:
                py.value() += 1
            else:
                px.value() += 1

    return sum

@fieldwise_init
struct kernel_params(RegisterPassable):
    var x: OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]
    var x_square: OptionalUnsafePointer[Float64, MutExternalOrigin]
    # svm_parameter
    var kernel_type: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64

def k_function(var x: OptionalUnsafePointer[svm_node, MutExternalOrigin], var y: OptionalUnsafePointer[svm_node, MutExternalOrigin], param: svm_parameter) -> Float64:
    if param.kernel_type == LINEAR:
        return dot(x,y)
    if param.kernel_type == POLY:
        return powi(param.gamma*dot(x,y)+param.coef0,param.degree)
    if param.kernel_type == RBF:
        var sum = 0.0
        while x.value()[].index != -1 and y.value()[].index !=-1:
            if x.value()[].index == y.value()[].index:
                var d = x.value()[].value - y.value()[].value
                sum += d*d
                x.value() += 1
                y.value() += 1
            else:
                if x.value()[].index > y.value()[].index:
                    sum += y.value()[].value * y.value()[].value
                    y.value() += 1
                else:
                    sum += x.value()[].value * x.value()[].value
                    x.value() += 1

        while x.value()[].index != -1:
            sum += x.value()[].value * x.value()[].value
            x.value() += 1

        while y.value()[].index != -1:
            sum += y.value()[].value * y.value()[].value
            y.value() += 1

        return math.exp(-param.gamma*sum)
    if param.kernel_type == SIGMOID:
        return math.tanh(param.gamma*dot(x,y)+param.coef0)
    if param.kernel_type == PRECOMPUTED:  # x: test (validation), y: SV
        return x.value()[Int(y.value()[].value)].value
    else:
        return 0  # Unreachable

@always_inline
def kernel_linear(k: kernel_params, i: Int, j: Int) -> Float64:
    return dot(k.x.value()[i],k.x.value()[j])
@always_inline
def kernel_poly(k: kernel_params, i: Int, j: Int) -> Float64:
    return powi(k.gamma*dot(k.x.value()[i],k.x.value()[j])+k.coef0,k.degree)
@always_inline
def kernel_rbf(k: kernel_params, i: Int, j: Int) -> Float64:
    return math.exp(-k.gamma*(k.x_square.value()[i]+k.x_square.value()[j]-2*dot(k.x.value()[i],k.x.value()[j])))
@always_inline
def kernel_sigmoid(k: kernel_params, i: Int, j: Int) -> Float64:
    return math.tanh(k.gamma*dot(k.x.value()[i],k.x.value()[j])+k.coef0)
@always_inline
def kernel_precomputed(k: kernel_params, i: Int, j: Int) -> Float64:
    return k.x.value()[i].value()[Int(k.x.value()[j].value()[0].value)].value

struct head_t(RegisterPassable):
    var prev: OptionalUnsafePointer[head_t, MutAnyOrigin]
    var next: OptionalUnsafePointer[head_t, MutAnyOrigin]	# a cicular list
    var data: OptionalUnsafePointer[Float32, MutExternalOrigin]
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
    var head: OptionalUnsafePointer[head_t, MutExternalOrigin]
    var lru_head: head_t

    @always_inline
    def __init__(out self, l_: Int, size_: UInt):
        self.l = l_
        self.size = (size_ - UInt(self.l * size_of[head_t]())) // 4
        self.head = alloc[head_t](self.l)
        memset_zero(self.head.value(), self.l) # initialized to 0
        self.size = max(self.size, UInt(2) * UInt(self.l))  # cache must be large enough for two columns
        self.lru_head = head_t()
        self.lru_head.next = self.lru_head.prev = UnsafePointer(to=self.lru_head)

    def __del__(deinit self):
        var h = self.lru_head.next
        while h != UnsafePointer(to=self.lru_head):
            if h.value()[].data:
                h.value()[].data.value().free()
            h = h.value()[].next
        if self.head:
            self.head.value().free()

    def lru_delete(self, h: OptionalUnsafePointer[head_t, MutAnyOrigin]):
        # delete from current location
        h.value()[].prev.value()[].next = h.value()[].next
        h.value()[].next.value()[].prev = h.value()[].prev

    def lru_insert(mut self, h: OptionalUnsafePointer[head_t, MutExternalOrigin]):
        # insert to last position
        h.value()[].next = UnsafePointer(to=self.lru_head)
        h.value()[].prev = self.lru_head.prev
        h.value()[].prev.value()[].next = h.value()
        h.value()[].next.value()[].prev = h.value()

    @always_inline
    def get_data(mut self, index: Int, data: OptionalUnsafePointer[OptionalUnsafePointer[Float32, MutExternalOrigin], MutAnyOrigin], var _len: Int) -> Int:
        var h = self.head.value() + index
        if h[]._len:
            self.lru_delete(h)
        var more = _len - h[]._len

        if more > 0:
            # free old space
            while self.size < UInt(more):
                var old = self.lru_head.next
                self.lru_delete(old)
                old.value()[].data.value().free()
                self.size += UInt(old.value()[]._len)
                old.value()[].data = OptionalUnsafePointer[Float32, MutExternalOrigin]()
                old.value()[]._len = 0

            # allocate new space
            var new = alloc[Float32](_len)
            memcpy(dest=new, src=h[].data, count=h[]._len)
            h[].data.value().free()
            h[].data = new
            self.size -= UInt(more)  # previous while loop guarantees size >= more and subtraction of size_t variable will not underflow
            swap(h[]._len, _len)

        self.lru_insert(h)
        data.value()[] = h[].data
        return _len

    @always_inline
    def swap_index(mut self, var i: Int, var j: Int):
        if i==j:
            return

        if self.head.value()[i]._len:
            self.lru_delete(self.head.value() + i)
        if self.head.value()[j]._len:
            self.lru_delete(self.head.value() + j)
        swap(self.head.value()[i].data,self.head.value()[j].data)
        swap(self.head.value()[i]._len,self.head.value()[j]._len)
        if self.head.value()[i]._len:
            self.lru_insert(self.head.value() + i)
        if self.head.value()[j]._len:
            self.lru_insert(self.head.value() + j)

        if i>j:
            swap(i,j)

        var h = self.lru_head.next
        while h != UnsafePointer(to=self.lru_head):
            if h.value()[]._len > i:
                if(h.value()[]._len > j):
                    swap(h.value()[].data.value()[i],h.value()[].data.value()[j])
                else:
                    # give up
                    self.lru_delete(h)
                    h.value()[].data.value().free()
                    self.size += UInt(h.value()[]._len)
                    h.value()[].data = OptionalUnsafePointer[Float32, MutExternalOrigin]()
                    h.value()[]._len = 0
            h=h.value()[].next

# Kernel evaluation
#
# the static method k_function is for doing single kernel evaluation
# the constructor of Kernel prepares to calculate the l*l kernel matrix
# the member function get_Q is for getting one column from the Q Matrix
#
trait QMatrix:
    def get_Q(mut self, column: Int, _len: Int) -> OptionalUnsafePointer[Float32, MutExternalOrigin]:
        ...
    def get_QD(self) -> OptionalUnsafePointer[Float64, MutExternalOrigin]:
        ...
    def swap_index(mut self, i: Int, j: Int):
        ...

#struct Kernel:
#    var _self: kernel_params
#
#    var kernel_function: def(kernel_params, Int, Int) -> Float64
#
#    @always_inline
#    def __init__(out self, l: Int, x_: OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin], param: svm_parameter):
#        var x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](l)
#        memcpy(dest=x, src=x_, count=l)
#
#        var x_square: OptionalUnsafePointer[Float64, MutExternalOrigin]
#        if param.kernel_type == svm_parameter.RBF:
#            x_square = alloc[Float64](l)
#            for i in range(l):
#                x_square[i] = dot(x[i], x[i])
#        else:
#            x_square = OptionalUnsafePointer[Float64, MutExternalOrigin]()
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
    var y: OptionalUnsafePointer[Int8, MutExternalOrigin]
    var G: OptionalUnsafePointer[Float64, MutExternalOrigin]	# gradient of objective function
    comptime LOWER_BOUND: Int8 = 0
    comptime UPPER_BOUND: Int8 = 1
    comptime FREE: Int8 = 2
    var alpha_status: OptionalUnsafePointer[Int8, MutExternalOrigin]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var QD: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var active_set: OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]
    var G_bar: OptionalUnsafePointer[Float64, MutExternalOrigin]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    @always_inline
    def __init__(out self):
        self.active_size = 0
        self.y = OptionalUnsafePointer[Int8, MutExternalOrigin]()
        self.G = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.alpha_status = OptionalUnsafePointer[Int8, MutExternalOrigin]()
        self.alpha = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.QD = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.eps = 0.0
        self.Cp = 0.0
        self.Cn = 0.0
        self.p = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.active_set = OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]()
        self.G_bar = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.l = 0
        self.unshrink = False

    def get_C(self, i: Int) -> Float64:
        return self.Cp if self.y.value()[i] > 0 else self.Cn

    def update_alpha_status(self, i: Int):
        if self.alpha.value()[i] >= self.get_C(i):
            self.alpha_status.value()[i] = self.UPPER_BOUND
        elif self.alpha.value()[i] <= 0:
            self.alpha_status.value()[i] = self.LOWER_BOUND
        else:
            self.alpha_status.value()[i] = self.FREE

    def is_upper_bound(self, i: Int) -> Bool:
        return self.alpha_status.value()[i] == self.UPPER_BOUND
    def is_lower_bound(self, i: Int) -> Bool:
        return self.alpha_status.value()[i] == self.LOWER_BOUND
    def is_free(self, i: Int) -> Bool:
        return self.alpha_status.value()[i] == self.FREE

    def swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int):
        Q.swap_index(i,j)
        swap(self.y.value()[i], self.y.value()[j])
        swap(self.G.value()[i], self.G.value()[j])
        swap(self.alpha_status.value()[i], self.alpha_status.value()[j])
        swap(self.alpha.value()[i], self.alpha.value()[j])
        swap(self.p.value()[i], self.p.value()[j])
        swap(self.active_set.value()[i], self.active_set.value()[j])
        swap(self.G_bar.value()[i], self.G_bar.value()[j])

    def reconstruct_gradient[QM: QMatrix](self, mut Q: QM):
        # reconstruct inactive elements of G from G_bar and free variables

        if self.active_size == self.l:
            return

        var nr_free = 0

        for j in range(self.active_size, self.l):
            self.G.value()[j] = self.G_bar.value()[j] + self.p.value()[j]

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
                        self.G.value()[i] += self.alpha.value()[j] * Q_i.value()[j].cast[DType.float64]()
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha.value()[i]
                    for j in range(self.active_size, self.l):
                        self.G.value()[j] += alpha_i * Q_i.value()[j].cast[DType.float64]()

    def Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: OptionalUnsafePointer[Float64, MutExternalOrigin], y_: OptionalUnsafePointer[Int8, MutExternalOrigin],
                alpha_: OptionalUnsafePointer[Float64, MutExternalOrigin], Cp: Float64, Cn: Float64, eps: Float64, mut si: SolutionInfo, shrinking: Int):
        self.l = l
        self.QD = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.QD = Q.get_QD()
        self.p = alloc[Float64](self.l)
        memcpy(dest=self.p, src=p_, count=self.l)
        self.y = alloc[Int8](self.l)
        memcpy(dest=self.y, src=y_, count=self.l)
        self.alpha = alloc[Float64](self.l)
        memcpy(dest=self.alpha, src=alpha_, count=self.l)
        self.Cp = Cp
        self.Cn = Cn
        self.eps = eps
        self.unshrink = False

        # initialize alpha_status
        self.alpha_status = alloc[Int8](self.l)
        for i in range(self.l):
            if self.alpha.value()[i] >= (self.Cp if self.y.value()[i] > 0 else self.Cn):
                self.alpha_status.value()[i] = self.UPPER_BOUND
            elif self.alpha.value()[i] <= 0:
                self.alpha_status.value()[i] = self.LOWER_BOUND
            else:
                self.alpha_status.value()[i] = self.FREE

        # initialize active set (for shrinking)
        try:
            self.active_set = fill_indices(self.l)
        except:
            self.active_set = alloc[Scalar[DType.int]](self.l)
            for i in range(Scalar[DType.int](self.l)):
                self.active_set.value()[i] = i
        self.active_size = self.l

        # initialize gradient
        self.G = alloc[Float64](self.l)
        self.G_bar = alloc[Float64](self.l)
        memcpy(dest=self.G, src=self.p, count=self.l)
        memset_zero(self.G_bar.value(), self.l)

        for i in range(self.l):
            if not self.is_lower_bound(i):
                var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha.value()[i]
                for j in range(self.l):
                    self.G.value()[j] += alpha_i*Q_i.value()[j].cast[DType.float64]()
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        self.G_bar.value()[j] += self.get_C(i) * Q_i.value()[j].cast[DType.float64]()

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

            var old_alpha_i = self.alpha.value()[i]
            var old_alpha_j = self.alpha.value()[j]

            if self.y.value()[i]!=self.y.value()[j]:
                var quad_coef = self.QD.value()[i]+self.QD.value()[j]+2*Q_i.value()[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-self.G.value()[i]-self.G.value()[j])/quad_coef
                var diff = self.alpha.value()[i] - self.alpha.value()[j]
                self.alpha.value()[i] += delta
                self.alpha.value()[j] += delta

                if(diff > 0):
                    if self.alpha.value()[j] < 0:
                        self.alpha.value()[j] = 0
                        self.alpha.value()[i] = diff
                else:
                    if self.alpha.value()[i] < 0:
                        self.alpha.value()[i] = 0
                        self.alpha.value()[j] = -diff
                if diff > C_i - C_j:
                    if self.alpha.value()[i] > C_i:
                        self.alpha.value()[i] = C_i
                        self.alpha.value()[j] = C_i - diff
                else:
                    if self.alpha.value()[j] > C_j:
                        self.alpha.value()[j] = C_j
                        self.alpha.value()[i] = C_j + diff
            else:
                var quad_coef = self.QD.value()[i]+self.QD.value()[j]-2*Q_i.value()[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (self.G.value()[i]-self.G.value()[j])/quad_coef
                var sum = self.alpha.value()[i] + self.alpha.value()[j]
                self.alpha.value()[i] -= delta
                self.alpha.value()[j] += delta

                if sum > C_i:
                    if self.alpha.value()[i] > C_i:
                        self.alpha.value()[i] = C_i
                        self.alpha.value()[j] = sum - C_i
                else:
                    if self.alpha.value()[j] < 0:
                        self.alpha.value()[j] = 0
                        self.alpha.value()[i] = sum
                if sum > C_j:
                    if self.alpha.value()[j] > C_j:
                        self.alpha.value()[j] = C_j
                        self.alpha.value()[i] = sum - C_j
                else:
                    if self.alpha.value()[i] < 0:
                        self.alpha.value()[i] = 0
                        self.alpha.value()[j] = sum

            # update G

            var delta_alpha_i = self.alpha.value()[i] - old_alpha_i
            var delta_alpha_j = self.alpha.value()[j] - old_alpha_j

            for k in range(self.active_size):
                self.G.value()[k] += Q_i.value()[k].cast[DType.float64]()*delta_alpha_i + Q_j.value()[k].cast[DType.float64]()*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar.value()[k] -= C_i * Q_i.value()[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar.value()[k] += C_i * Q_i.value()[k].cast[DType.float64]()

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar.value()[k] -= C_j * Q_j.value()[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar.value()[k] += C_j * Q_j.value()[k].cast[DType.float64]()

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
            v += self.alpha.value()[i] * (self.G.value()[i] + self.p.value()[i])

        si.obj = v/2

        # put back the solution

        for i in range(self.l):
            alpha_.value()[self.active_set.value()[i]] = self.alpha.value()[i]

        # juggle everything back

        #for i in range(self.l):
        #    while self.active_set[i] != i:
        #        self.swap_index(i,self.active_set[i])
        #       # or Q.swap_index(i,self.active_set[i])


        si.upper_bound_p = Cp
        si.upper_bound_n = Cn

        self.p.value().free()
        self.y.value().free()
        self.alpha.value().free()
        self.alpha_status.value().free()
        self.active_set.value().free()
        self.G.value().free()
        self.G_bar.value().free()

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
            if self.y.value()[t]== 1:
                if not self.is_upper_bound(t):
                    if -self.G.value()[t] >= Gmax:
                        Gmax = -self.G.value()[t]
                        Gmax_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G.value()[t] >= Gmax:
                        Gmax = self.G.value()[t]
                        Gmax_idx = t

        var i = Gmax_idx
        var Q_i = OptionalUnsafePointer[Float32, MutExternalOrigin]()
        if i != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_i = Q.get_Q(i,self.active_size)

        for j in range(self.active_size):
            if self.y.value()[j]==1:
                if not self.is_lower_bound(j):
                    var grad_diff=Gmax+self.G.value()[j]
                    if self.G.value()[j] >= Gmax2:
                        Gmax2 = self.G.value()[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD.value()[i]+self.QD.value()[j]-2.0*self.y.value()[i].cast[DType.float64]()*Q_i.value()[j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    var grad_diff= Gmax-self.G.value()[j]
                    if -self.G.value()[j] >= Gmax2:
                        Gmax2 = -self.G.value()[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD.value()[i]+self.QD.value()[j]+2.0*self.y.value()[i].cast[DType.float64]()*Q_i.value()[j].cast[DType.float64]()
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
            if self.y.value()[i]==1:
                return -self.G.value()[i] > Gmax1
            else:
                return -self.G.value()[i] > Gmax2
        elif self.is_lower_bound(i):
            if self.y.value()[i]==1:
                return self.G.value()[i] > Gmax2
            else:
                return self.G.value()[i] > Gmax1
        else:
            return False

    def do_shrinking[QM: QMatrix](mut self, mut Q: QM):
        var Gmax1 = -math.inf[DType.float64]()		# max { -y_i * grad(f)_i | i in I_up(\alpha) }
        var Gmax2 = -math.inf[DType.float64]()		# max { y_i * grad(f)_i | i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if self.y.value()[i]==1:
                if not self.is_upper_bound(i):
                    if -self.G.value()[i] >= Gmax1:
                        Gmax1 = -self.G.value()[i]
                if not self.is_lower_bound(i):
                    if self.G.value()[i] >= Gmax2:
                        Gmax2 = self.G.value()[i]
            else:
                if not self.is_upper_bound(i):
                    if -self.G.value()[i] >= Gmax2:
                        Gmax2 = -self.G.value()[i]
                if not self.is_lower_bound(i):
                    if self.G.value()[i] >= Gmax1:
                        Gmax1 = self.G.value()[i]

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
            var yG = self.y.value()[i].cast[DType.float64]()*self.G.value()[i]

            if self.is_upper_bound(i):
                if self.y.value()[i]==-1:
                    ub = min(ub,yG)
                else:
                    lb = max(lb,yG)
            elif self.is_lower_bound(i):
                if self.y.value()[i]==1:
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
    var y: OptionalUnsafePointer[Int8, MutExternalOrigin]
    var G: OptionalUnsafePointer[Float64, MutExternalOrigin]	# gradient of objective function
    comptime LOWER_BOUND: Int8 = 0
    comptime UPPER_BOUND: Int8 = 1
    comptime FREE: Int8 = 2
    var alpha_status: OptionalUnsafePointer[Int8, MutExternalOrigin]	# LOWER_BOUND, UPPER_BOUND, FREE
    var alpha: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var QD: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var eps: Float64
    var Cp: Float64
    var Cn: Float64
    var p: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var active_set: OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]
    var G_bar: OptionalUnsafePointer[Float64, MutExternalOrigin]	# gradient, if we treat free variables as 0
    var l: Int
    var unshrink: Bool

    @always_inline
    def __init__(out self):
        self.si = SolutionInfo()
        self.active_size = 0
        self.y = OptionalUnsafePointer[Int8, MutExternalOrigin]()
        self.G = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.alpha_status = OptionalUnsafePointer[Int8, MutExternalOrigin]()
        self.alpha = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.QD = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.eps = 0.0
        self.Cp = 0.0
        self.Cn = 0.0
        self.p = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.active_set = OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]()
        self.G_bar = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.l = 0
        self.unshrink = False

    def get_C(self, i: Int) -> Float64:
        return self.Cp if self.y.value()[i] > 0 else self.Cn

    def update_alpha_status(self, i: Int):
        if self.alpha.value()[i] >= self.get_C(i):
            self.alpha_status.value()[i] = self.UPPER_BOUND
        elif self.alpha.value()[i] <= 0:
            self.alpha_status.value()[i] = self.LOWER_BOUND
        else:
            self.alpha_status.value()[i] = self.FREE

    def is_upper_bound(self, i: Int) -> Bool:
        return self.alpha_status.value()[i] == self.UPPER_BOUND
    def is_lower_bound(self, i: Int) -> Bool:
        return self.alpha_status.value()[i] == self.LOWER_BOUND
    def is_free(self, i: Int) -> Bool:
        return self.alpha_status.value()[i] == self.FREE

    def swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int):
        Q.swap_index(i,j)
        swap(self.y.value()[i], self.y.value()[j])
        swap(self.G.value()[i], self.G.value()[j])
        swap(self.alpha_status.value()[i], self.alpha_status.value()[j])
        swap(self.alpha.value()[i], self.alpha.value()[j])
        swap(self.p.value()[i], self.p.value()[j])
        swap(self.active_set.value()[i], self.active_set.value()[j])
        swap(self.G_bar.value()[i], self.G_bar.value()[j])

    def reconstruct_gradient[QM: QMatrix](self, mut Q: QM):
        # reconstruct inactive elements of G from G_bar and free variables

        if self.active_size == self.l:
            return

        var nr_free = 0

        for j in range(self.active_size, self.l):
            self.G.value()[j] = self.G_bar.value()[j] + self.p.value()[j]

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
                        self.G.value()[i] += self.alpha.value()[j] * Q_i.value()[j].cast[DType.float64]()
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    var Q_i = Q.get_Q(i,self.l)
                    var alpha_i = self.alpha.value()[i]
                    for j in range(self.active_size, self.l):
                        self.G.value()[j] += alpha_i * Q_i.value()[j].cast[DType.float64]()

    def Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: OptionalUnsafePointer[Float64, MutExternalOrigin], y_: OptionalUnsafePointer[Int8, MutExternalOrigin],
                alpha_: OptionalUnsafePointer[Float64, MutExternalOrigin], Cp: Float64, Cn: Float64, eps: Float64, si: SolutionInfo, shrinking: Int):
        self.si = si
        # Solve
        self.l = l
        self.QD = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        self.QD = Q.get_QD()
        self.p = alloc[Float64](self.l)
        memcpy(dest=self.p, src=p_, count=self.l)
        self.y = alloc[Int8](self.l)
        memcpy(dest=self.y, src=y_, count=self.l)
        self.alpha = alloc[Float64](self.l)
        memcpy(dest=self.alpha, src=alpha_, count=self.l)
        self.Cp = Cp
        self.Cn = Cn
        self.eps = eps
        self.unshrink = False

        # initialize alpha_status
        self.alpha_status = alloc[Int8](self.l)
        for i in range(self.l):
            if self.alpha.value()[i] >= (self.Cp if self.y.value()[i] > 0 else self.Cn):
                self.alpha_status.value()[i] = self.UPPER_BOUND
            elif self.alpha.value()[i] <= 0:
                self.alpha_status.value()[i] = self.LOWER_BOUND
            else:
                self.alpha_status.value()[i] = self.FREE

        # initialize active set (for shrinking)
        try:
            self.active_set = fill_indices(self.l)
        except:
            self.active_set = alloc[Scalar[DType.int]](self.l)
            for i in range(Scalar[DType.int](self.l)):
                self.active_set.value()[i] = i
        self.active_size = self.l

        # initialize gradient
        self.G = alloc[Float64](self.l)
        self.G_bar = alloc[Float64](self.l)
        memcpy(dest=self.G, src=self.p, count=self.l)
        memset_zero(self.G_bar.value(), self.l)

        for i in range(self.l):
            if not self.is_lower_bound(i):
                var Q_i = Q.get_Q(i,self.l)
                var alpha_i = self.alpha.value()[i]
                for j in range(self.l):
                    self.G.value()[j] += alpha_i*Q_i.value()[j].cast[DType.float64]()
                if self.is_upper_bound(i):
                    for j in range(self.l):
                        self.G_bar.value()[j] += self.get_C(i) * Q_i.value()[j].cast[DType.float64]()

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

            var old_alpha_i = self.alpha.value()[i]
            var old_alpha_j = self.alpha.value()[j]

            if self.y.value()[i]!=self.y.value()[j]:
                var quad_coef = self.QD.value()[i]+self.QD.value()[j]+2*Q_i.value()[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (-self.G.value()[i]-self.G.value()[j])/quad_coef
                var diff = self.alpha.value()[i] - self.alpha.value()[j]
                self.alpha.value()[i] += delta
                self.alpha.value()[j] += delta

                if(diff > 0):
                    if self.alpha.value()[j] < 0:
                        self.alpha.value()[j] = 0
                        self.alpha.value()[i] = diff
                else:
                    if self.alpha.value()[i] < 0:
                        self.alpha.value()[i] = 0
                        self.alpha.value()[j] = -diff
                if diff > C_i - C_j:
                    if self.alpha.value()[i] > C_i:
                        self.alpha.value()[i] = C_i
                        self.alpha.value()[j] = C_i - diff
                else:
                    if self.alpha.value()[j] > C_j:
                        self.alpha.value()[j] = C_j
                        self.alpha.value()[i] = C_j + diff
            else:
                var quad_coef = self.QD.value()[i]+self.QD.value()[j]-2*Q_i.value()[j].cast[DType.float64]()
                if quad_coef <= 0:
                    quad_coef = TAU
                var delta = (self.G.value()[i]-self.G.value()[j])/quad_coef
                var sum = self.alpha.value()[i] + self.alpha.value()[j]
                self.alpha.value()[i] -= delta
                self.alpha.value()[j] += delta

                if sum > C_i:
                    if self.alpha.value()[i] > C_i:
                        self.alpha.value()[i] = C_i
                        self.alpha.value()[j] = sum - C_i
                else:
                    if self.alpha.value()[j] < 0:
                        self.alpha.value()[j] = 0
                        self.alpha.value()[i] = sum
                if sum > C_j:
                    if self.alpha.value()[j] > C_j:
                        self.alpha.value()[j] = C_j
                        self.alpha.value()[i] = sum - C_j
                else:
                    if self.alpha.value()[i] < 0:
                        self.alpha.value()[i] = 0
                        self.alpha.value()[j] = sum

            # update G

            var delta_alpha_i = self.alpha.value()[i] - old_alpha_i
            var delta_alpha_j = self.alpha.value()[j] - old_alpha_j

            for k in range(self.active_size):
                self.G.value()[k] += Q_i.value()[k].cast[DType.float64]()*delta_alpha_i + Q_j.value()[k].cast[DType.float64]()*delta_alpha_j

            # update alpha_status and G_bar

            var ui = self.is_upper_bound(i)
            var uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = Q.get_Q(i,self.l)
                if ui:
                    for k in range(self.l):
                        self.G_bar.value()[k] -= C_i * Q_i.value()[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar.value()[k] += C_i * Q_i.value()[k].cast[DType.float64]()

            if uj != self.is_upper_bound(j):
                Q_j = Q.get_Q(j,self.l)
                if uj:
                    for k in range(self.l):
                        self.G_bar.value()[k] -= C_j * Q_j.value()[k].cast[DType.float64]()
                else:
                    for k in range(self.l):
                        self.G_bar.value()[k] += C_j * Q_j.value()[k].cast[DType.float64]()

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
            v += self.alpha.value()[i] * (self.G.value()[i] + self.p.value()[i])

        self.si.obj = v/2

        # put back the solution

        for i in range(self.l):
            alpha_.value()[self.active_set.value()[i]] = self.alpha.value()[i]

        # juggle everything back

        #for i in range(self.l):
        #   while self.active_set[i] != i:
        #       self.swap_index(i,self.active_set[i])
        #       # or Q.swap_index(i,self.active_set[i])


        self.si.upper_bound_p = Cp
        self.si.upper_bound_n = Cn

        self.p.value().free()
        self.y.value().free()
        self.alpha.value().free()
        self.alpha_status.value().free()
        self.active_set.value().free()
        self.G.value().free()
        self.G_bar.value().free()

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
            if self.y.value()[t]== 1:
                if not self.is_upper_bound(t):
                    if -self.G.value()[t] >= Gmaxp:
                        Gmaxp = -self.G.value()[t]
                        Gmaxp_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G.value()[t] >= Gmaxn:
                        Gmaxn = self.G.value()[t]
                        Gmaxn_idx = t

        var i_p = Gmaxp_idx
        var i_n = Gmaxn_idx
        var Q_ip = OptionalUnsafePointer[Float32, MutExternalOrigin]()
        var Q_in = OptionalUnsafePointer[Float32, MutExternalOrigin]()
        if i_p != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_ip = Q.get_Q(i_p,self.active_size)
        if i_n != -1: # NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_in = Q.get_Q(i_n,self.active_size)

        for j in range(self.active_size):
            if self.y.value()[j]==1:
                if not self.is_lower_bound(j):
                    var grad_diff=Gmaxp+self.G.value()[j]
                    if self.G.value()[j] >= Gmaxp2:
                        Gmaxp2 = self.G.value()[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD.value()[i_p]+self.QD.value()[j]-2.0*Q_ip.value()[j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    var grad_diff= Gmaxn-self.G.value()[j]
                    if -self.G.value()[j] >= Gmaxn2:
                        Gmaxn2 = -self.G.value()[j]
                    if grad_diff > 0:
                        var obj_diff: Float64
                        var quad_coef = self.QD.value()[i_n]+self.QD.value()[j]+2.0*Q_in.value()[j].cast[DType.float64]()
                        if quad_coef > 0:
                            obj_diff = -(grad_diff*grad_diff)/quad_coef
                        else:
                            obj_diff = -(grad_diff*grad_diff)/TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx=j
                            obj_diff_min = obj_diff

        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < self.eps or Gmin_idx == -1:
            return 1

        if self.y.value()[Gmin_idx] == 1:
            out_i = Gmaxp_idx
        else:
            out_i = Gmaxn_idx
        out_j = Gmin_idx
        return 0

    def be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64, Gmax3: Float64, Gmax4: Float64) -> Bool:
        if self.is_upper_bound(i):
            if self.y.value()[i]==1:
                return -self.G.value()[i] > Gmax1
            else:
                return -self.G.value()[i] > Gmax4
        elif self.is_lower_bound(i):
            if self.y.value()[i]==1:
                return self.G.value()[i] > Gmax2
            else:
                return self.G.value()[i] > Gmax3
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
                if self.y.value()[i]==1:
                    if -self.G.value()[i] > Gmax1:
                        Gmax1 = -self.G.value()[i]
                else:
                    if -self.G.value()[i] > Gmax4:
                        Gmax4 = -self.G.value()[i]
            if not self.is_lower_bound(i):
                if self.y.value()[i]==1:
                    if self.G.value()[i] > Gmax2:
                        Gmax2 = self.G.value()[i]
                else:
                    if self.G.value()[i] > Gmax3:
                        Gmax3 = self.G.value()[i]

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
            if self.y.value()[i]==1:
                if self.is_upper_bound(i):
                    lb1 = max(lb1,self.G.value()[i])
                elif self.is_lower_bound(i):
                    ub1 = min(ub1,self.G.value()[i])
                else:
                    nr_free1 += 1
                    sum_free1 += self.G.value()[i]
            else:
                if self.is_upper_bound(i):
                    lb2 = max(lb2,self.G.value()[i])
                elif self.is_lower_bound(i):
                    ub2 = min(ub2,self.G.value()[i])
                else:
                    nr_free2 += 1
                    sum_free2 += self.G.value()[i]

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
struct SVC_Q[k_t: Int](QMatrix):
    var y: OptionalUnsafePointer[Int8, MutExternalOrigin]
    var cache: Cache
    var QD: OptionalUnsafePointer[Float64, MutExternalOrigin]

    var _self: kernel_params

    comptime kernel_function = (
        kernel_linear if Self.k_t == LINEAR else
        kernel_poly if Self.k_t == POLY else
        kernel_rbf if Self.k_t == RBF else
        kernel_sigmoid if Self.k_t == SIGMOID else
        kernel_precomputed
    )

    @always_inline
    def __init__(out self, prob: svm_problem, param: svm_parameter, y_: OptionalUnsafePointer[Int8, MutExternalOrigin]):
        # Kernel
        var x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](prob.l)
        memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: OptionalUnsafePointer[Float64, MutExternalOrigin]
        if param.kernel_type == RBF:
            x_square = alloc[Float64](prob.l)
            for i in range(prob.l):
                x_square.value()[i] = dot(x[i], x[i])
        else:
            x_square = OptionalUnsafePointer[Float64, MutExternalOrigin]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)
        ##
        self.y = alloc[Int8](prob.l)
        memcpy(dest=self.y, src=y_, count=prob.l)

        self.cache = Cache(prob.l, UInt(Int(param.cache_size*(1<<20))))

        self.QD = alloc[Float64](prob.l)
        for i in range(prob.l):
            self.QD.value()[i] = self.kernel_function(self._self, i,i)

    def get_Q(mut self, i: Int, _len: Int) -> OptionalUnsafePointer[Float32, MutExternalOrigin]:
        var data = OptionalUnsafePointer[Float32, MutExternalOrigin]()
        var start = self.cache.get_data(i, UnsafePointer(to=data),_len)
        if start < _len:
            @parameter
            def p(j: Int):
                data.value()[j+start] = ((self.y.value()[i]*self.y.value()[j+start]).cast[DType.float64]()*self.kernel_function(self._self, i,j+start)).cast[DType.float32]()
            parallelize[p](_len - start)
        return data

    def get_QD(self) -> OptionalUnsafePointer[Float64, MutExternalOrigin]:
        return self.QD

    def swap_index(mut self, i: Int, j: Int):
        self.cache.swap_index(i,j)

        swap(self._self.x.value()[i],self._self.x.value()[j])
        if self._self.x_square:
            swap(self._self.x_square.value()[i],self._self.x_square.value()[j])

        swap(self.y.value()[i],self.y.value()[j])
        swap(self.QD.value()[i],self.QD.value()[j])

    def __del__(deinit self):
        if self._self.x:
            self._self.x.value().free()
        if self._self.x_square:
            self._self.x_square.value().free()

        if self.y:
            self.y.value().free()
        if self.QD:
            self.QD.value().free()

struct ONE_CLASS_Q[k_t: Int](QMatrix):
    var cache: Cache
    var QD: OptionalUnsafePointer[Float64, MutExternalOrigin]

    var _self: kernel_params

    comptime kernel_function = (
        kernel_linear if Self.k_t == LINEAR else
        kernel_poly if Self.k_t == POLY else
        kernel_rbf if Self.k_t == RBF else
        kernel_sigmoid if Self.k_t == SIGMOID else
        kernel_precomputed
    )

    @always_inline
    def __init__(out self, prob: svm_problem, param: svm_parameter):
        # Kernel
        var x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](prob.l)
        memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: OptionalUnsafePointer[Float64, MutExternalOrigin]
        if param.kernel_type == RBF:
            x_square = alloc[Float64](prob.l)
            for i in range(prob.l):
                x_square.value()[i] = dot(x[i], x[i])
        else:
            x_square = OptionalUnsafePointer[Float64, MutExternalOrigin]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)
        ##
        self.cache = Cache(prob.l, UInt(Int(param.cache_size*(1<<20))))

        self.QD = alloc[Float64](prob.l)
        for i in range(prob.l):
            self.QD.value()[i] = self.kernel_function(self._self, i,i)

    def get_Q(mut self, i: Int, _len: Int) -> OptionalUnsafePointer[Float32, MutExternalOrigin]:
        var data = OptionalUnsafePointer[Float32, MutExternalOrigin]()
        var start = self.cache.get_data(i, UnsafePointer(to=data),_len)
        if start < _len:
            for j in range(start, _len):
                data.value()[j] = self.kernel_function(self._self, i,j).cast[DType.float32]()
        return data

    def get_QD(self) -> OptionalUnsafePointer[Float64, MutExternalOrigin]:
        return self.QD

    def swap_index(mut self, i: Int, j: Int):
        self.cache.swap_index(i,j)

        swap(self._self.x.value()[i],self._self.x.value()[j])
        if self._self.x_square:
            swap(self._self.x_square.value()[i],self._self.x_square.value()[j])

        swap(self.QD.value()[i],self.QD.value()[j])

    def __del__(deinit self):
        if self._self.x:
            self._self.x.value().free()
        if self._self.x_square:
            self._self.x_square.value().free()

        if self.QD:
            self.QD.value().free()

struct SVR_Q[k_t: Int](QMatrix):
    var l: Int
    var cache: Cache
    var sign: OptionalUnsafePointer[Int8, MutExternalOrigin]
    var index: OptionalUnsafePointer[Int, MutExternalOrigin]
    var next_buffer: Int
    var buffer: InlineArray[OptionalUnsafePointer[Float32, MutExternalOrigin], 2]
    var QD: OptionalUnsafePointer[Float64, MutExternalOrigin]

    var _self: kernel_params

    comptime kernel_function = (
        kernel_linear if Self.k_t == LINEAR else
        kernel_poly if Self.k_t == POLY else
        kernel_rbf if Self.k_t == RBF else
        kernel_sigmoid if Self.k_t == SIGMOID else
        kernel_precomputed
    )

    @always_inline
    def __init__(out self, prob: svm_problem, param: svm_parameter):
        # Kernel
        var x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](prob.l)
        memcpy(dest=x, src=prob.x, count=prob.l)

        var x_square: OptionalUnsafePointer[Float64, MutExternalOrigin]
        if param.kernel_type == RBF:
            x_square = alloc[Float64](prob.l)
            for i in range(prob.l):
                x_square.value()[i] = dot(x[i], x[i])
        else:
            x_square = OptionalUnsafePointer[Float64, MutExternalOrigin]()

        self._self = kernel_params(x, x_square, param.kernel_type, param.degree, param.gamma, param.coef0)
        ##
        self.l = prob.l
        self.cache = Cache(self.l, UInt(Int(param.cache_size*(1<<20))))
        self.QD = alloc[Float64](2*self.l)
        self.sign = alloc[Int8](2*self.l)
        self.index = alloc[Int](2*self.l)
        for k in range(self.l):
            self.sign.value()[k] = 1
            self.sign.value()[k+self.l] = -1
            self.index.value()[k] = k
            self.index.value()[k+self.l] = k
            self.QD.value()[k] = self.kernel_function(self._self, k,k)
            self.QD.value()[k+self.l] = self.QD.value()[k]
        self.buffer: InlineArray[OptionalUnsafePointer[Float32, MutExternalOrigin], 2] = [alloc[Float32](2*self.l), alloc[Float32](2*self.l)]
        self.next_buffer = 0

    def swap_index(self, i: Int, j: Int):
        swap(self.sign.value()[i],self.sign.value()[j])
        swap(self.index.value()[i],self.index.value()[j])
        swap(self.QD.value()[i],self.QD.value()[j])

    def get_Q(mut self, i: Int, _len: Int) -> OptionalUnsafePointer[Float32, MutExternalOrigin]:
        var data = OptionalUnsafePointer[Float32, MutExternalOrigin]()
        var real_i = self.index.value()[i]
        if self.cache.get_data(real_i, UnsafePointer(to=data),self.l) < self.l:
            @parameter
            def p(j: Int):
                data.value()[j] = self.kernel_function(self._self, real_i,j).cast[DType.float32]()
            parallelize[p](self.l)
        # reorder and copy
        var buf = self.buffer[self.next_buffer]
        self.next_buffer = 1 - self.next_buffer
        var si = self.sign.value()[i]
        for j in range(_len):
            buf.value()[j] = si.cast[DType.float32]() * self.sign.value()[j].cast[DType.float32]() * data.value()[self.index.value()[j]]
        return buf

    def get_QD(self) -> OptionalUnsafePointer[Float64, MutExternalOrigin]:
        return self.QD

    def __del__(deinit self):
        if self._self.x:
            self._self.x.value().free()
        if self._self.x_square:
            self._self.x_square.value().free()

        if self.QD:
            self.QD.value().free()
        if self.sign:
            self.sign.value().free()
        if self.index:
            self.index.value().free()
        if self.buffer[0]:
            self.buffer[0].value().free()
        if self.buffer[1]:
            self.buffer[1].value().free()

#
# construct and solve various formulations
#
def solve_c_svc[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    alpha: OptionalUnsafePointer[Float64, MutExternalOrigin], mut si: SolutionInfo, Cp: Float64, Cn: Float64):
    var l = prob.l
    var minus_ones = alloc[Float64](l)
    var y = alloc[Int8](l)

    memset_zero(alpha.value(), l)
    for i in range(l):
        minus_ones[i] = -1
        if prob.y.value()[i] > 0:
            y[i] = 1
        else:
            y[i] = -1

    var s = Solver()
    var q = SVC_Q[k_t](prob,param,y)
    s.Solve(l, q, minus_ones, y,
        alpha, Cp, Cn, param.eps, si, param.shrinking)

    var sum_alpha=0.0
    for i in range(l):
        sum_alpha += alpha.value()[i]

    for i in range(l):
        alpha.value()[i] *= y[i].cast[DType.float64]()

    minus_ones.free()
    y.free()

def solve_nu_svc[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    alpha: OptionalUnsafePointer[Float64, MutExternalOrigin], mut si: SolutionInfo):
    var l = prob.l
    var nu = param.nu

    var y = alloc[Int8](l)

    for i in range(l):
        if prob.y.value()[i]>0:
            y[i] = 1
        else:
            y[i] = -1

    var sum_pos = nu*Float64(l)/2
    var sum_neg = nu*Float64(l)/2

    for i in range(l):
        if y[i] == 1:
            alpha.value()[i] = min(1.0,sum_pos)
            sum_pos -= alpha.value()[i]
        else:
            alpha.value()[i] = min(1.0,sum_neg)
            sum_neg -= alpha.value()[i]

    var zeros = alloc[Float64](l)
    memset_zero(zeros, l)

    var s = Solver_NU()
    var q = SVC_Q[k_t](prob,param,y)
    s.Solve(l, q, zeros, y,
        alpha, 1.0, 1.0, param.eps, si, param.shrinking)
    var r = si.r

    for i in range(l):
        alpha.value()[i] *= y[i].cast[DType.float64]()/r

    si.rho /= r
    si.obj /= (r*r)
    si.upper_bound_p = 1/r
    si.upper_bound_n = 1/r 

    y.free()
    zeros.free()

def solve_one_class[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    alpha: OptionalUnsafePointer[Float64, MutExternalOrigin], mut si: SolutionInfo):
    var l = prob.l
    var zeros = alloc[Float64](l)
    var ones = alloc[Int8](l)

    var n = Int(param.nu*Float64(prob.l))	# # of alpha's at upper bound

    for i in range(n):
        alpha.value()[i] = 1
    if n<prob.l:
        alpha.value()[n] = param.nu * Float64(prob.l) - Float64(n)
    for i in range(n+1, l):
        alpha.value()[i] = 0

    memset_zero(zeros, l)
    for i in range(l):
        ones[i] = 1

    var s = Solver()
    var q = ONE_CLASS_Q[k_t](prob,param)
    s.Solve(l, q, zeros, ones,
        alpha, 1.0, 1.0, param.eps, si, param.shrinking)

    zeros.free()
    ones.free()

def solve_epsilon_svr[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    alpha: OptionalUnsafePointer[Float64, MutExternalOrigin], mut si: SolutionInfo):
    var l = prob.l
    var alpha2 = alloc[Float64](2*l)
    var linear_term = alloc[Float64](2*l)
    var y = alloc[Int8](2*l)

    for i in range(l):
        alpha2[i] = 0
        linear_term[i] = param.p - prob.y.value()[i]
        y[i] = 1

        alpha2[i+l] = 0
        linear_term[i+l] = param.p + prob.y.value()[i]
        y[i+l] = -1

    var s = Solver()
    var q = SVR_Q[k_t](prob,param)
    s.Solve(2*l, q, linear_term, y,
        alpha2, param.C, param.C, param.eps, si, param.shrinking)

    var sum_alpha = 0.0
    for i in range(l):
        alpha.value()[i] = alpha2[i] - alpha2[i+l]
        sum_alpha += abs(alpha.value()[i])

    alpha2.free()
    linear_term.free()
    y.free()

def solve_nu_svr[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    alpha: OptionalUnsafePointer[Float64, MutExternalOrigin], mut si: SolutionInfo):
    var l = prob.l
    var C = param.C
    var alpha2 = alloc[Float64](2*l)
    var linear_term = alloc[Float64](2*l)
    var y = alloc[Int8](2*l)

    var sum = C * param.nu * Float64(l) / 2
    for i in range(l):
        alpha2[i] = alpha2[i+l] = min(sum,C)
        sum -= alpha2[i]

        linear_term[i] = - prob.y.value()[i]
        y[i] = 1

        linear_term[i+l] = prob.y.value()[i]
        y[i+l] = -1

    var s = Solver_NU()
    var q = SVR_Q[k_t](prob,param)
    s.Solve(2*l, q, linear_term, y,
        alpha2, C, C, param.eps, si, param.shrinking)

    for i in range(l):
        alpha.value()[i] = alpha2[i] - alpha2[i+l]

    alpha2.free()
    linear_term.free()
    y.free()

#
# decision_function
#
@fieldwise_init
struct decision_function(RegisterPassable, Copyable):
    var alpha: OptionalUnsafePointer[Float64, MutExternalOrigin]
    var rho: Float64

def svm_train_one[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    Cp: Float64, Cn: Float64) -> decision_function:
    var alpha = alloc[Float64](prob.l)
    var si = SolutionInfo()
    if param.svm_type == svm_parameter.C_SVC:
        solve_c_svc[k_t](prob,param,alpha,si,Cp,Cn)
    elif param.svm_type == svm_parameter.NU_SVC:
        solve_nu_svc[k_t](prob,param,alpha,si)
    elif param.svm_type == svm_parameter.ONE_CLASS:
        solve_one_class[k_t](prob,param,alpha,si)
    elif param.svm_type == svm_parameter.EPSILON_SVR:
        solve_epsilon_svr[k_t](prob,param,alpha,si)
    elif param.svm_type == svm_parameter.NU_SVR:
        solve_nu_svr[k_t](prob,param,alpha,si)

    # output SVs

    var nSV = 0
    var nBSV = 0
    for i in range(prob.l):
        if abs(alpha[i]) > 0:
            nSV += 1
            if prob.y.value()[i] > 0:
                if abs(alpha[i]) >= si.upper_bound_p:
                    nBSV += 1
            else:
                if abs(alpha[i]) >= si.upper_bound_n:
                    nBSV += 1

    return decision_function(alpha=alpha, rho=si.rho)

# Platt's binary SVM Probablistic Output: an improvement from Lin et al.
def sigmoid_train(
    l: Int, dec_values: OptionalUnsafePointer[Float64, MutExternalOrigin], labels: OptionalUnsafePointer[Float64, MutExternalOrigin],
    mut A: Float64, mut B: Float64):
    var prior1 = 0.0
    var prior0 = 0.0

    for i in range(l):
        if labels.value()[i] > 0:
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
        if (labels.value()[i]>0):
            t[i]=hiTarget
        else:
            t[i]=loTarget
        fApB = dec_values.value()[i]*A+B
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
            fApB = dec_values.value()[i]*A+B
            if (fApB >= 0):
                p=math.exp(-fApB)/(1.0+math.exp(-fApB))
                q=1.0/(1.0+math.exp(-fApB))
            else:
                p=1.0/(1.0+math.exp(fApB))
                q=math.exp(fApB)/(1.0+math.exp(fApB))

            d2=p*q
            h11+=dec_values.value()[i]*dec_values.value()[i]*d2
            h22+=d2
            h21+=dec_values.value()[i]*d2
            d1=t[i]-p
            g1+=dec_values.value()[i]*d1
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
                fApB = dec_values.value()[i]*newA+newB
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

def sigmoid_predict(decision_value: Float64, A: Float64, B: Float64) -> Float64:
    var fApB = decision_value*A+B
    # 1-p used later; avoid catastrophic cancellation
    if fApB >= 0:
        return math.exp(-fApB)/(1.0+math.exp(-fApB))
    else:
        return 1.0/(1+math.exp(fApB))

# Method 2 from the multiclass_prob paper by Wu, Lin, and Weng to predict probabilities
def multiclass_probability(k: Int, r: OptionalUnsafePointer[OptionalUnsafePointer[Float64, MutExternalOrigin], MutExternalOrigin], p: OptionalUnsafePointer[Float64, MutExternalOrigin]):
    var max_iter=max(100,k)
    var Q=alloc[OptionalUnsafePointer[Float64, MutExternalOrigin]](k)
    var Qp=alloc[Float64](k)
    var pQp: Float64
    var eps=0.005/Float64(k)

    for t in range(k):
        p.value()[t]=1.0/Float64(k)  # Valid if k = 1
        Q[t]=alloc[Float64](k)
        Q[t].value()[t]=0
        for j in range(t):
            Q[t].value()[t]+=r.value()[j].value()[t]*r.value()[j].value()[t]
            Q[t].value()[j]=Q[j].value()[t]
        for j in range(t+1,k):
            Q[t].value()[t]+=r.value()[j].value()[t]*r.value()[j].value()[t]
            Q[t].value()[j]=-r.value()[j].value()[t]*r.value()[t].value()[j]
    var iter = 0
    while iter<max_iter:
        # stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0.0
        for t in range(k):
            Qp[t]=0
            for j in range(k):
                Qp[t]+=Q[t].value()[j]*p.value()[j]
            pQp+=p.value()[t]*Qp[t]

        var max_error=0.0
        for t in range(k):
            var error=abs(Qp[t]-pQp)
            if error>max_error:
                max_error=error

        if max_error<eps:
            break

        for t in range(k):
            var diff=(-Qp[t]+pQp)/Q[t].value()[t]
            p.value()[t]+=diff
            pQp=(pQp+diff*(diff*Q[t].value()[t]+2*Qp[t]))/(1+diff)/(1+diff)
            for j in range(k):
                Qp[j]=(Qp[j]+diff*Q[t].value()[j])/(1+diff)
                p.value()[j]/=(1+diff)

        iter += 1

    if iter>=max_iter:
        print("Exceeds max_iter in multiclass_prob\n")
    for t in range(k):
        Q[t].value().free()
    Q.free()
    Qp.free()

# Using cross-validation decision values to get parameters for SVC probability estimates
def svm_binary_svc_probability[k_t: Int](
    prob: svm_problem, param: svm_parameter,
    Cp: Float64, Cn: Float64, mut probA: Float64, mut probB: Float64):
    var nr_fold = 5
    var perm: OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]
    var dec_values = alloc[Float64](prob.l)

    # random shuffle
    try:
        perm = fill_indices(prob.l)
    except:
        perm = alloc[Scalar[DType.int]](prob.l)
        for i in range(Scalar[DType.int](prob.l)):
            perm.value()[i]=i

    for i in range(prob.l - 1, 0, -1):
        var j = Int(random.random_ui64(0, UInt64(i)))
        swap(perm.value()[i],perm.value()[j])

    for i in range(nr_fold):
        var begin = i*prob.l//nr_fold
        var end = (i+1)*prob.l//nr_fold
        var k = 0
        var subprob = svm_problem()

        subprob.l = prob.l-(end-begin)
        subprob.x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](subprob.l)
        subprob.y = alloc[Float64](subprob.l)

        for j in range(begin):
            subprob.x.value()[k] = prob.x.value()[perm.value()[j]]
            subprob.y.value()[k] = prob.y.value()[perm.value()[j]]
            k += 1

        for j in range(end, prob.l):
            subprob.x.value()[k] = prob.x.value()[perm.value()[j]]
            subprob.y.value()[k] = prob.y.value()[perm.value()[j]]
            k += 1

        var p_count, n_count = 0, 0
        for j in range(k):
            if subprob.y.value()[j]>0:
                p_count += 1
            else:
                n_count += 1

        if p_count==0 and n_count==0:
            for j in range(begin, end):
                dec_values[perm.value()[j]] = 0
        elif p_count > 0 and n_count == 0:
            for j in range(begin, end):
                dec_values[perm.value()[j]] = 1
        elif p_count == 0 and n_count > 0:
            for j in range(begin, end):
                dec_values[perm.value()[j]] = -1
        else:
            var subparam = param.copy()
            subparam.probability=0
            subparam.C=1.0
            subparam.nr_weight=2
            subparam.weight_label = alloc[Int](2)
            subparam.weight = alloc[Float64](2)
            subparam.weight_label.value()[0]=+1
            subparam.weight_label.value()[1]=-1
            subparam.weight.value()[0]=Cp
            subparam.weight.value()[1]=Cn
            var submodel = svm_train[k_t](subprob,subparam)
            for j in range(begin, end):
                _ = svm_predict_values(submodel.value()[],prob.x.value()[perm.value()[j]],dec_values + perm.value()[j])
                # ensure +1 -1 order; reason not using CV subroutine
                dec_values[perm.value()[j]] *= Float64(submodel.value()[].label.value()[0])

            svm_free_and_destroy_model(submodel)
            svm_destroy_param(subparam)

        subprob.x.value().free()
        subprob.y.value().free()

    sigmoid_train(prob.l,dec_values,prob.y,probA,probB)
    dec_values.free()
    perm.value().free()

# Binning method from the oneclass_prob paper by Que and Lin to predict the probability as a normal instance (i.e., not an outlier)
def predict_one_class_probability(model: svm_model, dec_value: Float64) -> Float64:
    var prob_estimate = 0.0
    var nr_marks = 10

    if dec_value < model.prob_density_marks.value()[0]:
        prob_estimate = 0.001
    elif dec_value > model.prob_density_marks.value()[nr_marks-1]:
        prob_estimate = 0.999
    else:
        for i in range(1,nr_marks):
            if dec_value < model.prob_density_marks.value()[i]:
                prob_estimate = Float64(i)/Float64(nr_marks)
                break

    return prob_estimate

# Get parameters for one-class SVM probability estimates
def svm_one_class_probability(prob: svm_problem, model: svm_model, prob_density_marks: OptionalUnsafePointer[Float64, MutExternalOrigin]) -> Int:
    var dec_values = alloc[Float64](prob.l)
    var pred_results = alloc[Float64](prob.l)
    var ret = 0
    var nr_marks = 10

    for i in range(prob.l):
        pred_results[i] = svm_predict_values(model,prob.x.value()[i], dec_values + i)
    @parameter
    def cmp_fn(a: Float64, b: Float64) -> Bool:
        return a < b

    sort[cmp_fn](
        Span[
            Float64,
            MutExternalOrigin,
        ](ptr=dec_values, length=prob.l)
    )

    var neg_counter=0
    for i in range(prob.l):
        if dec_values[i]>=0:
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
            tmp_marks[i] = dec_values[i*neg_counter//mid]
        tmp_marks[mid] = 0
        for i in range(mid+1, nr_marks+1):
            tmp_marks[i] = dec_values[neg_counter-1+(i-mid)*pos_counter//mid]

        for i in range(nr_marks):
            prob_density_marks.value()[i] = (tmp_marks[i]+tmp_marks[i+1])/2
        tmp_marks.free()

    dec_values.free()
    pred_results.free()
    return ret

# Return parameter of a Laplace distribution
def svm_svr_probability[k_t: Int](prob: svm_problem, param: svm_parameter) -> Float64:
    var nr_fold = 5
    var ymv = alloc[Float64](prob.l)
    var mae = 0.0

    var newparam = param.copy()
    newparam.probability = 0
    svm_cross_validation[k_t](prob, newparam, nr_fold, ymv)
    for i in range(prob.l):
        ymv[i]=prob.y.value()[i]-ymv[i]
        mae += abs(ymv[i])
    mae /= Float64(prob.l)
    var std=math.sqrt(2*mae*mae)
    var count=0
    mae=0.0
    for i in range(prob.l):
        if abs(ymv[i]) > 5*std:
            count=count+1
        else:
            mae+=abs(ymv[i])
    mae /= Float64(prob.l-count)

    ymv.free()
    return mae

# label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
# perm, length l, must be allocated before calling this subroutine
def svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: OptionalUnsafePointer[Int, MutExternalOrigin], mut start_ret: OptionalUnsafePointer[Int, MutExternalOrigin], mut count_ret: OptionalUnsafePointer[Int, MutExternalOrigin], perm: OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]):
    var l = prob.l
    var max_nr_class = 16
    var nr_class = 0
    var label = alloc[Int](max_nr_class)
    var count = alloc[Int](max_nr_class)
    var data_label = alloc[Int](l)

    for i in range(l):
        var this_label = Int(prob.y.value()[i])
        var j = 0
        while j<nr_class:
            if this_label == label[j]:
                count[j] += 1
                break
            j += 1

        data_label[i] = j
        if j == nr_class:
            if nr_class == max_nr_class:
                var new = alloc[Int](max_nr_class*2)
                memcpy(dest=new, src=label, count=max_nr_class)
                label.free()
                label = new
                new = alloc[Int](max_nr_class*2)
                memcpy(dest=new, src=count, count=max_nr_class)
                count.free()
                count = new
            label[nr_class] = this_label
            count[nr_class] = 1
            nr_class += 1

    #
    # Labels are ordered by their first occurrence in the training set.
    # However, for two-class sets with -1/+1 labels and -1 appears first,
    # we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
    #
    if nr_class == 2 and label[0] == -1 and label[1] == 1:
        swap(label[0],label[1])
        swap(count[0],count[1])
        for i in range(l):
            if data_label[i] == 0:
                data_label[i] = 1
            else:
                data_label[i] = 0

    var start = alloc[Int](nr_class)
    start[0] = 0
    for i in range(1,nr_class):
        start[i] = start[i-1]+count[i-1]
    for i in range(Scalar[DType.int](l)):
        perm.value()[start[data_label[i]]] = i
        start[data_label[i]] += 1
    start[0] = 0
    for i in range(1,nr_class):
        start[i] = start[i-1]+count[i-1]

    nr_class_ret = nr_class
    label_ret = label
    start_ret = start
    count_ret = count
    data_label.free()

#
# Interface functions
#
def svm_train[k_t: Int](prob: svm_problem, param: svm_parameter) -> OptionalUnsafePointer[svm_model, MutExternalOrigin]:
    var model = alloc[svm_model](1)
    model[].param = param.copy()
    model[].free_sv = 0

    if param.svm_type == svm_parameter.ONE_CLASS or param.svm_type == svm_parameter.EPSILON_SVR or param.svm_type == svm_parameter.NU_SVR:
        # regression or one-class-svm
        model[].nr_class = 2
        model[].label = OptionalUnsafePointer[Int, MutExternalOrigin]()
        model[].nSV = OptionalUnsafePointer[Int, MutExternalOrigin]()
        model[].probA = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        model[].probB = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        model[].prob_density_marks = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        model[].sv_coef = alloc[OptionalUnsafePointer[Float64, MutExternalOrigin]](1)

        var f = svm_train_one[k_t](prob,param,0,0)
        model[].rho = alloc[Float64](1)
        model[].rho.value()[0] = f.rho

        var nSV = 0
        for i in range(prob.l):
            if abs(f.alpha.value()[i]) > 0:
                nSV += 1
        model[].l = nSV
        model[].SV = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](nSV)
        model[].sv_coef.value()[0] = alloc[Float64](nSV)
        model[].sv_indices = alloc[Scalar[DType.int]](nSV)
        var j = 0
        for i in range(Scalar[DType.int](prob.l)):
            if abs(f.alpha.value()[i]) > 0:
                model[].SV.value()[j] = prob.x.value()[i]
                model[].sv_coef.value()[0].value()[j] = f.alpha.value()[i]
                model[].sv_indices.value()[j] = i+1
                j += 1

        if param.probability and (param.svm_type == svm_parameter.EPSILON_SVR or param.svm_type == svm_parameter.NU_SVR):
            model[].probA = alloc[Float64](1)
            model[].probA.value()[0] = svm_svr_probability[k_t](prob,param)
        elif param.probability and param.svm_type == svm_parameter.ONE_CLASS:
            var nr_marks = 10
            var prob_density_marks = alloc[Float64](nr_marks)

            if svm_one_class_probability(prob,model[],prob_density_marks) == 0:
                model[].prob_density_marks = prob_density_marks
            else:
                prob_density_marks.free()

        f.alpha.value().free()
    else:
        # classification
        var l = prob.l
        var nr_class = 0
        var label = OptionalUnsafePointer[Int, MutExternalOrigin]()
        var start = OptionalUnsafePointer[Int, MutExternalOrigin]()
        var count = OptionalUnsafePointer[Int, MutExternalOrigin]()
        var perm = alloc[Scalar[DType.int]](l)

        # group training data of the same class
        svm_group_classes(prob,nr_class,label,start,count,perm)

        var x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](l)
        for i in range(l):
            x[i] = prob.x.value()[perm[i]]

        # calculate weighted C
        var weighted_C = alloc[Float64](nr_class)
        for i in range(nr_class):
            weighted_C[i] = param.C
        for i in range(param.nr_weight):
            var j = 0
            while j<nr_class:
                if param.weight_label.value()[i] == label.value()[j]:
                    break
                j += 1
            if j == nr_class:
                print("WARNING: class label", param.weight_label.value()[i], "specified in weight is not found\n")
            else:
                weighted_C[j] *= param.weight.value()[i]

        # train k*(k-1)/2 models

        var nonzero = alloc[Bool](l)
        memset_zero(nonzero, l)
        var f = alloc[decision_function](nr_class*(nr_class-1)//2)

        var probA = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        var probB = OptionalUnsafePointer[Float64, MutExternalOrigin]()
        if param.probability:
            probA = alloc[Float64](nr_class*(nr_class-1)//2)
            probB = alloc[Float64](nr_class*(nr_class-1)//2)

        var p = 0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                var sub_prob = svm_problem()
                var si = start.value()[i]
                var sj = start.value()[j]
                var ci = count.value()[i]
                var cj = count.value()[j]
                sub_prob.l = ci+cj
                sub_prob.x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](sub_prob.l)
                sub_prob.y = alloc[Float64](sub_prob.l)

                for k in range(ci):
                    sub_prob.x.value()[k] = x[si+k]
                    sub_prob.y.value()[k] = 1

                for k in range(cj):
                    sub_prob.x.value()[ci+k] = x[sj+k]
                    sub_prob.y.value()[ci+k] = -1

                if param.probability:
                    svm_binary_svc_probability[k_t](sub_prob,param,weighted_C[i],weighted_C[j],probA.value()[p],probB.value()[p])

                f[p] = svm_train_one[k_t](sub_prob,param,weighted_C[i],weighted_C[j])
                for k in range(ci):
                    if not nonzero[si+k] and abs(f[p].alpha.value()[k]) > 0:
                        nonzero[si+k] = True
                for k in range(cj):
                    if not nonzero[sj+k] and abs(f[p].alpha.value()[ci+k]) > 0:
                        nonzero[sj+k] = True
                sub_prob.x.value().free()
                sub_prob.y.value().free()
                p += 1

        # build output

        model[].nr_class = nr_class

        model[].label = alloc[Int](nr_class)
        for i in range(nr_class):
            model[].label.value()[i] = label.value()[i]

        model[].rho = alloc[Float64](nr_class*(nr_class-1)//2)
        for i in range(nr_class*(nr_class-1)//2):
            model[].rho.value()[i] = f[i].rho

        if param.probability:
            model[].probA = alloc[Float64](nr_class*(nr_class-1)//2)
            model[].probB = alloc[Float64](nr_class*(nr_class-1)//2)
            for i in range(nr_class*(nr_class-1)//2):
                model[].probA.value()[i] = probA.value()[i]
                model[].probB.value()[i] = probB.value()[i]
        else:
            model[].probA=OptionalUnsafePointer[Float64, MutExternalOrigin]()
            model[].probB=OptionalUnsafePointer[Float64, MutExternalOrigin]()

        model[].prob_density_marks=OptionalUnsafePointer[Float64, MutExternalOrigin]()	# for one-class SVM probabilistic outputs only

        var total_sv = 0
        var nz_count = alloc[Int](nr_class)
        model[].nSV = alloc[Int](nr_class)
        for i in range(nr_class):
            var nSV = 0
            for j in range(count.value()[i]):
                if nonzero[start.value()[i]+j]:
                    nSV += 1
                    total_sv += 1

            model[].nSV.value()[i] = nSV
            nz_count[i] = nSV

        model[].l = total_sv
        model[].SV = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](total_sv)
        model[].sv_indices = alloc[Scalar[DType.int]](total_sv)
        p = 0
        for i in range(l):
            if nonzero[i]:
                model[].SV.value()[p] = x[i]
                model[].sv_indices.value()[p] = perm[i] + 1
                p += 1

        var nz_start = alloc[Int](nr_class)
        nz_start[0] = 0
        for i in range(1, nr_class):
            nz_start[i] = nz_start[i-1]+nz_count[i-1]

        model[].sv_coef = alloc[OptionalUnsafePointer[Float64, MutExternalOrigin]](nr_class-1)
        for i in range(nr_class-1):
            model[].sv_coef.value()[i] = alloc[Float64](total_sv)

        p = 0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                # classifier (i,j): coefficients with
                # i are in sv_coef[j-1][nz_start[i]...],
                # j are in sv_coef[i][nz_start[j]...]

                var si = start.value()[i]
                var sj = start.value()[j]
                var ci = count.value()[i]
                var cj = count.value()[j]

                var q = nz_start[i]
                for k in range(ci):
                    if nonzero[si+k]:
                        model[].sv_coef.value()[j-1].value()[q] = f[p].alpha.value()[k]
                        q += 1
                q = nz_start[j]
                for k in range(cj):
                    if nonzero[sj+k]:
                        model[].sv_coef.value()[i].value()[q] = f[p].alpha.value()[ci+k]
                        q += 1
                p += 1

        label.value().free()
        probA.value().free()
        probB.value().free()
        count.value().free()
        perm.free()
        start.value().free()
        x.free()
        weighted_C.free()
        nonzero.free()
        for i in range(nr_class*(nr_class-1)//2):
            f[i].alpha.value().free()
        f.free()
        nz_count.free()
        nz_start.free()

    return model

# Stratified cross validation
def svm_cross_validation[k_t: Int](prob: svm_problem, param: svm_parameter, var nr_fold: Int, target: OptionalUnsafePointer[Float64, MutExternalOrigin]):
    var fold_start = alloc[Int](nr_fold+1)
    var l = prob.l
    var perm = alloc[Scalar[DType.int]](l)
    var nr_class = 0
    if nr_fold > l:
        print("WARNING: # folds ("+ String(nr_fold) +") > # data ("+ String(l) +"). Will use # folds = # data instead (i.e., leave-one-out cross validation)\n")
        nr_fold = l

    # stratified cv may not give leave-one-out rate
    # Each class to l folds -> some folds may have zero elements
    if (param.svm_type == svm_parameter.C_SVC or param.svm_type == svm_parameter.NU_SVC) and nr_fold < l:
        var start = OptionalUnsafePointer[Int, MutExternalOrigin]()
        var label = OptionalUnsafePointer[Int, MutExternalOrigin]()
        var count = OptionalUnsafePointer[Int, MutExternalOrigin]()
        svm_group_classes(prob,nr_class,label,start,count,perm)

        # random shuffle and then data grouped by fold using the array perm
        var fold_count = alloc[Int](nr_fold)
        var index = alloc[Scalar[DType.int]](l)
        memcpy(dest=index, src=perm, count=l)
        for c in range(nr_class):
            for i in range(count.value()[c] - 1, 0, -1):
                var j = Int(random.random_ui64(0, UInt64(i)))
                swap(index[start.value()[c]+j],index[start.value()[c]+i])

        for i in range(nr_fold):
            fold_count[i] = 0
            for c in range(nr_class):
                fold_count[i]+=(i+1)*count.value()[c]//nr_fold-i*count.value()[c]//nr_fold

        fold_start[0]=0
        for i in range(1, nr_fold+1):
            fold_start[i] = fold_start[i-1]+fold_count[i-1]
        for c in range(nr_class):
            for i in range(nr_fold):
                var begin = start.value()[c]+i*count.value()[c]//nr_fold
                var end = start.value()[c]+(i+1)*count.value()[c]//nr_fold
                for j in range(begin, end):
                    perm[fold_start[i]] = index[j]
                    fold_start[i] += 1

        fold_start[0]=0
        for i in range(1, nr_fold+1):
            fold_start[i] = fold_start[i-1]+fold_count[i-1]
        start.value().free()
        label.value().free()
        count.value().free()
        index.free()
        fold_count.free()
    else:
        try:
            perm = fill_indices(l)
        except:
            perm = alloc[Scalar[DType.int]](l)
            for i in range(Scalar[DType.int](l)):
                perm[i]=i
        for i in range(l - 1, 0, -1):
            var j = Int(random.random_ui64(0, UInt64(i)))
            swap(perm[i],perm[j])

        for i in range(nr_fold+1):
            fold_start[i]=i*l//nr_fold

    for i in range(nr_fold):
        var begin = fold_start[i]
        var end = fold_start[i+1]
        var k = 0
        var subprob = svm_problem()

        subprob.l = l-(end-begin)
        subprob.x = alloc[OptionalUnsafePointer[svm_node, MutExternalOrigin]](subprob.l)
        subprob.y = alloc[Float64](subprob.l)

        for j in range(begin):
            subprob.x.value()[k] = prob.x.value()[perm[j]]
            subprob.y.value()[k] = prob.y.value()[perm[j]]
            k += 1

        for j in range(end,l):
            subprob.x.value()[k] = prob.x.value()[perm[j]]
            subprob.y.value()[k] = prob.y.value()[perm[j]]
            k += 1

        var submodel = svm_train[k_t](subprob,param)
        if param.probability and (param.svm_type == svm_parameter.C_SVC or param.svm_type == svm_parameter.NU_SVC):
            var prob_estimates = alloc[Float64](svm_get_nr_class(submodel.value()[]))
            for j in range(begin, end):
                target.value()[perm[j]] = svm_predict_probability(submodel.value()[],prob.x.value()[perm[j]],prob_estimates)
            prob_estimates.free()
        else:
            for j in range(begin, end):
                target.value()[perm[j]] = svm_predict(submodel.value()[],prob.x.value()[perm[j]])
        svm_free_and_destroy_model(submodel)
        subprob.x.value().free()
        subprob.y.value().free()

    fold_start.free()
    perm.free()

@always_inline
def svm_get_svm_type(model: svm_model) -> Int:
    return model.param.svm_type

@always_inline
def svm_get_nr_class(model: svm_model) -> Int:
    return model.nr_class

def svm_get_labels(model: svm_model, label: OptionalUnsafePointer[Int, MutExternalOrigin]):
    if model.label:
        for i in range(model.nr_class):
            label.value()[i] = model.label.value()[i]

def svm_get_sv_indices(model: svm_model, indices: OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin]):
    if model.sv_indices:
        memcpy(dest=indices, src=model.sv_indices, count=model.l)

@always_inline
def svm_get_nr_sv(model: svm_model) -> Int:
    return model.l

def svm_get_svr_probability(model: svm_model) -> Float64:
    if (model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR) and model.probA:
        return model.probA.value()[0]
    else:
        print("Model doesn't contain information for SVR probability inference\n")
        return 0.0

def svm_predict_values(model: svm_model, x: OptionalUnsafePointer[svm_node, MutExternalOrigin], dec_values: OptionalUnsafePointer[Float64, MutAnyOrigin]) -> Float64:
    if model.param.svm_type == svm_parameter.ONE_CLASS or model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR:
        var sv_coef = model.sv_coef.value()[0]
        var sum = 0.0

        var values = alloc[Float64](model.l)
        @parameter
        def p(i: Int):
            values[i] = sv_coef.value()[i] * k_function(x,model.SV.value()[i],model.param)
        parallelize[p](model.l)
        try:
            sum = reduction.sum(Span[Float64, MutExternalOrigin](ptr=values, length=model.l))
        except e:
            print('Error:', e)
        values.free()
        
        sum -= model.rho.value()[0]
        dec_values.value()[] = sum

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
            kvalue[i] = k_function(x,model.SV.value()[i],model.param)
        parallelize[pv](l)

        var start = alloc[Int](nr_class)
        start[0] = 0
        for i in range(1, nr_class):
            start[i] = start[i-1]+model.nSV.value()[i-1]

        var vote = alloc[Int](nr_class)
        for i in range(nr_class):
            vote[i] = 0

        var p=0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                var sum = 0.0
                var si = start[i]
                var sj = start[j]
                var ci = model.nSV.value()[i]
                var cj = model.nSV.value()[j]

                var coef1 = model.sv_coef.value()[j-1]
                var coef2 = model.sv_coef.value()[i]
                for k in range(ci):
                    sum += coef1.value()[si+k] * kvalue[si+k]
                for k in range(cj):
                    sum += coef2.value()[sj+k] * kvalue[sj+k]
                sum -= model.rho.value()[p]
                dec_values.value()[p] = sum

                if dec_values.value()[p] > 0:
                    vote[i] += 1
                else:
                    vote[j] += 1
                p += 1

        var vote_max_idx = 0
        for i in range(1, nr_class):
            if vote[i] > vote[vote_max_idx]:
                vote_max_idx = i

        kvalue.free()
        start.free()
        vote.free()
        return Float64(model.label.value()[vote_max_idx])

def svm_predict(model: svm_model, x: OptionalUnsafePointer[svm_node, MutExternalOrigin]) -> Float64:
    var nr_class = model.nr_class
    var dec_values: UnsafePointer[Float64, MutExternalOrigin]
    if model.param.svm_type == svm_parameter.ONE_CLASS or model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR:
        dec_values = alloc[Float64](1)
    else:
        dec_values = alloc[Float64](nr_class*(nr_class-1)//2)
    var pred_result = svm_predict_values(model, x, dec_values)
    dec_values.free()
    return pred_result

def svm_predict_probability(model: svm_model, x: OptionalUnsafePointer[svm_node, MutExternalOrigin], prob_estimates: OptionalUnsafePointer[Float64, MutExternalOrigin]) -> Float64:
    if (model.param.svm_type == svm_parameter.C_SVC or model.param.svm_type == svm_parameter.NU_SVC) and model.probA and model.probB:
        var nr_class = model.nr_class
        var dec_values = alloc[Float64](nr_class*(nr_class-1)//2)
        _ = svm_predict_values(model, x, dec_values)

        var min_prob=1e-7
        var pairwise_prob=alloc[OptionalUnsafePointer[Float64, MutExternalOrigin]](nr_class)
        for i in range(nr_class):
            pairwise_prob[i]=alloc[Float64](nr_class)
        var k=0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                pairwise_prob[i].value()[j]=min(max(sigmoid_predict(dec_values[k],model.probA.value()[k],model.probB.value()[k]),min_prob),1-min_prob)
                pairwise_prob[j].value()[i]=1-pairwise_prob[i].value()[j]
                k += 1
        if nr_class == 2:
            prob_estimates.value()[0] = pairwise_prob[0].value()[1]
            prob_estimates.value()[1] = pairwise_prob[1].value()[0]
        else:
            multiclass_probability(nr_class,pairwise_prob,prob_estimates)

        var prob_max_idx = 0
        for i in range(1, nr_class):
            if prob_estimates.value()[i] > prob_estimates.value()[prob_max_idx]:
                prob_max_idx = i
        for i in range(nr_class):
            pairwise_prob[i].value().free()
        dec_values.free()
        pairwise_prob.free()
        return Float64(model.label.value()[prob_max_idx])
    elif model.param.svm_type == svm_parameter.ONE_CLASS and model.prob_density_marks:
        var dec_value = 0.0
        var pred_result = svm_predict_values(model,x,UnsafePointer(to=dec_value))
        prob_estimates.value()[0] = predict_one_class_probability(model,dec_value)
        prob_estimates.value()[1] = 1-prob_estimates.value()[0]
        return pred_result
    else:
        return svm_predict(model, x)

def svm_decision_function(model: svm_model, x: OptionalUnsafePointer[svm_node, MutExternalOrigin]) -> Tuple[OptionalUnsafePointer[Float64, MutExternalOrigin], Int]:
    var nr_class = model.nr_class
    var l: Int
    var dec_values: UnsafePointer[Float64, MutExternalOrigin]
    if model.param.svm_type == svm_parameter.ONE_CLASS or model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR:
        l = 1
    else:
        l = nr_class*(nr_class-1)//2
    dec_values = alloc[Float64](l)
    _ = svm_predict_values(model, x, dec_values)
    return dec_values, l

def svm_free_model_content(mut model_ptr: svm_model):
    if model_ptr.free_sv and model_ptr.l > 0 and model_ptr.SV:
        model_ptr.SV.value()[0].value().free()
    if model_ptr.sv_coef:
        for i in range(model_ptr.nr_class-1):
            model_ptr.sv_coef.value()[i].value().free()

    model_ptr.SV.value().free()
    model_ptr.SV = None

    model_ptr.sv_coef.value().free()
    model_ptr.sv_coef = None

    model_ptr.rho.value().free()
    model_ptr.rho = None

    model_ptr.label.value().free()
    model_ptr.label = None

    model_ptr.probA.value().free()
    model_ptr.probA = None

    model_ptr.probB.value().free()
    model_ptr.probB = None

    model_ptr.prob_density_marks.value().free()
    model_ptr.prob_density_marks = None

    model_ptr.sv_indices.value().free()
    model_ptr.sv_indices = None

    model_ptr.nSV.value().free()
    model_ptr.nSV = None

def svm_free_and_destroy_model(mut model_ptr_ptr: OptionalUnsafePointer[svm_model, MutExternalOrigin]):
    if model_ptr_ptr:
        svm_free_model_content(model_ptr_ptr.value()[])
        model_ptr_ptr.value().free()
        model_ptr_ptr = OptionalUnsafePointer[svm_model, MutExternalOrigin]()

def svm_destroy_param(param: svm_parameter):
    if param.weight_label:
        param.weight_label.value().free()
    if param.weight:
        param.weight.value().free()

def svm_check_parameter(prob: svm_problem, param: svm_parameter) -> String:
    # svm_type

    var svm_type = param.svm_type
    if svm_type != svm_parameter.C_SVC and svm_type != svm_parameter.NU_SVC and svm_type != svm_parameter.ONE_CLASS and svm_type != svm_parameter.EPSILON_SVR and svm_type != svm_parameter.NU_SVR:
        return "unknown svm type"

    # kernel_type, degree

    var kernel_type = param.kernel_type
    if kernel_type != LINEAR and kernel_type != POLY and kernel_type != RBF and kernel_type != SIGMOID and kernel_type != PRECOMPUTED:
        return "unknown kernel type"

    if (kernel_type == POLY or kernel_type == RBF or kernel_type == SIGMOID) and param.gamma < 0:
        return "gamma < 0"

    if kernel_type == POLY and param.degree < 0:
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
            var this_label = Int(prob.y.value()[i])
            var j = 0
            while j<nr_class:
                if this_label == label[j]:
                    count[j] += 1
                    break
                j += 1
            if j == nr_class:
                if nr_class == max_nr_class:
                    var new = alloc[Int](max_nr_class*2)
                    memcpy(dest=new, src=label, count=max_nr_class)
                    label.free()
                    label = new
                    new = alloc[Int](max_nr_class*2)
                    memcpy(dest=new, src=count, count=max_nr_class)
                    count.free()
                    count = new
                label[nr_class] = this_label
                count[nr_class] = 1
                nr_class += 1

        for i in range(nr_class):
            var n1 = count[i]
            for j in range(i+1, nr_class):
                var n2 = count[j]
                if param.nu*Float64(n1+n2)/2 > Float64(min(n1,n2)):
                    label.free()
                    count.free()
                    return "specified nu is infeasible"

        label.free()
        count.free()

    return ""

def svm_check_probability_model(model: svm_model) -> Bool:
    return
        ((model.param.svm_type == svm_parameter.C_SVC or model.param.svm_type == svm_parameter.NU_SVC) and
        model.probA and model.probB) or
        (model.param.svm_type == svm_parameter.ONE_CLASS and model.prob_density_marks) or
        ((model.param.svm_type == svm_parameter.EPSILON_SVR or model.param.svm_type == svm_parameter.NU_SVR) and
        model.probA)
