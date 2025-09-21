from memory import memcpy, memset_zero
from .svm_node import svm_node

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
        self.size = size // 4
        self.head = UnsafePointer[head_t].alloc(1)
        self.head.init_pointee_move(head_t(UnsafePointer[head_t](), UnsafePointer[head_t](), 0)) # initialized to 0
        var header_size = UInt(l) * 8
        self.size = max(size, UInt(2) * UInt(l) + header_size) - header_size  # cache must be large enough for two columns
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
            memcpy(new, h[].data, h[]._len)
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
