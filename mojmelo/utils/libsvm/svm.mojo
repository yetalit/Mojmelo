from memory import memset_zero

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
    var len: Int		# data[0,len) is cached in this entry

    @always_inline
    fn __init__(out self, p: UnsafePointer[head_t], n: UnsafePointer[head_t], len: Int):
        self.prev = p
        self.next = n
        self.len = len
        self.data = UnsafePointer[Float32].alloc(len)
        memset_zero(self.data, self.len)

    @always_inline
    fn __del__(var self):
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

    fn __del__(var self):
        var h = self.lru_head.next
        while h != UnsafePointer(to=self.lru_head):
            h.destroy_pointee()
            h = h[].next
        if self.head:
            self.head.free()
