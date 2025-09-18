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
# Kernel Cache
#
# l is the number of total data items
# size is the cache size limit in bytes
struct Cache:
	var l: Int
	var size: UInt
	var head: UnsafePointer[head_t]
	var lru_head: head_t
