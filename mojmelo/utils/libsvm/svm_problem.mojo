from .svm_node import svm_node

@register_passable("trivial")
struct svm_problem:
	var l: Int
	var y: UnsafePointer[Float64, MutOrigin.external]
	var x: UnsafePointer[UnsafePointer[svm_node, MutOrigin.external], MutOrigin.external]

	@always_inline
	fn __init__(out self):
		self.l = 0
		self.y = UnsafePointer[Float64, MutOrigin.external]()
		self.x = UnsafePointer[UnsafePointer[svm_node, MutOrigin.external], MutOrigin.external]()
