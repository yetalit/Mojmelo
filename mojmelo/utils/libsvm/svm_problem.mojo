from .svm_node import svm_node

struct svm_problem:
	var l: Int
	var y: UnsafePointer[Float64]
	var x: UnsafePointer[UnsafePointer[svm_node]]

	@always_inline
	fn __init__(out self):
		self.l = 0
		self.y = UnsafePointer[Float64]()
		self.x = UnsafePointer[UnsafePointer[svm_node]]()
