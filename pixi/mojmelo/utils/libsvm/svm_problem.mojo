from .svm_node import svm_node

struct svm_problem(TrivialRegisterPassable):
	var l: Int
	var y: UnsafePointer[Float64, MutExternalOrigin]
	var x: UnsafePointer[UnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]

	@always_inline
	def __init__(out self):
		self.l = 0
		self.y = UnsafePointer[Float64, MutExternalOrigin]()
		self.x = UnsafePointer[UnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]()
