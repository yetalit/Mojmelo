from .svm_node import svm_node

struct svm_problem(RegisterPassable):
	var l: Int
	var y: OptionalUnsafePointer[Float64, MutExternalOrigin]
	var x: OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]

	@always_inline
	def __init__(out self):
		self.l = 0
		self.y = OptionalUnsafePointer[Float64, MutExternalOrigin]()
		self.x = OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]()
