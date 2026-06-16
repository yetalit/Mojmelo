from .svm_node import svm_node

struct svm_problem(RegisterPassable):
	var l: Int
	var y: OptionalUnsafePointer[Float64, MutUntrackedOrigin]
	var x: OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]

	@always_inline
	def __init__(out self):
		self.l = 0
		self.y = OptionalUnsafePointer[Float64, MutUntrackedOrigin]()
		self.x = OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]()
