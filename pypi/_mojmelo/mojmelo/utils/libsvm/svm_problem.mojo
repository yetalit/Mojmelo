from .svm_node import svm_node

struct svm_problem(RegisterPassable):
	var l: Int
	var y: UnsafePointer[Float64, MutUntrackedOrigin]
	var x: UnsafePointer[UnsafePointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]

	@always_inline
	def __init__(out self):
		self.l = 0
		self.y = UnsafePointer[Float64, MutUntrackedOrigin].unsafe_dangling()
		self.x = UnsafePointer[UnsafePointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()
