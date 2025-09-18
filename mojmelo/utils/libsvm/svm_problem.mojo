from .svm_node import svm_node

struct svm_problem:
	var l: Int
	var y: UnsafePointer[Float64]
	var x: UnsafePointer[UnsafePointer[svm_node]]
