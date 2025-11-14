@fieldwise_init
struct svm_parameter(Copyable, Movable):
	# svm_type
	comptime C_SVC: Int = 0
	comptime NU_SVC: Int = 1
	comptime ONE_CLASS: Int = 2
	comptime EPSILON_SVR: Int = 3
	comptime NU_SVR: Int = 4

	# kernel_type
	comptime LINEAR: Int = 0
	comptime POLY: Int = 1
	comptime RBF: Int = 2
	comptime SIGMOID: Int = 3
	comptime PRECOMPUTED: Int = 4

	var svm_type: Int
	var kernel_type: Int
	var degree: Int	# for poly
	var gamma: Float64	# for poly/rbf/sigmoid
	var coef0: Float64	# for poly/sigmoid

	# these are for training only
	var cache_size: Float64 # in MB
	var eps: Float64	# stopping criteria
	var C: Float64	# for C_SVC, EPSILON_SVR and NU_SVR
	var nr_weight: Int		# for C_SVC
	var weight_label: UnsafePointer[Int, MutOrigin.external]	# for C_SVC
	var weight: UnsafePointer[Float64, MutOrigin.external]		# for C_SVC
	var nu: Float64	# for NU_SVC, ONE_CLASS, and NU_SVR
	var p: Float64	# for EPSILON_SVR
	var shrinking: Int	# use the shrinking heuristics
	var probability: Int # do probability estimates
