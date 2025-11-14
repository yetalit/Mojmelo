from .svm_parameter import svm_parameter
from .svm_node import svm_node

struct svm_model:
	var param: svm_parameter # parameter
	var nr_class: Int # number of classes, = 2 in regression/one class svm
	var l: Int # total SV
	var SV: UnsafePointer[UnsafePointer[svm_node, MutOrigin.external], MutOrigin.external] # SVs (SV[l])
	var sv_coef: UnsafePointer[UnsafePointer[Float64, MutOrigin.external], MutOrigin.external]	# coefficients for SVs in decision functions (sv_coef[k-1][l])
	var rho: UnsafePointer[Float64, MutOrigin.external]	# constants in decision functions (rho[k*(k-1)/2])
	var probA: UnsafePointer[Float64, MutOrigin.external] # pariwise probability information
	var probB: UnsafePointer[Float64, MutOrigin.external]
	var prob_density_marks: UnsafePointer[Float64, MutOrigin.external] # probability information for ONE_CLASS
	var sv_indices: UnsafePointer[Scalar[DType.int], MutOrigin.external] # sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

	# for classification only

	var label: UnsafePointer[Int, MutOrigin.external] # label of each class (label[k])
	var nSV: UnsafePointer[Int, MutOrigin.external]	# number of SVs for each class (nSV[k])
				# nSV[0] + nSV[1] + ... + nSV[k-1] = l
	var free_sv: Int # 1 if svm_model is created by svm_load_model
				# 0 if svm_model is created by svm_train
