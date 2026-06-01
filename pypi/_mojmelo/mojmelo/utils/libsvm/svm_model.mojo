from .svm_parameter import svm_parameter
from .svm_node import svm_node

struct svm_model:
	var param: svm_parameter # parameter
	var nr_class: Int # number of classes, = 2 in regression/one class svm
	var l: Int # total SV
	var SV: OptionalUnsafePointer[OptionalUnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin] # SVs (SV[l])
	var sv_coef: OptionalUnsafePointer[OptionalUnsafePointer[Float64, MutExternalOrigin], MutExternalOrigin]	# coefficients for SVs in decision functions (sv_coef[k-1][l])
	var rho: OptionalUnsafePointer[Float64, MutExternalOrigin]	# constants in decision functions (rho[k*(k-1)/2])
	var probA: OptionalUnsafePointer[Float64, MutExternalOrigin] # pariwise probability information
	var probB: OptionalUnsafePointer[Float64, MutExternalOrigin]
	var prob_density_marks: OptionalUnsafePointer[Float64, MutExternalOrigin] # probability information for ONE_CLASS
	var sv_indices: OptionalUnsafePointer[Scalar[DType.int], MutExternalOrigin] # sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

	# for classification only

	var label: OptionalUnsafePointer[Int, MutExternalOrigin] # label of each class (label[k])
	var nSV: OptionalUnsafePointer[Int, MutExternalOrigin]	# number of SVs for each class (nSV[k])
				# nSV[0] + nSV[1] + ... + nSV[k-1] = l
	var free_sv: Int # 1 if svm_model is created by svm_load_model
				# 0 if svm_model is created by svm_train
