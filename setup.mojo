from sys import external_call, os_is_linux, os_is_macos, argv
from sys.ffi import *
from memory import UnsafePointer
from collections import InlineArray
import math

fn is_power_of_2(x: Float64) -> Bool:
    _x = int(x)
    return (_x & (_x - 1)) == 0

fn cachel1() -> Int32:
    var l1_cache_size: c_int = 0
    alias length: c_size_t = 4
    alias hw_perflevel0_l1dcachesize = InlineArray[Int8, 27](104,119,46,112,101,114,102,108,101,118,101,108,48,46,108,49,100,99,97,99,104,101,115,105,122,101,0)
    alias hw_l1dcachesize = InlineArray[Int8, 16](104,119,46,108,49,100,99,97,99,104,101,115,105,122,101,0)
    # Get L1 Cache Size
    if external_call["sysctlbyname", c_int, UnsafePointer[c_char], UnsafePointer[c_int], UnsafePointer[c_size_t], OpaquePointer, c_size_t](hw_perflevel0_l1dcachesize.unsafe_ptr(), UnsafePointer.address_of(l1_cache_size), UnsafePointer.address_of(length), UnsafePointer[NoneType](), 0) == 0:
        if l1_cache_size <= 1:
            if external_call["sysctlbyname", c_int, UnsafePointer[c_char], UnsafePointer[c_int], UnsafePointer[c_size_t], OpaquePointer, c_size_t](hw_l1dcachesize.unsafe_ptr(), UnsafePointer.address_of(l1_cache_size), UnsafePointer.address_of(length), UnsafePointer[NoneType](), 0) == 0:
                if l1_cache_size <= 1:
                    return 65536
                return l1_cache_size
            else:
                return 65536
        return l1_cache_size
    else:
        if external_call["sysctlbyname", c_int, UnsafePointer[c_char], UnsafePointer[c_int], UnsafePointer[c_size_t], OpaquePointer, c_size_t](hw_l1dcachesize.unsafe_ptr(), UnsafePointer.address_of(l1_cache_size), UnsafePointer.address_of(length), UnsafePointer[NoneType](), 0) == 0:
            if l1_cache_size <= 1:
                return 65536
            return l1_cache_size
        else:
            return 65536


fn cachel2() -> Int32:
    var l2_cache_size: c_int = 0
    alias length: c_size_t = 4
    alias hw_perflevel0_l2dcachesize = InlineArray[Int8, 27](104,119,46,112,101,114,102,108,101,118,101,108,48,46,108,50,100,99,97,99,104,101,115,105,122,101,0)
    alias hw_l2dcachesize = InlineArray[Int8, 16](104,119,46,108,50,100,99,97,99,104,101,115,105,122,101,0)
    # Get L2 Cache Size
    if external_call["sysctlbyname", c_int, UnsafePointer[c_char], UnsafePointer[c_int], UnsafePointer[c_size_t], OpaquePointer, c_size_t](hw_perflevel0_l2dcachesize.unsafe_ptr(), UnsafePointer.address_of(l2_cache_size), UnsafePointer.address_of(length), UnsafePointer[NoneType](), 0) == 0:
        if l2_cache_size <= 1:
            if external_call["sysctlbyname", c_int, UnsafePointer[c_char], UnsafePointer[c_int], UnsafePointer[c_size_t], OpaquePointer, c_size_t](hw_l2dcachesize.unsafe_ptr(), UnsafePointer.address_of(l2_cache_size), UnsafePointer.address_of(length), UnsafePointer[NoneType](), 0) == 0:
                if l2_cache_size <= 1:
                    return 4194304
                return l2_cache_size
            else:
                return 4194304
        return l2_cache_size
    else:
        if external_call["sysctlbyname", c_int, UnsafePointer[c_char], UnsafePointer[c_int], UnsafePointer[c_size_t], OpaquePointer, c_size_t](hw_l2dcachesize.unsafe_ptr(), UnsafePointer.address_of(l2_cache_size), UnsafePointer.address_of(length), UnsafePointer[NoneType](), 0) == 0:
            if l2_cache_size <= 1:
                return 4194304
            return l2_cache_size
        else:
            return 4194304


fn main() raises:
    if os_is_linux():
            with open("/sys/devices/system/cpu/cpu0/cache/index0/size", "r") as f:
                cache_l1_size = atol(f.read().split('K')[0]) * 1024
            with open("/sys/devices/system/cpu/cpu0/cache/index0/ways_of_associativity", "r") as f:
                cache_l1_associativity = atol(f.read())
            with open("/sys/devices/system/cpu/cpu0/cache/index2/size", "r") as f:
                cache_l2_size = atol(f.read().split('K')[0]) * 1024
            with open("/sys/devices/system/cpu/cpu0/cache/index2/ways_of_associativity", "r") as f:
                cache_l2_associativity = atol(f.read())
            with open("./mojmelo/utils/params.mojo", "w") as f:
                code = 'alias L1_CACHE_SIZE = ' + str(cache_l1_size) + '\n'
                code += 'alias L1_ASSOCIATIVITY = ' + str(cache_l1_associativity) + '\n'
                code += 'alias L2_CACHE_SIZE = ' + str(cache_l2_size) + '\n'
                code += 'alias L2_ASSOCIATIVITY = ' + str(cache_l2_associativity) + '\n'
                f.write(code)
            print('Setup Done!')
    if os_is_macos():
        if len(argv()) == 1:
            cache_l1_size = int(cachel1())
            cache_l2_size = int(cachel2())
            possible_l1_associativity = cache_l1_size / 4096
            possible_l2_associativity = cache_l2_size / 65536
            possible_l1_associativities = List[Int](
                                            int(2 ** (math.log2(possible_l1_associativity) - 1.0)) if is_power_of_2(possible_l1_associativity) else int(2 ** math.floor(math.log2(possible_l1_associativity))),
                                            int(possible_l1_associativity),
                                            int(2 ** (math.log2(possible_l1_associativity) + 1.0)) if is_power_of_2(possible_l1_associativity) else int(2 ** math.ceil(math.log2(possible_l1_associativity)))
                                        )
            possible_l2_associativities = List[Int](
                                            int(2 ** (math.log2(possible_l2_associativity) - 1.0)) if is_power_of_2(possible_l2_associativity) else int(2 ** math.floor(math.log2(possible_l2_associativity))),
                                            int(possible_l2_associativity),
                                            int(2 ** (math.log2(possible_l2_associativity) + 1.0)) if is_power_of_2(possible_l2_associativity) else int(2 ** math.ceil(math.log2(possible_l2_associativity)))
                                        )
            with open("./mojmelo/utils/params.mojo", "w") as f:
                code = 'alias L1_CACHE_SIZE = ' + str(cache_l1_size) + '\n'
                code += 'alias L1_ASSOCIATIVITY = ' + str(possible_l1_associativities[0]) + '\n'
                code += 'alias L2_CACHE_SIZE = ' + str(cache_l2_size) + '\n'
                code += 'alias L2_ASSOCIATIVITY = ' + str(possible_l2_associativities[0]) + '\n'
                f.write(code)
            for i in range(3):
                for j in range(1, 4):
                    with open("./param" + str(i * 3 + j), "w") as f:
                        code = 'alias L1_CACHE_SIZE = ' + str(cache_l1_size) + '\n'
                        code += 'alias L1_ASSOCIATIVITY = ' + str(possible_l1_associativities[i]) + '\n'
                        code += 'alias L2_CACHE_SIZE = ' + str(cache_l2_size) + '\n'
                        code += 'alias L2_ASSOCIATIVITY = ' + str(possible_l2_associativities[j - 1]) + '\n'
                        f.write(code)
            print('Setup initialization Done!')
        else:
            command = str(argv()[1])

            from mojmelo.utils.Matrix import Matrix
            from collections import InlineArray
            import time

            alias NUM_ITER = 25
            results = InlineArray[Int, 3](fill=0)
            var junk: Float32 = 0.0
            a = Matrix.random(512, 4096)
            b = Matrix.random(4096, 512)
            for i in range(NUM_ITER):
                t = time.perf_counter_ns()
                c = a * b
                seconds = time.perf_counter_ns() - t
                junk += c[0, 0]
                if i != 0:
                    results[0] += seconds // (NUM_ITER - 1)
            a = Matrix.random(4096, 4096)
            b = Matrix.random(4096, 4096)
            for i in range(NUM_ITER):
                t = time.perf_counter_ns()
                c = a * b
                seconds = time.perf_counter_ns() - t
                junk += c[0, 0]
                if i != 0:
                    results[1] += seconds // (NUM_ITER - 1)
            a = Matrix.random(4096, 512)
            b = Matrix.random(512, 4096)
            for i in range(NUM_ITER):
                t = time.perf_counter_ns()
                c = a * b
                seconds = time.perf_counter_ns() - t
                junk += c[0, 0]
                if i != 0:
                    results[2] += seconds // (NUM_ITER - 1)
            if command != '9':
                with open("./results" + command, "w") as f:
                    f.write(str(results[0]) + ',' + str(results[1]) + ',' + str(results[2]) + ',' + str(junk))
                code = ''
                with open("./param" + str(int(command) + 1), "r") as f:
                    code = f.read()
                with open("./mojmelo/utils/params.mojo", "w") as f:
                    f.write(code)
                print('Setup', command, 'Done!')
            else:
                results_list = List[InlineArray[Int, 3]]()
                for i in range(1, 9):
                    with open("./results" + str(i), "r") as f:
                        res = f.read().split(',')
                        results_list.append(InlineArray[Int, 3](fill=0))
                        results_list[i - 1][0] = atol(res[0])
                        results_list[i - 1][1] = atol(res[1])
                        results_list[i - 1][2] = atol(res[2])
                results_list.append(results)
                
                from collections import Counter

                votes = List[Int]()
                for i in range(3):
                    _min = results_list[0][i]
                    m_index = 0
                    for j in range(9):
                        if results_list[j][i] < _min:
                            _min = results_list[j][i]
                            m_index = j
                    votes.append(m_index)

                code = ''
                with open("./param" + str(Counter[Int](votes).most_common(1)[0]._value + 1), "r") as f:
                    code = f.read()
                with open("./mojmelo/utils/params.mojo", "w") as f:
                    f.write(code)

                from python import Python
                ospy = Python.import_module("os")
                for i in range(1, 10):
                    ospy.remove("./param" + str(i))
                    if i != 9:
                        ospy.remove("./results" + str(i))
                print('Setup Done!')
