from sys import external_call, os_is_linux, os_is_macos, argv
from sys.ffi import *
from memory import UnsafePointer

fn cachel1() -> Int32:
    var l1_cache_size: c_int = 0
    alias length: c_size_t = 4
    # Get L1 Cache Size
    if external_call["sysctlbyname", c_int]("hw.perflevel0.l1dcachesize".unsafe_cstr_ptr(), UnsafePointer(to=l1_cache_size), UnsafePointer(to=length), OpaquePointer(), 0) == 0:
        if l1_cache_size <= 1:
            if external_call["sysctlbyname", c_int]("hw.l1dcachesize".unsafe_cstr_ptr(), UnsafePointer(to=l1_cache_size), UnsafePointer(to=length), OpaquePointer(), 0) == 0:
                if l1_cache_size <= 1:
                    return 65536
                return l1_cache_size
            else:
                return 65536
        return l1_cache_size
    else:
        if external_call["sysctlbyname", c_int]("hw.l1dcachesize".unsafe_cstr_ptr(), UnsafePointer(to=l1_cache_size), UnsafePointer(to=length), OpaquePointer(), 0) == 0:
            if l1_cache_size <= 1:
                return 65536
            return l1_cache_size
        else:
            return 65536


fn cachel2() -> Int32:
    var l2_cache_size: c_int = 0
    alias length: c_size_t = 4
    # Get L2 Cache Size
    if external_call["sysctlbyname", c_int]("hw.perflevel0.l2cachesize".unsafe_cstr_ptr(), UnsafePointer(to=l2_cache_size), UnsafePointer(to=length), OpaquePointer(), 0) == 0:
        if l2_cache_size <= 1:
            if external_call["sysctlbyname", c_int]("hw.l2cachesize".unsafe_cstr_ptr(), UnsafePointer(to=l2_cache_size), UnsafePointer(to=length), OpaquePointer(), 0) == 0:
                if l2_cache_size <= 1:
                    return 4194304
                return l2_cache_size
            else:
                return 4194304
        return l2_cache_size
    else:
        if external_call["sysctlbyname", c_int]("hw.l2cachesize".unsafe_cstr_ptr(), UnsafePointer(to=l2_cache_size), UnsafePointer(to=length), OpaquePointer(), 0) == 0:
            if l2_cache_size <= 1:
                return 4194304
            return l2_cache_size
        else:
            return 4194304


fn initialize(cache_l1_size: Int, cache_l1_associativity: Int, cache_l2_size: Int, cache_l2_associativity: Int) raises:
    if cache_l1_associativity <= 1 or cache_l2_associativity <= 1:
        possible_l1_associativities = InlineArray[Int, 3](fill=0)
        if cache_l1_associativity > 1:
            possible_l1_associativities[0] = possible_l1_associativities[1] = possible_l1_associativities[2] = cache_l1_associativity
        else:
            possible_l1_associativities[0] = 4 if cache_l1_size < 65534 else 8
            possible_l1_associativities[1] = possible_l1_associativities[0] * 2
            possible_l1_associativities[2] = 12
        possible_l2_associativities = InlineArray[Int, 3](fill=0)
        if cache_l2_associativity > 1:
            possible_l2_associativities[0] = possible_l2_associativities[1] = possible_l2_associativities[2] = cache_l2_associativity
        else:
            possible_l2_associativities[0] = 4 if cache_l2_size <= 2097154 else 8
            possible_l2_associativities[1] = possible_l2_associativities[0] * 2
            possible_l2_associativities[2] = possible_l2_associativities[0] * 4
        with open("./mojmelo/utils/mojmelo_matmul/params.mojo", "w") as f:
            code = 'alias L1_CACHE_SIZE = ' + String(cache_l1_size) + '\n'
            code += 'alias L1_ASSOCIATIVITY = ' + String(possible_l1_associativities[0]) + '\n'
            code += 'alias L2_CACHE_SIZE = ' + String(cache_l2_size) + '\n'
            code += 'alias L2_ASSOCIATIVITY = ' + String(possible_l2_associativities[0]) + '\n'
            f.write(code)
        for i in range(3):
            for j in range(1, 4):
                with open("./param" + String(i * 3 + j), "w") as f:
                    code = 'alias L1_CACHE_SIZE = ' + String(cache_l1_size) + '\n'
                    code += 'alias L1_ASSOCIATIVITY = ' + String(possible_l1_associativities[i]) + '\n'
                    code += 'alias L2_CACHE_SIZE = ' + String(cache_l2_size) + '\n'
                    code += 'alias L2_ASSOCIATIVITY = ' + String(possible_l2_associativities[j - 1]) + '\n'
                    f.write(code)
    else:
        with open("./mojmelo/utils/mojmelo_matmul/params.mojo", "w") as f:
            code = 'alias L1_CACHE_SIZE = ' + String(cache_l1_size) + '\n'
            code += 'alias L1_ASSOCIATIVITY = ' + String(cache_l1_associativity) + '\n'
            code += 'alias L2_CACHE_SIZE = ' + String(cache_l2_size) + '\n'
            code += 'alias L2_ASSOCIATIVITY = ' + String(cache_l2_associativity) + '\n'
            f.write(code)
        with open("./done", "w") as f:
            f.write("done")
    print('Setup initialization done!')

fn main() raises:
    if len(argv()) == 1:
        cache_l1_size = 0
        cache_l2_size = 0
        cache_l1_associativity = 0
        cache_l2_associativity = 0
        if os_is_linux():
            with open("/sys/devices/system/cpu/cpu0/cache/index0/size", "r") as f:
                txt = f.read()
                if txt.find('K') != -1:
                    cache_l1_size = atol(txt.split('K')[0]) * 1024
                else:
                    cache_l1_size = atol(txt.split('M')[0]) * 1048576
            try:
                with open("/sys/devices/system/cpu/cpu0/cache/index0/ways_of_associativity", "r") as f:
                    cache_l1_associativity = atol(f.read())
            except:
                cache_l1_associativity = 0
            with open("/sys/devices/system/cpu/cpu0/cache/index2/size", "r") as f:
                txt = f.read()
                if txt.find('K') != -1:
                    cache_l2_size = atol(txt.split('K')[0]) * 1024
                else:
                    cache_l2_size = atol(txt.split('M')[0]) * 1048576
            try:
                with open("/sys/devices/system/cpu/cpu0/cache/index2/ways_of_associativity", "r") as f:
                    cache_l2_associativity = atol(f.read())
            except:
                cache_l2_associativity = 0
        elif os_is_macos():
            cache_l1_size = Int(cachel1())
            cache_l2_size = Int(cachel2())
        initialize(cache_l1_size, cache_l1_associativity, cache_l2_size, cache_l2_associativity)
    else:
        command = String(argv()[1])

        from python import Python
        os_py = Python.import_module("os")
        os_path_py = Python.import_module("os.path")
        if os_path_py.isfile('./done'):
            if command != '9':
                print('Setup', command + '/8', 'skipped!')
            else:
                os_py.remove("./done")
                print('Setup done!')
            return

        from mojmelo.utils.Matrix import Matrix
        import time

        alias NUM_ITER = 16
        results = InlineArray[Int, 3](fill=0)
        var junk: Float32 = 0.0
        a = Matrix.random(512, 4096)
        b = Matrix.random(4096, 512)
        for i in range(NUM_ITER):
            start = time.perf_counter_ns()
            c = a * b
            finish = time.perf_counter_ns()
            junk += c[0, 0]
            if i != 0:
                results[0] += (finish - start) // (NUM_ITER - 1)
        a = Matrix.random(4096, 4096)
        b = Matrix.random(4096, 4096)
        for i in range(NUM_ITER):
            start = time.perf_counter_ns()
            c = a * b
            finish = time.perf_counter_ns()
            junk += c[0, 0]
            if i != 0:
                results[1] += (finish - start) // (NUM_ITER - 1)
        a = Matrix.random(4096, 512)
        b = Matrix.random(512, 4096)
        for i in range(NUM_ITER):
            start = time.perf_counter_ns()
            c = a * b
            finish = time.perf_counter_ns()
            junk += c[0, 0]
            if i != 0:
                results[2] += (finish - start) // (NUM_ITER - 1)
        if command != '9':
            with open("./results" + command, "w") as f:
                f.write(String(results[0]) + ',' + String(results[1]) + ',' + String(results[2]) + ',' + String(junk))
            var code: String
            with open("./param" + String(Int(command) + 1), "r") as f:
                code = f.read()
            with open("./mojmelo/utils/mojmelo_matmul/params.mojo", "w") as f:
                f.write(code)
            print('Setup', command + '/8', 'done!')
        else:
            results_list = List[InlineArray[Int, 3]]()
            for i in range(1, 9):
                with open("./results" + String(i), "r") as f:
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

            var code: String
            with open("./param" + String(Counter[Int](votes).most_common(1)[0]._value + 1), "r") as f:
                code = f.read()
            with open("./mojmelo/utils/mojmelo_matmul/params.mojo", "w") as f:
                f.write(code)

            for i in range(1, 10):
                os_py.remove("./param" + String(i))
                if i != 9:
                    os_py.remove("./results" + String(i))
            print('Setup done!')
