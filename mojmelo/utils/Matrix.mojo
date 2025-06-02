from .mojmelo_matmul import matmul
from sys.info import simdwidthof, is_apple_silicon
from memory import memcpy, memcmp, memset_zero, UnsafePointer
from algorithm import vectorize, parallelize
from buffer import NDBuffer
import algorithm
from collections import Dict, Set
import math
import random
from mojmelo.utils.utils import argn, cov_value, complete_orthonormal_basis, add, sub, mul, div
from python import Python, PythonObject

struct Matrix(Stringable, Writable):
    var height: Int
    var width: Int
    var size: Int
    var data: UnsafePointer[Float32]
    var order: String
    alias simd_width: Int = 4 * simdwidthof[DType.float32]() if is_apple_silicon() else 2 * simdwidthof[DType.float32]()

    # initialize from UnsafePointer
    @always_inline
    fn __init__(out self, data: UnsafePointer[Float32], height: Int, width: Int, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = data
        self.order = order.lower()

    # initialize by copying from UnsafePointer
    @always_inline
    fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float32] = UnsafePointer[Float32](), order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = order.lower()
        if data:
            memcpy(self.data, data, self.size)

    # initialize from List
    fn __init__(out self, height: Int, width: Int, def_input: List[Float32]):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = 'c'
        if len(def_input) > 0:
            memcpy(self.data, def_input.data, self.size)

    # initialize from list object
    fn __init__(out self, height: Int, width: Int, def_input: PythonObject) raises:
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = 'c'
        var rng: Int = len(def_input)
        for i in range(rng):
            self.data[i] = atof(String(def_input[i])).cast[DType.float32]()

    # initialize in 2D numpy style
    fn __init__(out self, npstyle: String, order: String = 'c') raises:
        var mat = npstyle.replace(' ', '')
        if mat[0] == '[' and mat[1] == '[' and mat[len(mat) - 1] == ']' and mat[len(mat) - 2] == ']':
            self.width = 0
            self.size = 0
            self.data = UnsafePointer[Float32]()
            self.order = order.lower()
            var rows = mat[:-1].split(']')
            self.height = len(rows) - 1
            for i in range(self.height):
                var values = rows[i][2:].split(',')
                if i == 0:
                    self.width = len(values)
                    self.size = self.height * self.width
                    self.data = UnsafePointer[Float32].alloc(self.size)
                for j in range(self.width):
                    self.store[1](i, j, atof(values[j]).cast[DType.float32]())
        else:
            raise Error('Error: Matrix is not initialized in the correct form!')

    fn __copyinit__(out self, other: Self):
        self.height = other.height
        self.width = other.width
        self.size = other.size
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = other.order
        memcpy(self.data, other.data, self.size)

    fn __moveinit__(out self, owned existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.size = existing.size
        self.data = existing.data
        self.order = existing.order
        existing.height = existing.width = existing.size = 0
        existing.order = ''
        existing.data = UnsafePointer[Float32]()

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        var loc: Int
        if self.order == 'c':
            loc = (y * self.width) + x
        else:
            loc = (x * self.height) + y
        return self.data.load[width=nelts](loc)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        var loc: Int
        if self.order == 'c':
            loc = (y * self.width) + x
        else:
            loc = (x * self.height) + y
        return self.data.store(loc, val)

    # access an element
    @always_inline
    fn __getitem__(self, row: Int, column: Int) raises -> Float32:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        return self.data[loc]

    # access a row
    @always_inline
    fn __getitem__(self, row: Int) raises -> Matrix:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width, self.data + (row * self.width), self.order)
        var mat = Matrix(1, self.width, order= self.order)
        var tmpPtr = self.data + row
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a row (unsafe)
    @always_inline
    fn __getitem__(self, row: Int, *, unsafe: Bool) -> Matrix:
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width, self.data + (row * self.width), self.order)
        var mat = Matrix(1, self.width, order= self.order)
        var tmpPtr = self.data + row
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a row with offset
    @always_inline
    fn __getitem__(self, row: Int, offset: Bool, start_i: Int) raises -> Matrix:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width - start_i, self.data + (row * self.width) + start_i, self.order)
        var mat = Matrix(1, self.width - start_i, order= self.order)
        var tmpPtr = self.data + row + (start_i * self.height)
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a column
    @always_inline
    fn __getitem__(self, row: String, column: Int) raises -> Matrix:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height, 1)
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height, 1, self.data + (column * self.height), self.order)

    # access a column (unsafe)
    @always_inline
    fn __getitem__(self, row: String, column: Int, *, unsafe: Bool) -> Matrix:
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height, 1)
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height, 1, self.data + (column * self.height), self.order)

    # access a column with offset
    @always_inline
    fn __getitem__(self, offset: Bool, start_i: Int, column: Int) raises -> Matrix:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height - start_i, 1)
            var tmpPtr = self.data + column + (start_i * self.width)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height - start_i, 1, self.data + (column * self.height) + start_i, self.order)

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: Matrix) raises -> Matrix:
        var mat = Matrix(rows.size, self.width, order= self.order)
        if rows.size > 96:
            @parameter
            fn p(i: Int):
                mat[i, unsafe=True] = self[Int(rows.data[i]), unsafe=True]
            parallelize[p](rows.size)
        else:
            for i in range(rows.size):
                mat[i] = self[Int(rows.data[i])]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: Matrix) raises -> Matrix:
        var mat = Matrix(self.height, columns.size, order= self.order)
        if columns.size > 96 or (self.order == 'c' and self.height * columns.size > 24576):
            @parameter
            fn p(i: Int):
                mat[row, i, unsafe=True] = self[row, Int(columns.data[i]), unsafe=True]
            parallelize[p](columns.size)
        else:
            for i in range(columns.size):
                mat[row, i] = self[row, Int(columns.data[i])]
        return mat^

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: List[Int]) raises -> Matrix:
        var mat = Matrix(len(rows), self.width, order= self.order)
        if len(rows) > 96:
            @parameter
            fn p(i: Int):
                mat[i, unsafe=True] = self[rows[i], unsafe=True]
            parallelize[p](len(rows))
        else:
            for i in range(mat.height):
                mat[i] = self[rows[i]]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: List[Int]) raises -> Matrix:
        var mat = Matrix(self.height, len(columns), order= self.order)
        if len(columns) > 96 or (self.order == 'c' and self.height * len(columns) > 24576):
            @parameter
            fn p(i: Int):
                mat[row, i, unsafe=True] = self[row, columns[i], unsafe=True]
            parallelize[p](len(columns))
        else:
            for i in range(mat.width):
                mat[row, i] = self[row, columns[i]]
        return mat^
    
    # replace an element
    @always_inline
    fn __setitem__(mut self, row: Int, column: Int, val: Float32) raises:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        self.data[loc] = val
    
    # replace the given row
    @always_inline
    fn __setitem__(mut self, row: Int, val: Matrix) raises:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width), val.data, val.size)
        else:
            var tmpPtr = self.data + row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given row (unsafe)
    @always_inline
    fn __setitem__(mut self, row: Int, val: Matrix, *, unsafe: Bool):
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width), val.data, val.size)
        else:
            var tmpPtr = self.data + row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given row with offset
    @always_inline
    fn __setitem__(mut self, row: Int, offset: Bool, start_i: Int, val: Matrix) raises:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width) + start_i, val.data, val.size)
        else:
            var tmpPtr = self.data + row + (start_i * self.height)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given column
    @always_inline
    fn __setitem__(mut self, row: String, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height), val.data, val.size)

    # replace the given column (unsafe)
    @always_inline
    fn __setitem__(mut self, row: String, column: Int, val: Matrix, *, unsafe: Bool):
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height), val.data, val.size)

    # replace the given column with offset
    @always_inline
    fn __setitem__(mut self, offset: Bool, start_i: Int, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column + (start_i * self.width)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height) + start_i, val.data, val.size)

    # replace given rows (by their indices)
    @always_inline
    fn __setitem__(mut self, rows: Matrix, rhs: Matrix) raises:
        for i in range(rows.size):
            self[Int(rows.data[i])] = rhs[i]

    # replace given columns (by their indices)
    @always_inline
    fn __setitem__(mut self, row: String, columns: Matrix, rhs: Matrix) raises:
        for i in range(columns.size):
            self[row, Int(columns.data[i])] = rhs[row, i]
    
    @always_inline
    fn load_columns(self, _range: Int) raises -> Matrix:
        if _range > self.width:
            raise Error("Error: Index out of range!")
        var mat = Matrix(self.height, _range, order=self.order)
        if self.order == 'f' or self.height == 1:
            memcpy(mat.data, self.data, mat.size)
        else:
            @parameter
            fn p(i: Int):
                memcpy(mat.data + i * _range, self.data + i * self.width, _range)
            parallelize[p](self.height)
        return mat^

    @always_inline
    fn load_rows(self, _range: Int) raises -> Matrix:
        if _range > self.height:
            raise Error("Error: Index out of range!")
        var mat = Matrix(_range, self.width, order=self.order)
        if self.order == 'c' or self.width == 1:
            memcpy(mat.data, self.data, mat.size)
        else:
            @parameter
            fn p(i: Int):
                memcpy(mat.data + i * _range, self.data + i * self.height, _range)
            parallelize[p](self.width)
        return mat^

    @always_inline
    fn __del__(owned self):
        if self.data:
            self.data.free()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn __eq__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] == rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] == rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __ne__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] != rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] != rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __gt__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] > rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] > rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __ge__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] >= rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] >= rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __lt__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] < rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] < rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __le__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] <= rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] <= rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        return self.height == rhs.height and self.width == rhs.width and memcmp(self.data, rhs.data, self.size) == 0

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    @always_inline
    fn __add__(self, rhs: Self) raises -> Self:
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self + rhs.data[0]
            if self.width == 1:
                return self.data[0] + rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[add](rhs)
            raise Error("Error: Cannot add matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self + rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[add](rhs)
            raise Error("Error: Cannot add matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self + rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[add](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot add matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[add](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot add matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[add](rhs)
            return self._elemwise_matrix[add](rhs.asorder(self.order))
        raise Error("Error: Cannot add matrices with different shapes!")

    @always_inline
    fn __iadd__(mut self, rhs: Self) raises:
        self = self + rhs

    @always_inline
    fn __add__(self, rhs: Float32) -> Self:
        return self._elemwise_scalar[add](rhs)

    @always_inline
    fn __radd__(self, lhs: Float32) -> Self:
        return self + lhs

    @always_inline
    fn __iadd__(mut self, rhs: Float32):
        self = self + rhs

    @always_inline
    fn __sub__(self, rhs: Self) raises -> Self:
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self - rhs.data[0]
            if self.width == 1:
                return self.data[0] - rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[sub](rhs)
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self - rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[sub](rhs)
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self - rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[sub](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[sub](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[sub](rhs)
            return self._elemwise_matrix[sub](rhs.asorder(self.order))
        raise Error("Error: Cannot subtract matrices with different shapes!")

    @always_inline
    fn __isub__(mut self, rhs: Self) raises:
        self = self - rhs

    @always_inline
    fn __sub__(self, rhs: Float32) -> Self:
        return self._elemwise_scalar[sub](rhs)

    @always_inline
    fn __rsub__(self, lhs: Float32) -> Self:
        return -(self - lhs)

    @always_inline
    fn __isub__(mut self, rhs: Float32):
        self = self - rhs

    @always_inline
    fn __truediv__(self, rhs: Self) raises -> Self:
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self / rhs.data[0]
            if self.width == 1:
                return self.data[0] / rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[div](rhs)
            raise Error("Error: Cannot divide matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self / rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[div](rhs)
            raise Error("Error: Cannot divide matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self / rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[div](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot divide matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[div](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot divide matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[div](rhs)
            return self._elemwise_matrix[div](rhs.asorder(self.order))
        raise Error("Error: Cannot divide matrices with different shapes!")

    @always_inline
    fn __itruediv__(mut self, rhs: Self) raises:
        self = self / rhs

    @always_inline
    fn __truediv__(self, rhs: Float32) -> Self:
        return self._elemwise_scalar[div](rhs)

    @always_inline
    fn __rtruediv__(self, lhs: Float32) -> Self:
        return lhs * (self ** -1)

    @always_inline
    fn __itruediv__(mut self, rhs: Float32):
        self = self / rhs

    @always_inline
    fn __mul__(self, rhs: Self) raises -> Self:
        if self.width != rhs.height:
            raise Error('Error: Cannot multiply matrices with shapes (' + String(self.height) + ', ' + String(self.width) + ') and (' + String(rhs.height) + ', ' + String(rhs.width) + ')')
        if self.height * self.width * rhs.width <= 4096:
            # matmul naive
            var mat = Self(self.height, rhs.width)
            for i in range(self.size):
                var rhsr = i % self.width
                var j_s = rhsr * rhs.width
                var j_e = rhsr * rhs.width + rhs.width
                for j in range(j_s, j_e):
                    if rhsr != 0:
                        mat.data[(Int(i / self.width) * mat.width) + (j % rhs.width)] += self.data[i] * rhs.data[j]
                    else:
                        mat.data[(Int(i / self.width) * mat.width) + (j % rhs.width)] = self.data[i] * rhs.data[j]
            return mat^
        var A = matmul.Matrix[DType.float32](self.data, (self.height, self.width))
        var B = matmul.Matrix[DType.float32](rhs.data, (rhs.height, rhs.width))
        var C = matmul.Matrix[DType.float32]((self.height, rhs.width))
        memset_zero(C.data, self.height * rhs.width)
        matmul.matmul(self.height, self.width, rhs.width, C, A, B)
        return Matrix(C.data, self.height, rhs.width)

    @always_inline
    fn __imul__(mut self, rhs: Self) raises:
        self = self * rhs

    @always_inline
    fn __mul__(self, rhs: Float32) -> Self:
        return self._elemwise_scalar[mul](rhs)

    @always_inline
    fn __rmul__(self, lhs: Float32) -> Self:
        return self * lhs

    @always_inline
    fn __imul__(mut self, rhs: Float32):
        self = self * rhs

    @always_inline
    fn __neg__(self) -> Self:
        return self * (-1.0)

    @always_inline
    fn __pow__(self, p: Int) -> Self:
        if p == 1:
            return self
        var mat = Self(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, pow(self.data.load[width=simd_width](idx), p))
            vectorize[math_vectorize, self.simd_width](self.size)
        else:
            var n_vects = Int(math.ceil(self.size / self.simd_width))
            @parameter
            fn math_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, pow(self.data.load[width=self.simd_width](idx), p))
            parallelize[math_vectorize_parallelize](n_vects)
        return mat^

    @always_inline
    fn __ipow__(mut self, rhs: Int):
        self = self ** rhs

    @always_inline
    fn ele_mul(self, rhs: Matrix) raises -> Matrix:
        # element-wise multiplication
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self * rhs.data[0]
            if self.width == 1:
                return self.data[0] * rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[mul](rhs)
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self * rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[mul](rhs)
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self * rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[mul](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[mul](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[mul](rhs)
            return self._elemwise_matrix[mul](rhs.asorder(self.order))
        raise Error("Error: Cannot element-wise multiply matrices with different shapes!")

    @always_inline
    fn where(self, cmp: List[Bool], _true: Float32, _false: Float32) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 40960:
            for i in range(self.size):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false
        else:
            @parameter
            fn p(i: Int):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false
            parallelize[p](self.size)
        return mat^

    fn where(self, cmp: List[Bool], _true: Matrix, _false: Float32) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 40960:
            for i in range(self.size):
                if cmp[i]:
                    mat.data[i] = _true.data[i]
                else:
                    mat.data[i] = _false
        else:
            @parameter
            fn p(i: Int):
                if cmp[i]:
                    mat.data[i] = _true.data[i]
                else:
                    mat.data[i] = _false
            parallelize[p](self.size)
        return mat^

    fn where(self, cmp: List[Bool], _true: Float32, _false: Matrix) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 40960:
            for i in range(self.size):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false.data[i]
        else:
            @parameter
            fn p(i: Int):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false.data[i]
            parallelize[p](self.size)
        return mat^

    @always_inline
    fn argwhere(self, cmp: List[Bool]) -> Matrix:
        var args = List[Float32]()
        for i in range(self.size):
            if cmp[i]:
                args.append(i // self.width)
                args.append(i % self.width)
        return Matrix(len(args) // 2, 2, args)

    @always_inline
    fn argwhere_l(self, cmp: List[Bool]) -> List[Int]:
        var args = List[Int]()
        for i in range(self.size):
            if cmp[i]:
                args.append(i)
        return args^

    @always_inline
    fn C_transpose(self) -> Matrix:
        var mat = Matrix(self.width, self.height)
        if self.size < 98304:
            for idx_col in range(self.width):
                var tmpPtr = self.data + idx_col
                @parameter
                fn convert[simd_width: Int](idx: Int):
                    mat.data.store(idx + idx_col * self.height, tmpPtr.strided_load[width=simd_width](self.width))
                    tmpPtr += simd_width * self.width
                vectorize[convert, self.simd_width](self.height)
        else:
            @parameter
            fn p(idx_col: Int):
                var tmpPtr = self.data + idx_col
                @parameter
                fn pconvert[simd_width: Int](idx: Int):
                    mat.data.store(idx + idx_col * self.height, tmpPtr.strided_load[width=simd_width](self.width))
                    tmpPtr += simd_width * self.width
                vectorize[pconvert, self.simd_width](self.height)
            parallelize[p](self.width)
        return mat^

    @always_inline
    fn F_transpose(self) -> Matrix:
        var mat = Matrix(self.width, self.height, order= self.order)
        if self.size < 98304:
            for idx_row in range(self.height):
                var tmpPtr = self.data + idx_row
                @parameter
                fn convert[simd_width: Int](idx: Int):
                    mat.data.store(idx + idx_row * self.width, tmpPtr.strided_load[width=simd_width](self.height))
                    tmpPtr += simd_width * self.height
                vectorize[convert, self.simd_width](self.width)
        else:
            @parameter
            fn p(idx_row: Int):
                var tmpPtr = self.data + idx_row
                @parameter
                fn pconvert[simd_width: Int](idx: Int):
                    mat.data.store(idx + idx_row * self.width, tmpPtr.strided_load[width=simd_width](self.height))
                    tmpPtr += simd_width * self.height
                vectorize[pconvert, self.simd_width](self.width)
            parallelize[p](self.height)
        return mat^
    
    @always_inline
    fn T(self) -> Matrix:
        if self.height == 1 or self.width == 1:
            return self.reshape(self.width, self.height)
        if self.order == 'c':
            return self.C_transpose()
        return self.F_transpose()

    fn asorder(self, order: String) -> Matrix:
        _order = order.lower()
        if _order == self.order:
            return self
        var mat = self.T().reshape(self.height, self.width)
        mat.order = _order
        return mat^

    @always_inline
    fn cumsum(self) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        algorithm.reduction.cumsum(NDBuffer[type=DType.float32, rank=1](mat.data, self.size), NDBuffer[type=DType.float32, rank=1](self.data, self.size))
        return mat^

    @always_inline
    fn sum(self) raises -> Float32:
        return algorithm.reduction.sum(NDBuffer[type=DType.float32, rank=1](self.data, self.size))

    @always_inline
    fn sum(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].sum()
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True].sum()
                    except:
                        print('Error: failed to find sum!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].sum()
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True].sum()
                    except:
                        print('Error: failed to find sum!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn mean(self) raises -> Float32:
        return self.sum() / self.size

    @always_inline
    fn mean(self, axis: Int) raises -> Matrix:
        if axis == 0:
            return self.sum(0) / self.height
        return self.sum(1) / self.width

    fn mean_slow(self) raises -> Float32:
        return (self / self.size).sum()

    fn mean_slow0(self) raises -> Matrix:
        var mat = Matrix(1, self.width, order= self.order)
        if self.width < 768:
            for i in range(self.width):
                mat.data[i] = self['', i, unsafe=True].mean_slow()
        else:
            @parameter
            fn p0(i: Int):
                try:
                    mat.data[i] = self['', i, unsafe=True].mean_slow()
                except:
                    print('Error: failed to find mean!')
            parallelize[p0](self.width)
        return mat^

    @always_inline
    fn _var(self, correction: Bool = False) raises -> Float32:
        return algorithm.reduction.variance(NDBuffer[type=DType.float32, rank=1](self.data, self.size), correction=correction)

    @always_inline
    fn _var(self, _mean: Float32, correction: Bool = False) raises -> Float32:
        return algorithm.reduction.variance(NDBuffer[type=DType.float32, rank=1](self.data, self.size), mean_value=_mean, correction=correction)

    @always_inline
    fn _var(self, axis: Int, correction: Bool = False) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True]._var(correction=correction)
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True]._var(correction=correction)
                    except:
                        print('Error: failed to find variance!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True]._var(correction=correction)
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True]._var(correction=correction)
                    except:
                        print('Error: failed to find variance!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn _var(self, axis: Int, _mean: Matrix, correction: Bool = False) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True]._var(_mean.data[i], correction=correction)
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True]._var(_mean.data[i], correction=correction)
                    except:
                        print('Error: failed to find variance!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True]._var(_mean.data[i], correction=correction)
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True]._var(_mean.data[i], correction=correction)
                    except:
                        print('Error: failed to find variance!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn std(self, correction: Bool = False) raises -> Float32:
        return math.sqrt(self._var(correction=correction))

    @always_inline
    fn std(self, _mean: Float32, correction: Bool = False) raises -> Float32:
        return math.sqrt(self._var(_mean, correction=correction))

    @always_inline
    fn std(self, axis: Int, correction: Bool = False) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].std(correction=correction)
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True].std(correction=correction)
                    except:
                        print('Error: failed to find std!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].std(correction=correction)
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True].std(correction=correction)
                    except:
                        print('Error: failed to find std!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn std(self, axis: Int, _mean: Matrix, correction: Bool = False) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].std(_mean.data[i], correction=correction)
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True].std(_mean.data[i], correction=correction)
                    except:
                        print('Error: failed to find std!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].std(_mean.data[i], correction=correction)
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True].std(_mean.data[i], correction=correction)
                    except:
                        print('Error: failed to find std!')
                parallelize[p1](self.height)
        return mat^

    fn std_slow(self, _mean: Float32) raises -> Float32:
        return math.sqrt(((self - _mean) ** 2).mean_slow())

    fn std_slow(self, axis: Int, _mean: Matrix) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].std_slow(_mean.data[i])
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True].std_slow(_mean.data[i])
                    except:
                        print('Error: failed to find std!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].std_slow(_mean.data[i])
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True].std_slow(_mean.data[i])
                    except:
                        print('Error: failed to find std!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn abs(self) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, abs(self.data.load[width=simd_width](idx)))
            vectorize[math_vectorize, self.simd_width](self.size)
        else:
            var n_vects = Int(math.ceil(self.size / self.simd_width))
            @parameter
            fn math_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, abs(self.data.load[width=self.simd_width](idx)))
            parallelize[math_vectorize_parallelize](n_vects)
        return mat^

    @always_inline
    fn log(self) -> Matrix:
        return self._elemwise_math[math.log]()

    @always_inline
    fn sqrt(self) -> Matrix:
        return self._elemwise_math[math.sqrt]()

    @always_inline
    fn exp(self) -> Matrix:
        return self._elemwise_math[math.exp]()

    @always_inline
    fn argmin(self) -> Int:
        var output = Matrix(1, 1)
        argn[False](self, output)
        var min_index = Int(output.data[0])
        if self.order == 'c':
            return min_index
        return (min_index % self.height) * self.width + min_index // self.height

    @always_inline
    fn argmin(self, axis: Int) -> List[Int]:
        var vect = UnsafePointer[Int]()
        var length = 0
        if axis == 0:
            vect = UnsafePointer[Int].alloc(self.width)
            length = self.width
            if self.width < 512:
                for i in range(self.width):
                    vect[i] = self['', i, unsafe=True].argmin()
            else:
                @parameter
                fn p0(i: Int):
                    vect[i] = self['', i, unsafe=True].argmin()
                parallelize[p0](self.width)
        elif axis == 1:
            vect = UnsafePointer[Int].alloc(self.height)
            length = self.height
            if self.height < 512:
                for i in range(self.height):
                    vect[i] = self[i, unsafe=True].argmin()
            else:
                @parameter
                fn p1(i: Int):
                    vect[i] = self[i, unsafe=True].argmin()
                parallelize[p1](self.height)
        var list = List[Int](unsafe_uninit_length=length)
        list.data=vect
        return list^

    @always_inline
    fn argmax(self) -> Int:
        var output = Matrix(1, 1)
        argn[True](self, output)
        var max_index = Int(output.data[0])
        if self.order == 'c':
            return max_index
        return (max_index % self.height) * self.width + max_index // self.height

    @always_inline
    fn argmax(self, axis: Int) -> List[Int]:
        var vect = UnsafePointer[Int]()
        var length = 0
        if axis == 0:
            vect = UnsafePointer[Int].alloc(self.width)
            length = self.width
            if self.width < 512:
                for i in range(self.width):
                    vect[i] = self['', i, unsafe=True].argmax()
            else:
                @parameter
                fn p0(i: Int):
                    vect[i] = self['', i, unsafe=True].argmax()
                parallelize[p0](self.width)
        elif axis == 1:
            vect = UnsafePointer[Int].alloc(self.height)
            length = self.height
            if self.height < 512:
                for i in range(self.height):
                    vect[i] = self[i, unsafe=True].argmax()
            else:
                @parameter
                fn p1(i: Int):
                    vect[i] = self[i, unsafe=True].argmax()
                parallelize[p1](self.height)
        var list = List[Int](unsafe_uninit_length=length)
        list.data=vect
        return list^

    @always_inline
    fn min(self) raises -> Float32:
        return algorithm.reduction.min(NDBuffer[type=DType.float32, rank=1](self.data, self.size))

    @always_inline
    fn min(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].min()
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True].min()
                    except:
                        print('Error: failed to find min!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].min()
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True].min()
                    except:
                        print('Error: failed to find min!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn max(self) raises -> Float32:
        return algorithm.reduction.max(NDBuffer[type=DType.float32, rank=1](self.data, self.size))

    @always_inline
    fn max(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].max()
            else:
                @parameter
                fn p0(i: Int):
                    try:
                        mat.data[i] = self['', i, unsafe=True].max()
                    except:
                        print('Error: failed to find max!')
                parallelize[p0](self.width)
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].max()
            else:
                @parameter
                fn p1(i: Int):
                    try:
                        mat.data[i] = self[i, unsafe=True].max()
                    except:
                        print('Error: failed to find max!')
                parallelize[p1](self.height)
        return mat^

    @always_inline
    fn reshape(self, height: Int, width: Int) -> Matrix:
        var mat: Matrix = self
        mat.height = height
        mat.width = width
        return mat^
    
    fn cov(self) raises -> Matrix:
        var c = Matrix(self.height, self.height, order=self.order)
        var mean_diff = self - self.mean(axis=1)
        @parameter
        fn p(i: Int):
            try:
                for j in range(self.height):
                    c[i, j] = cov_value(mean_diff[j], mean_diff[i])
            except:
                print('Error: failed to find cov!')
        parallelize[p](self.height)
        return c^

    @staticmethod
    @always_inline
    fn lu_factor(mut A: Matrix, piv: UnsafePointer[Int], N: Int) raises:
        for i in range(N):
            piv[i] = i

        for k in range(N - 1):
            var max_row = k
            for i in range(k + 1, N):
                if (abs(A[i, k]) > abs(A[max_row, k])):
                    max_row = i

            if k != max_row:
                swap(A[k], A[max_row])

                var temp = piv[k]
                piv[k] = piv[max_row]
                piv[max_row] = temp

            # LU decomposition (Gaussian elimination)
            for i in range(k + 1, N):
                A[i, k] /= A[k, k]
                A[i, True, k + 1] -= A[i, k] * A[k, True, k + 1]

    @staticmethod
    @always_inline
    fn lu_solve(A: Matrix, piv: UnsafePointer[Int], b: Matrix, mut x: Matrix, N: Int, Mi: Int) raises:
        var y = Matrix(1, N)

        # Forward substitution: solve L * y = P * b
        for i in range(N):
            y.data[i] = b[piv[i], Mi]
            for j in range(i):
                y.data[i] -= A[i, j] * y.data[j]

        # Backward substitution: solve U * x = y
        for i in range(N - 1, -1, -1):
            x[i, Mi] = y.data[i]
            for j in range(i + 1, N):
                x[i, Mi] -= A[i, j] * x[j, Mi]
            x[i, Mi] /= A[i, i]
    
    @staticmethod
    @always_inline
    fn solve(owned A: Matrix, b: Matrix) raises -> Matrix:
        if A.height != A.width:
            raise Error("Error: \"A\" must be square!")
        if A.width != b.height:
            raise Error("Error: \"B\" has an unrelated shape to \"A\"!")
        var N = A.height
        var M = b.width
        var X = Matrix(N, M, order=A.order)
        var piv = UnsafePointer[Int].alloc(N)

        Matrix.lu_factor(A, piv, N)
        if M > 1:
            @parameter
            fn p(i: Int):
                try:
                    Matrix.lu_solve(A, piv, b, X, N, i)
                except:
                    print('Error: failed to find LU solution!')
            parallelize[p](M)
        else:
            Matrix.lu_solve(A, piv, b, X, N, 0)

        piv.free()

        return X^

    fn inv(self) raises -> Matrix:
        if self.height != self.width:
            raise Error("Error: Matrix must be square to inverse!")
        return Matrix.solve(self, Matrix.eye(self.height, self.order))

    @staticmethod
    @always_inline
    fn eye(n: Int, order: String = 'c') -> Matrix:
        var result = Matrix.zeros(n, n, order)
        var tmpPtr = result.data
        @parameter
        fn convert[simd_width: Int](idx: Int):
            tmpPtr.strided_store[width=simd_width](1.0, (n + 1))
            tmpPtr += simd_width * (n + 1)
        vectorize[convert, result.simd_width](n)
        return result^

    @always_inline
    fn norm(self) raises -> Float32:
        return math.sqrt((self ** 2).sum())

    @always_inline
    fn qr(self, standard: Bool = False) raises -> Tuple[Matrix, Matrix]:
        # QR decomposition. standard: make R diag positive
        # Householder algorithm, i.e., not Gram-Schmidt or Givens
        # if not square, verify m greater-than n ("tall")
        # if standard==True verify m == n

        var Q = Matrix.eye(self.height, self.order)
        var R = self
        var end: Int
        if self.height == self.width:
            end = self.width - 1
        else:
            end = self.width
        for i in range(end):
            var H = Matrix.eye(self.height, self.order)
            # -------------------
            var a: Matrix = R[True, i, i]  # partial column vector
            var norm_a: Float32 = a.norm()
            if a.data[0] < 0.0: norm_a = -norm_a
            var v: Matrix = a / (a.data[0] + norm_a)
            v.data[0] = 1.0
            var h = Matrix.eye(a.height, self.order)  # H reflection
            h -= (2 / (v.T() * v))[0, 0] * (v * v.T())
            # -------------------
            for j in range(H.height - i):
                H[j + i, True, i] = h[j]  # copy h into H
            Q = Q * H
            R = H * R

        if standard:  # A must be square
            var S = Matrix.zeros(self.width, self.width, order= self.order)  # signs of diagonal
            for i in range(self.width):
                if R[i, i] < 0.0:
                    S[i, i] = -1.0
                else:
                    S[i, i] = 1.0
            Q = Q * S
            R = S * R

        return Q^, R^

    @always_inline
    fn is_upper_tri(self, tol: Float32 = 1.0e-8) raises -> Bool:
        for i in range(self.height):
            for j in range(i):
                if abs(self[i, j]) > tol:
                    return False
        return True

    fn svd(self, EPSILON: Float32 = 1.0e-08, full_matrices: Bool = True) raises -> Tuple[Matrix, Matrix, Matrix]:
        var A = self  # working copy U
        var m = A.height
        var n = A.width

        var Q = Matrix.eye(n, self.order)  # working copy V
        var t = Matrix.zeros(1, n, order=self.order)  # working copy s

        # init counters
        var count = 1
        var sweep = 0
        var sweep_max = max(5 * n, 12)  # heuristic

        var tolerance = 10 * m * EPSILON  # heuristic

        # store the column error estimates in t
        @parameter
        fn p(j: Int):
            try:
                t.data[j] = A['', j].norm()
            except:
                print('Error: Failed to find norm of columns!')
        parallelize[p](n)
        t *= EPSILON

        # orthogonalize A by plane rotations
        while (count > 0 and sweep <= sweep_max):
            # initialize rotation counter
            count = n * (n - 1) // 2
            for j in range(n-1):
                for k in range(j+1, n):
                    var cj = A['', j]
                    var ck = A['', k]
                    var p = 2 * cj.ele_mul(ck).sum()
                    var aj = cj.norm()
                    var ak = ck.norm()

                    # test for columns j,k orthogonal,
                    # or dominant errors 
                    var abserr_a = t.data[j]
                    var abserr_b = t.data[k]

                    var q = (aj * aj) - (ak * ak)
                    var v = math.sqrt(p**2 + q**2)  # hypot()
            
                    var sorted = (aj >= ak)
                    var orthog = (abs(p) <= tolerance * (aj*ak))
                    var noisya = (aj < abserr_a)
                    var noisyb = (ak < abserr_b)

                    if sorted and (orthog or \
                    noisya or noisyb):
                        count -= 1
                        continue

                    var cosine: Float32
                    var sine: Float32
                    # calculate rotation angles
                    if v == 0 or sorted == False:
                        cosine = 0.0
                        sine = 1.0
                    else:
                        cosine = math.sqrt((v + q) / (2.0 * v))
                        sine = p / (2.0 * v * cosine)

                    # apply rotation to A (U)
                    var Aik = A['', k]
                    var Aij = A['', j]
                    A['', j] = Aij * cosine + Aik * sine
                    A['', k] = -Aij * sine + Aik * cosine

                    # update singular values
                    t.data[j] = abs(cosine) * abserr_a + \
                    abs(sine) * abserr_b
                    t.data[k] = abs(sine) * abserr_a + \
                    abs(cosine) * abserr_b

                    # apply rotation to Q (V)
                    Qij = Q['', j]
                    Qik = Q['', k]
                    Q['', j] = Qij * cosine + Qik * sine
                    Q['', k] = -Qij * sine + Qik * cosine

            sweep += 1
        # while

        # compute singular values
        var prev_norm: Float32 = -1.0
        for j in range(n):
            var norm = A['', j].norm()
            # determine if singular value is zero
            if norm == 0.0 or prev_norm == 0.0 or \
            (j > 0 and norm <= tolerance * prev_norm):
                t.data[j] = 0.0
                A['', j].fill_zero()
                prev_norm = 0.0
            else:
                t.data[j] = norm
                A['', j] /= norm
                prev_norm = norm

        if count > 0:
            print("WARN: Jacobi iterations no converge!")

        # Trim near-zero singular values
        var nonzero = t.argwhere_l(t > EPSILON)
        U = A['', nonzero]
        s = t['', nonzero]

        if full_matrices:
            # Complete U to m x m
            # Complete Vh to n x n
            return complete_orthonormal_basis(U, m), s^, complete_orthonormal_basis(Q['', nonzero], n).T()
        return U^, s^, Q.T()[nonzero]

    fn eigvectors_from_eigvalues(self, eigenvalues: Matrix, tol: Float32) raises -> Matrix:
        var n = self.height
        eigvecs = Matrix.zeros(n, len(eigenvalues), order=self.order)

        for i in range(len(eigenvalues)):
            # Construct (A - lambda * I)
            var B = self - eigenvalues.data[i] * Matrix.eye(n, self.order)

            # Compute SVD
            _, S, Vh = B.svd()

            # Identify right-singular vectors corresponding to zero singular values
            var null_space = Vh[S.argwhere_l(S < tol)]

            # If no null vectors found, fallback to smallest singular vector
            if null_space.size == 0:
                vec = Vh[Vh.height - 1]
            else:
                vec = null_space[0]

            # Normalize and assign as eigenvector
            eigvecs['', i] = vec / vec.norm()

        return eigvecs^

    fn eigen(self, tol: Float32 = 1.0e-8, max_ct: Int = 10000) raises -> Tuple[Matrix, Matrix]:
        var X = self
        #var pq = Matrix.eye(self.height, self.order)

        var ct: Int = 0
        while ct < max_ct:
            var Q: Matrix
            var R: Matrix
            Q, R = X.qr()
            #pq = pq * Q  # accum Q
            X = R * Q
            ct += 1

            if X.is_upper_tri(tol):
                break

        if ct == max_ct:
            print("WARN (eigen): no converge!")

        # eigenvalues are diag elements of X
        var e_vals = Matrix.zeros(1, self.height, order= self.order)
        for i in range(self.height):
            e_vals.data[i] = X[i, i]

        # eigenvectors are columns of pq
        return e_vals^, self.eigvectors_from_eigvalues(e_vals, tol)

    fn outer(self, rhs: Matrix) raises -> Matrix:
        var mat = Matrix(self.size, rhs.size, order= self.order)
        if mat.order == 'c':
            @parameter
            fn p1(i: Int):
                try:
                    mat[i] = self.data[i] * rhs
                except:
                    print('Error: failed to find outer!')
            parallelize[p1](mat.height)
        else:
            @parameter
            fn p2(i: Int):
                try:
                    mat['', i] = self * rhs.data[i]
                except:
                    print('Error: failed to find outer!')
            parallelize[p2](mat.width)
        return mat^

    fn concatenate(self, rhs: Matrix, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(self.height + rhs.height, self.width, order= self.order)
            if self.order == 'c' or self.height == 1:
                memcpy(mat.data, self.data, self.size)
                memcpy(mat.data + self.size, rhs.data, rhs.size)
            else:
                @parameter
                fn pf(i: Int):
                    memcpy(mat.data + i * mat.height, self.data + i * self.height, self.height)
                    memcpy(mat.data + i * mat.height + self.height, rhs.data + i * rhs.height, rhs.height)
                parallelize[pf](self.width)
        elif axis == 1:
            mat = Matrix(self.height, self.width + rhs.width, order= self.order)
            if self.order == 'c' and self.width > 1:
                @parameter
                fn pc(i: Int):
                    memcpy(mat.data + i * mat.width, self.data + i * self.width, self.width)
                    memcpy(mat.data + i * mat.width + self.width, rhs.data + i * rhs.width, rhs.width)
                parallelize[pc](self.height)
            else:
                memcpy(mat.data, self.data, self.size)
                memcpy(mat.data + self.size, rhs.data, rhs.size)
        return mat^

    @always_inline
    fn bincount(self) raises -> List[Int]:
        var max_val = Int(self.max())
        var vect = UnsafePointer[Int].alloc(max_val + 1)
        memset_zero(vect, max_val + 1)

        for i in range(self.size):
            vect[Int(self.data[i])] += 1
        var list = List[Int](unsafe_uninit_length=max_val + 1)
        list.data=vect
        return list^

    @staticmethod
    @always_inline
    fn unique(data: PythonObject) raises -> Tuple[List[String], List[Int]]:
        var list = List[String]()
        var freq = List[Int]()
        var rng = len(data)
        for i in range(rng):
            var d = String(data[i])
            if d in list:
                freq[list.index(d)] += 1
            else:
                list.append(d)
                freq.append(1)
        return list^, freq^

    @always_inline
    fn unique(self) raises -> Dict[Int, Int]:
        var freq = Dict[Int, Int]()
        for i in range(self.size):
            var d = Int(self.data[i])
            if d in freq:
                freq[d] += 1
            else:
                freq[d] = 1
        return freq^

    @always_inline
    fn uniquef(self) -> Matrix:
        var list = List[String](capacity=self.size)
        list.resize(self.size, '')
        for i in range(self.size):
            var bytes = self.data[i].as_bytes().unsafe_ptr()
            list[i] = String(bytes=Span[UInt8, __origin_of(bytes)](ptr=bytes, length=DType.float32.sizeof()))
        var u_vals = Set(list)
        var result = Matrix(len(u_vals), 1, order=self.order)
        var index = 0
        for val in u_vals:
            var bytes = val[].as_bytes()
            result.data[index] = Float32.from_bytes(InlineArray[UInt8, DType.float32.sizeof()](bytes[0], bytes[1], bytes[2], bytes[3]))
            index += 1
        return result^

    @staticmethod
    @always_inline
    fn zeros(height: Int, width: Int, order: String = 'c') -> Matrix:
        var mat = Matrix(height, width, order= order)
        memset_zero(mat.data, mat.size)
        return mat^

    @staticmethod
    @always_inline
    fn ones(height: Int, width: Int, order: String = 'c') -> Matrix:
        return Matrix.full(height, width, 1.0, order)

    @staticmethod
    fn full(height: Int, width: Int, val: Float32, order: String = 'c') -> Matrix:
        var mat = Matrix(height, width, order= order)
        mat.fill(val)
        return mat^

    @always_inline
    fn fill_zero(self):
        memset_zero(self.data, self.size)

    @always_inline
    fn fill(self, val: Float32):
        NDBuffer[type=DType.float32, rank=1](self.data, self.size).fill(val)

    @staticmethod
    @always_inline
    fn random(height: Int, width: Int, order: String = 'c') -> Matrix:
        random.seed()
        var mat = Matrix(height, width, order= order)
        random.rand(mat.data, mat.size, min=0.0, max=1.0)
        return mat^

    @staticmethod
    @always_inline
    fn rand_choice(arang: Int, size: Int, replace: Bool = True) -> List[Int]:
        random.seed()
        var result = UnsafePointer[Int].alloc(size)
        if not replace:
            for i in range(size):
                result[i] = i
        for i in range(size - 1, -1, -1):
            if not replace:
                # Fisher-Yates shuffle
                var j = Int(random.random_ui64(0, i))
                result[i], result[j] = result[j], result[i]
            else:
                result[i] = Int(random.random_ui64(0, arang - 1))
        var list = List[Int](unsafe_uninit_length=size)
        list.data=result
        return list^

    @staticmethod
    @always_inline
    fn rand_choice(arang: Int, size: Int, replace: Bool, seed: Int) -> List[Int]:
        random.seed(seed)
        var result = UnsafePointer[Int].alloc(size)
        if not replace:
            for i in range(size):
                result[i] = i
        for i in range(size - 1, 0, -1):
            if not replace:
                # Fisher-Yates shuffle
                var j = Int(random.random_ui64(0, i))
                result[i], result[j] = result[j], result[i]
            else:
                result[i] = Int(random.random_ui64(0, arang - 1))
        var list = List[Int](unsafe_uninit_length=size)
        list.data=result
        return list^

    @staticmethod
    fn from_numpy(np_arr: PythonObject, order: String = 'c') raises -> Matrix:
        var np = Python.import_module("numpy")
        var np_arr_f = np.array(np_arr, dtype= 'f', order= order.upper())
        var height = Int(np_arr_f.shape[0])
        var width: Int
        try:
            width = Int(np_arr_f.shape[1])
        except:
            width = height
            height = 1
        var mat = Matrix(height, width, np_arr_f.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), order)
        _ = np_arr_f.__array_interface__['data'][0].__index__()
        return mat^

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.empty(Python.tuple(self.height,self.width), dtype='f', order= self.order.upper())
        memcpy(np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), self.data, self.size)
        return np_arr^

    @always_inline
    fn _broadcast_row(self, height: Int, width: Int, order: String) -> Matrix:
        var mat = Matrix(height, width, order=order)
        if height * width < 262144 and height < 1024:
            for i in range(mat.height):
                mat[i, unsafe=True] = self
        else:
            @parameter
            fn broadcast(i: Int):
                mat[i, unsafe=True] = self
            parallelize[broadcast](mat.height)
        return mat^

    @always_inline
    fn _broadcast_column(self, height: Int, width: Int, order: String) -> Matrix:
        var mat = Matrix(height, width, order=order)
        if height * width < 262144 and width < 1024:
            for i in range(mat.width):
                mat['', i, unsafe=True] = self
        else:
            @parameter
            fn broadcast(i: Int):
                mat['', i, unsafe=True] = self
            parallelize[broadcast](mat.width)
        return mat^

    @always_inline
    fn _elemwise_scalar[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, rhs: Float32) -> Self:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn scalar_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, func[DType.float32, simd_width](self.data.load[width=simd_width](idx), rhs))
            vectorize[scalar_vectorize, self.simd_width](self.size)
        else:
            var n_vects = Int(math.ceil(self.size / self.simd_width))
            @parameter
            fn scalar_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, func[DType.float32, self.simd_width](self.data.load[width=self.simd_width](idx), rhs))
            parallelize[scalar_vectorize_parallelize](n_vects)
        return mat^

    @always_inline
    fn _elemwise_matrix[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, rhs: Self) -> Self:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn matrix_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, func[DType.float32, simd_width](self.data.load[width=simd_width](idx), rhs.data.load[width=simd_width](idx)))
            vectorize[matrix_vectorize, self.simd_width](self.size)
        else:
            var n_vects = Int(math.ceil(self.size / self.simd_width))
            @parameter
            fn matrix_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, func[DType.float32, self.simd_width](self.data.load[width=self.simd_width](idx), rhs.data.load[width=self.simd_width](idx)))
            parallelize[matrix_vectorize_parallelize](n_vects)
        return mat^

    @always_inline
    fn _elemwise_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, func(self.data.load[width=simd_width](idx)))
            vectorize[math_vectorize, self.simd_width](self.size)
        else:
            var n_vects = Int(math.ceil(self.size / self.simd_width))
            @parameter
            fn math_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, func(self.data.load[width=self.simd_width](idx)))
            parallelize[math_vectorize_parallelize](n_vects)
        return mat^

    fn write_to[W: Writer](self, mut writer: W):
        var res: String = "["
        var strings = List[String]()
        for i in range(self.width):
            var max_len: Int = 0
            for j in range(self.height):
                strings.append("")
                var val = self.load[1](j, i)
                if val >= 0:
                    strings[j] += " "
                strings[j] += String(val)
                if len(strings[j]) > max_len:
                    max_len = len(strings[j])
            for j in range(self.height):
                var rng: Int = max_len - len(strings[j]) + 1
                for _ in range(rng):
                    strings[j] += " "

        for i in range(self.height):
            if i != 0:
                res += " "
            res += "[" + strings[i] + "]"
            if i != self.height - 1:
                res += "\n"
        writer.write(res + "]")

    fn __str__(self) -> String:
        return String.write(self)
