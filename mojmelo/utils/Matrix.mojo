import memory
from collections.vector import InlinedFixedVector
import math
from mojmelo.utils.utils import cov_value, gauss_jordan
from python import Python
import time

struct Matrix:
    var height: Int
    var width: Int
    var size: Int
    var data: Pointer[Float32]

    fn __init__(inout self, height: Int, width: Int, data: Pointer[Float32]):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = Pointer[Float32].alloc(self.size)
        memcpy(self.data, data, self.size)

    fn __init__(inout self, height: Int, width: Int, def_input: List[Float32] = List[Float32]()):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = Pointer[Float32].alloc(self.size)
        if len(def_input) > 0:
            memcpy(self.data, Pointer[Float32](address = def_input.data.address), self.size)

    fn __init__(inout self, height: Int, width: Int, def_input: object) raises:
        self.height = height
        self.width = width
        self.size = height * width
        self.data = Pointer[Float32].alloc(self.size)
        var rng: Int = len(def_input)
        for i in range(rng):
            self.data[i] = atof(str(def_input[i]))

    fn __init__(inout self, npstyle: String) raises:
        var mat = npstyle.replace(' ', '')
        if mat[0] == '[' and mat[1] == '[' and mat[len(mat) - 1] == ']' and mat[len(mat) - 2] == ']':
            self.width = 0
            self.size = 0
            self.data = Pointer[Float32]()
            var rows = mat[:-1].split(']')
            self.height = len(rows) - 1
            for i in range(self.height):
                var values = rows[i][2:].split(',')
                if i == 0:
                    self.width = len(values)
                    self.size = self.height * self.width
                    self.data = Pointer[Float32].alloc(self.size)
                for j in range(self.width):
                    self.data[i * self.width + j] = atof(values[j])
        else:
            raise Error('Error: Matrix is not initialized in the correct form!')
            
    fn __copyinit__(inout self, other: Self):
        self.height = other.height
        self.width = other.width
        self.size = other.size
        self.data = Pointer[Float32].alloc(self.size)
        memcpy(self.data, other.data, self.size)

    fn __moveinit__(inout self, owned existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.size = existing.size
        self.data = existing.data
        existing.height = existing.width = existing.size = 0
        existing.data = Pointer[Float32]()

    fn __getitem__(self, row: Int, column: Int) raises -> Float32:
        var loc: Int = (row * self.width) + column
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        return self.data[loc]

    fn __getitem__(self, row: Int) raises -> Matrix:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        var mat = Matrix(1, self.width)
        var r: Int = row * self.width
        for i in range(mat.width):
            mat.data[i] = self.data[r + i]
        return mat^

    fn __getitem__(self, row: Int, offset: Bool, start_i: Int) raises -> Matrix:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        var mat = Matrix(1, self.width - start_i)
        var r: Int = row * self.width + start_i
        for i in range(mat.width):
            mat.data[i] = self.data[r + i]
        return mat^

    fn __getitem__(self, row: String, column: Int) raises -> Matrix:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        var mat = Matrix(self.height, 1)
        for i in range(mat.height):
            mat.data[i] = self.data[i * self.width + column]
        return mat^

    fn __getitem__(self, offset: Bool, start_i: Int, column: Int) raises -> Matrix:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        var mat = Matrix(self.height - start_i, 1)
        for i in range(mat.height):
            mat.data[i] = self.data[(i + start_i) * self.width + column]
        return mat^

    fn __getitem__(self, rows: Matrix) raises -> Matrix:
        var mat = Matrix(rows.size, self.width)
        for i in range(rows.size):
            mat[i] = self[rows[i]]
        return mat^

    fn __getitem__(self, row: String, columns: Matrix) raises -> Matrix:
        var mat = Matrix(self.height, columns.size)
        for i in range(columns.size):
            mat[row, i] = self[row, columns[i]]
        return mat^

    fn __getitem__(self, rows: List[Int]) raises -> Matrix:
        var mat = Matrix(len(rows), self.width)
        for i in range(mat.height):
            mat[i] = self[rows[i]]
        return mat^

    fn __getitem__(self, row: String, columns: List[Int]) raises -> Matrix:
        var mat = Matrix(self.height, len(columns))
        for i in range(mat.width):
            mat[row, i] = self[row, columns[i]]
        return mat^
            
    fn __setitem__(inout self, row: Int, column: Int, val: Float32) raises:
        var loc: Int = (row * self.width) + column
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        self.data[loc] = val

    fn __setitem__(inout self, row: Int, val: Matrix) raises:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        memcpy(self.data + (row * self.width), val.data, self.width)

    fn __setitem__(inout self, row: Int, offset: Bool, start_i: Int, val: Matrix) raises:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        memcpy(self.data + (row * self.width) + start_i, val.data, self.width - start_i)

    fn __setitem__(inout self, row: String, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        for i in range(self.height):
            self.data[i * self.width + column] = val.data[i]

    fn __setitem__(inout self, offset: Bool, start_i: Int, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        for i in range(self.height - start_i):
            self.data[(i + start_i) * self.width + column] = val.data[i]

    fn __setitem__(inout self, rows: Matrix, rhs: Matrix) raises:
        for i in range(rows.size):
            self[rows[i]] = rhs[i]

    fn __setitem__(inout self, row: String, columns: Matrix, rhs: Matrix) raises:
        for i in range(columns.size):
            self[row, columns[i]] = rhs[row, i]

    fn __del__(owned self):
        if self.data:
            self.data.free()

    fn __len__(self) -> Int:
        return self.size

    fn __eq__(self, rhs: Self) -> Bool:
        if self.height == rhs.height and self.width == rhs.width and memcmp(self.data, rhs.data, self.size) == 0:
            return True
        return False

    fn __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    fn __eq__(self, rhs: Float32) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        for i in range(self.size):
            if self.data[i] == rhs:
                result[i] = True
            else:
                result[i] = False
        return result^

    fn __ne__(self, rhs: Float32) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        for i in range(self.size):
            if self.data[i] != rhs:
                result[i] = True
            else:
                result[i] = False
        return result^

    fn __gt__(self, rhs: Float32) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        for i in range(self.size):
            if self.data[i] > rhs:
                result[i] = True
            else:
                result[i] = False
        return result^

    fn __ge__(self, rhs: Float32) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        for i in range(self.size):
            if self.data[i] == rhs:
                result[i] = True
            else:
                result[i] = False
        return result^

    fn __lt__(self, rhs: Float32) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        for i in range(self.size):
            if self.data[i] < rhs:
                result[i] = True
            else:
                result[i] = False
        return result^

    fn __le__(self, rhs: Float32) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        for i in range(self.size):
            if self.data[i] <= rhs:
                result[i] = True
            else:
                result[i] = False
        return result^

    fn __add__(self, rhs: Self) raises -> Self:
        if self.height != rhs.height or self.width != rhs.width:
            raise Error("Error: Cannot add matrices with different shapes!")
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] + rhs.data[i]
        return mat^

    fn __iadd__(inout self, rhs: Self) raises:
        self = self + rhs

    fn __add__(self, rhs: Float32) -> Self:
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] + rhs
        return mat^

    fn __radd__(self, lhs: Float32) -> Self:
        return self + lhs

    fn __iadd__(inout self, rhs: Float32):
        self = self + rhs

    fn __sub__(self, rhs: Self) raises -> Self:
        if self.height != rhs.height or self.width != rhs.width:
            raise Error("Error: Cannot subtract matrices with different shapes!")
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] - rhs.data[i]
        return mat^

    fn __isub__(inout self, rhs: Self) raises:
        self = self - rhs

    fn __sub__(self, rhs: Float32) -> Self:
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] - rhs
        return mat^

    fn __rsub__(self, lhs: Float32) -> Self:
        return -(self - lhs)

    fn __isub__(inout self, rhs: Float32):
        self = self - rhs

    fn __truediv__(self, rhs: Self) raises -> Self:
        if self.height != rhs.height or self.width != rhs.width:
            raise Error("Error: Cannot divide matrices with different shapes!")
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] / rhs.data[i]
        return mat^

    fn __itruediv__(inout self, rhs: Self) raises:
        self = self / rhs

    fn __truediv__(self, rhs: Float32) -> Self:
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] / rhs
        return mat^

    fn __rtruediv__(self, lhs: Float32) -> Self:
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = lhs / self.data[i]
        return mat^

    fn __itruediv__(inout self, rhs: Float32):
        self = self / rhs

    fn __mul__(self, rhs: Self) raises -> Self:
        if self.width != rhs.height:
            raise Error("Error: Cannot multiply matrices with unrelated shapes!")
        var mat = Self(self.height, rhs.width)
        for i in range(self.size):
            var rhsr: Int = i % self.width
            var j_s: Int = rhsr * rhs.width
            var j_e: Int = rhsr * rhs.width + rhs.width
            for j in range(j_s, j_e):
                if rhsr != 0:
                    mat.data[(int(i / self.width) * mat.width) + (j % rhs.width)] += self.data[i] * rhs.data[j]
                else:
                    mat.data[(int(i / self.width) * mat.width) + (j % rhs.width)] = self.data[i] * rhs.data[j]
        return mat^

    fn __imul__(inout self, rhs: Self) raises:
        self = self * rhs

    fn __mul__(self, rhs: Float32) -> Self:
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] * rhs
        return mat^

    fn __rmul__(self, lhs: Float32) -> Self:
        return self * lhs

    fn __imul__(inout self, rhs: Float32):
        self = self * rhs

    fn __neg__(self) -> Self:
        return self * (-1.0)
    
    fn __pow__(self, rhs: Int) -> Self:
        if rhs == 1:
            return self
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = self.data[i] ** rhs
        return mat^

    fn __ipow__(inout self, rhs: Int):
        self = self ** rhs

    fn ele_mul(self, rhs: Self) -> Self:
        var mat = Self(0, 0)
        if self.height == rhs.height:
            var width = max(self.width, rhs.width)
            mat = Self(self.height, width)
            for i in range(mat.size):
                mat.data[i] = self.data[int(i / width) * self.width + (i % self.width)] * rhs.data[int(i / width) * rhs.width + (i % rhs.width)]
        elif self.width == rhs.width:
            mat = Self(max(self.height, rhs.height), self.width)
            for i in range(mat.size):
                mat.data[i] = self.data[(int(i / self.width) % self.height) * self.width + (i % self.width)] * rhs.data[(int(i / rhs.width) % rhs.height) * rhs.width + (i % rhs.width)]
        return mat^

    fn where(self, cmp: InlinedFixedVector[Bool], _true: Float32, _false: Float32) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            if cmp[i]:
                mat.data[i] = _true
            else:
                mat.data[i] = _false
        return mat^

    fn where(self, cmp: InlinedFixedVector[Bool], _true: Matrix, _false: Float32) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            if cmp[i]:
                mat.data[i] = _true.data[i]
            else:
                mat.data[i] = _false
        return mat^

    fn where(self, cmp: InlinedFixedVector[Bool], _true: Float32, _false: Matrix) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            if cmp[i]:
                mat.data[i] = _true
            else:
                mat.data[i] = _false.data[i]
        return mat^

    fn where(self, cmp: InlinedFixedVector[Bool], _true: Matrix, _false: Matrix) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            if cmp[i]:
                mat.data[i] = _true.data[i]
            else:
                mat.data[i] = _false.data[i]
        return mat^

    fn argwhere(self, cmp: InlinedFixedVector[Bool]) -> Matrix:
        var args = List[Float32]()
        for i in range(self.size):
            if cmp[i]:
                args.append(int(i / self.width))
                args.append(i % self.width)
        return Matrix(int(len(args) / 2), 2, args)

    fn argwhere_l(self, cmp: InlinedFixedVector[Bool]) -> List[Int]:
        var args = List[Int]()
        for i in range(self.size):
            if cmp[i]:
                args.append(i)
        return args^
    
    fn T(self) -> Matrix:
        var mat = Matrix(self.width, self.height)
        for i in range(self.size):
            mat.data[(i % self.width) * self.height + int(i / self.width)] = self.data[i]
        return mat^

    fn inv(self) raises -> Matrix:
        if self.height != self.width:
            raise Error("Error: Matrix must be square to inverse!")
        var tmp = gauss_jordan(self.concatenate(Matrix.eye(self.height), 1))
        var mat = Matrix(self.height, self.height)
        for i in range(tmp.height):
            mat[i] = tmp[i, True, tmp[i].size//2]
        return mat^

    fn sum(self) -> Float32:
        var res: Float32 = 0.0
        for i in range(self.size):
            res += self.data[i]
        return res

    fn sum(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].sum()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].sum()
        return mat^

    fn mean(self) -> Float32:
        return self.sum() / self.size

    fn mean(self, axis: Int) raises -> Matrix:
        if axis == 0:
            return self.sum(0) / self.height
        elif axis == 1:
            return self.sum(1) / self.width
        raise Error("Error: Wrong axis value is given!")

    fn mean_slow(self) raises -> Float32:
        return (self / self.size).sum()

    fn mean_slow0(self) raises -> Matrix:
        var mat = Matrix(1, self.width)
        for i in range(self.width):
            mat.data[i] = self['', i].mean_slow()
        return mat^

    fn _var(self) -> Float32:
        return ((self - self.mean()) ** 2).mean()

    fn _var(self, _mean: Float32) -> Float32:
        return ((self - _mean) ** 2).mean()

    fn _var(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i]._var()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i]._var()
        return mat^

    fn _var(self, axis: Int, _mean: Matrix) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i]._var(_mean.data[i])
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i]._var(_mean.data[i])
        return mat^

    fn std(self) -> Float32:
        return math.sqrt(self._var())

    fn std(self, _mean: Float32) raises -> Float32:
        return math.sqrt(self._var(_mean))

    fn std_slow(self, _mean: Float32) raises -> Float32:
        return math.sqrt(((self - _mean) ** 2).mean_slow())

    fn std(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].std()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].std()
        return mat^

    fn std(self, axis: Int, _mean: Matrix) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].std(_mean.data[i])
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].std(_mean.data[i])
        return mat^

    fn std_slow(self, axis: Int, _mean: Matrix) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].std_slow(_mean.data[i])
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].std_slow(_mean.data[i])
        return mat^

    fn abs(self) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = abs(self.data[i])
        return mat^

    fn log(self) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = math.log(self.data[i])
        return mat^

    fn sqrt(self) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = math.sqrt(self.data[i])
        return mat^

    fn exp(self) -> Matrix:
        var mat = Matrix(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = math.exp(self.data[i])
        return mat^

    fn argmin(self) -> Int:
        var i_min: Int = 0
        for i in range(1, self.size):
            if self.data[i] < self.data[i_min]:
                i_min = i
        return i_min

    fn argmin(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].argmin()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].argmin()
        return mat^

    fn argmax(self) -> Int:
        var i_max: Int = 0
        for i in range(1, self.size):
            if self.data[i] > self.data[i_max]:
                i_max = i
        return i_max

    fn argmax(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].argmax()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].argmax()
        return mat^

    fn min(self) -> Float32:
        return self.data[self.argmin()]

    fn min(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].min()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].min()
        return mat^

    fn max(self) -> Float32:
        return self.data[self.argmax()]

    fn max(self, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width)
            for i in range(self.width):
                mat.data[i] = self['', i].max()
        elif axis == 1:
            mat = Matrix(self.height, 1)
            for i in range(self.height):
                mat.data[i] = self[i].max()
        return mat^

    fn reshape(self, height: Int, width: Int) -> Matrix:
        var mat: Matrix = self
        mat.height = height
        mat.width = width
        return mat^
        
    fn cov(self) raises -> Matrix:
        var c = Matrix(self.height, self.height)
        for i in range(self.height):
            for j in range(self.height):
                c.data[(i * c.width) + j] = cov_value(self[j], self[i])
        return c^

    @staticmethod
    fn eye(n: Int) -> Matrix:
        var result = Matrix.zeros(n, n)
        for i in range(n):
            result.data[i * n + i] = 1.0
        return result^

    fn norm(self) -> Float32:
        var sum: Float32 = 0.0
        for i in range(self.size):
            sum += self.data[i] ** 2
        return math.sqrt(sum)

    fn qr(self, standard: Bool = False) raises -> Tuple[Matrix, Matrix]:
        # QR decomposition. standard: make R diag positive
        # Householder algorithm, i.e., not Gram-Schmidt or Givens
        # if not square, verify m greater-than n ("tall")
        # if standard==True verify m == n

        var Q = Matrix.eye(self.height)
        var R = self
        var end: Int
        if self.height == self.width:
            end = self.width - 1
        else:
            end = self.width
        for i in range(end):
            var H = Matrix.eye(self.height)
            # -------------------
            var a: Matrix = R[True, i, i]  # partial column vector
            var norm_a: Float32 = a.norm()
            if a.data[0] < 0.0: norm_a = -norm_a
            var v: Matrix = a / (a.data[0] + norm_a)
            v.data[0] = 1.0
            var h = Matrix.eye(a.height)  # H reflection
            h -= (2 / (v.T() * v))[0, 0] * (v * v.T())
            # -------------------
            for j in range(H.height - i):
                H[j + i, True, i] = h[j]  # copy h into H
            Q = Q * H
            R = H * R

        if standard == True:  # A must be square
            var S = Matrix.zeros(self.width, self.width)  # signs of diagonal
            for i in range(self.width):
                if R.data[i * R.width + i] < 0.0:
                    S.data[i * S.width + i] = -1.0
                else:
                    S.data[i * S.width + i] = 1.0
            Q = Q * S
            R = S * R

        return Q^, R^

    fn is_upper_tri(self, tol: Float32) -> Bool:
        for i in range(self.height):
            for j in range(i):
                if abs(self.data[(i * self.width) + j]) > tol:
                    return False
        return True

    fn eigen(self, max_ct: Int = 10000) raises -> Tuple[Matrix, Matrix]:
        var X = self
        var pq = Matrix.eye(self.height)

        var ct: Int = 0
        while ct < max_ct:
            var Q: Matrix
            var R: Matrix
            Q, R = X.qr()
            pq = pq * Q  # accum Q
            X = R * Q  # note order
            ct += 1

            if X.is_upper_tri(1.0e-8) == True:
                break

        if ct == max_ct:
            print("WARN (eigen): no converge!")

        # eigenvalues are diag elements of X
        var e_vals = Matrix.zeros(1, self.height)
        for i in range(self.size):
            e_vals.data[i] = X.data[i * X.width + i]

        # eigenvectors are columns of pq
        var e_vecs = pq

        return e_vals^, e_vecs^

    fn outer(self, rhs: Matrix) -> Matrix:
        var mat = Matrix(self.size, rhs.size)
        for i in range(mat.height):
            for j in range(mat.width):
                mat.data[(i * mat.width) + j] = self.data[i] * rhs.data[j]
        return mat^

    fn concatenate(self, rhs: Matrix, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(self.height + rhs.height, self.width)
            memcpy(mat.data, self.data, self.size)
            memcpy(mat.data + self.size, rhs.data, rhs.size)
        elif axis == 1:
            mat = Matrix(self.height, self.width + rhs.width)
            for i in range(self.height):
                memcpy(mat.data + i * mat.width, self.data + i * self.width, self.width)
                memcpy(mat.data + i * mat.width + self.width, rhs.data + i * rhs.width, rhs.width)
        return mat^
    
    fn bincount(self) -> InlinedFixedVector[Int]:
        var freq = Dict[Int, Int]()
        var max_val: Int = 0
        for i in range(self.size):
            var d = int(self.data[i])
            try:
                freq[d] += 1
            except:
                freq[d] = 1
                if d > max_val:
                    max_val = d

        var vect = InlinedFixedVector[Int](capacity = max_val + 1)
        for i in range(max_val + 1):
            try:
                vect[i] = freq[i]
            except:
                vect[i] = 0
        return vect^

    @staticmethod
    fn unique(data: PythonObject) raises -> Tuple[List[String], List[Int]]:
        var list = List[String]()
        var freq = List[Int]()
        var rng = len(data)
        for i in range(rng):
            var d = str(data[i])
            if d in list:
                freq[list.index(d)] += 1
            else:
                list.append(d)
                freq.append(1)
        return list^, freq^

    fn unique(self) -> Dict[Int, Int]:
        var freq = Dict[Int, Int]()
        for i in range(self.size):
            var d = int(self.data[i])
            try:
                freq[d] += 1
            except:
                freq[d] = 1
        return freq^

    fn uniquef(self) -> List[Float32]:
        var list = List[Float32]()
        for i in range(self.size):
            var contains = False
            for j in list:
                if j[] == self.data[i]:
                    contains = True
            if not contains:
                list.append(self.data[i])
        return list^

    @staticmethod
    fn zeros(height: Int, width: Int) -> Matrix:
        var mat = Matrix(height, width)
        memset_zero(mat.data, mat.size)
        return mat^

    @staticmethod
    fn ones(height: Int, width: Int) -> Matrix:
        var mat = Matrix(height, width)
        for i in range(mat.size):
            mat.data[i] = 1.0
        return mat^

    @staticmethod
    fn full(height: Int, width: Int, val: Float32) -> Matrix:
        var mat = Matrix(height, width)
        for i in range(mat.size):
            mat.data[i] = val
        return mat^

    @staticmethod
    fn random(height: Int, width: Int) -> Matrix:
        random.seed()
        var mat = Matrix(height, width)
        random.rand(mat.data, mat.size)
        return mat^

    @staticmethod
    fn rand_choice(arang: Int, size: Int, replace: Bool = True) -> List[Int]:
        random.seed()
        var cache = Matrix(0, 0)
        if not replace:
            cache = Matrix.zeros(1, arang)
        var result = List[Int](capacity = size)
        for _ in range(size):
            var rand_int = int(random.random_ui64(0, arang - 1))
            if not replace:
                while cache.data[rand_int] == 1.0:
                    rand_int = int(random.random_ui64(0, arang - 1))
                cache.data[rand_int] = 1.0
            result.append(rand_int)
        return result^

    @staticmethod
    fn rand_choice(arang: Int, size: Int, replace: Bool, seed: Int) -> List[Int]:
        random.seed(seed)
        var cache = Matrix(0, 0)
        if not replace:
            cache = Matrix.zeros(1, arang)
        var result = List[Int](capacity = size)
        for _ in range(size):
            var rand_int = int(random.random_ui64(0, arang - 1))
            if not replace:
                while cache.data[rand_int] == 1.0:
                    rand_int = int(random.random_ui64(0, arang - 1))
                cache.data[rand_int] = 1.0
            result.append(rand_int)
        return result^

    @staticmethod
    fn from_numpy(np_arr: PythonObject) raises -> Matrix:
        var np_arr_f = np_arr.astype('f')
        var height = int(np_arr_f.shape[0])
        var width = 0
        try:
            width = int(np_arr_f.shape[1])
        except:
            width = height
            height = 1
        var mat = Self(height, width, Pointer[Float32](address = (np_arr_f.__array_interface__['data'][0]).__index__()))
        _ = (np_arr_f.__array_interface__['data'][0]).__index__()
        return mat^

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.empty((self.height,self.width), dtype='f')
        memcpy(Pointer[Float32](address = (np_arr.__array_interface__['data'][0]).__index__()), self.data, self.size)
        return np_arr^

    fn apply_fun[func: fn(Float32) -> Float32](self) -> Self:
        var mat = Self(self.height, self.width)
        for i in range(self.size):
            mat.data[i] = func(self.data[i])
        return mat^

    fn __str__(self) -> String:
        var res: String = "["
        var strings = List[String]()
        for i in range(self.width):
            var max_len: Int = 0
            for j in range(self.height):
                strings.append("")
                var val: Float32 = self.data[(j * self.width) + i]
                if val >= 0:
                    strings[j] += " "
                strings[j] += str(val)
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
        return res + "]"
