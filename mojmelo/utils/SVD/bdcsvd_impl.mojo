# ==============================================================================
# Based on libeigen — Eigen/src/SVD/BDCSVDImpl.h, Copyright (C) 2013 Gauthier Brun,
# Nicolas Carre, Jean Ceccato, Pierre Zoppitelli, Jitse Niesen, and Copyright (C) 2014-2017
# Gael Guennebaud, MPL-2.0.
# ------------------------------------------------------------------------------
# The divide-and-conquer phase of the bidiagonal SVD, following Gu & Eisenstat,
# "A Divide-and-Conquer Algorithm for the Bidiagonal SVD".
# ==============================================================================

from std.math import sqrt, hypot
from .linalg_core import (
    RealScalar,
    REAL_EPSILON,
    REAL_MIN,
    SQRT_REAL_MAX,
    swap_vecs,
    reverse_cols,
    ComputationInfo,
    INFO_SUCCESS,
    INFO_NUMERICAL_ISSUE,
    INFO_NO_CONVERGENCE,
    INFO_INVALID_INPUT,
    Vec,
    IVec,
    Mat,
    matmul
)
from .jacobi_svd import JacobiSVD
from .jacobi import JacobiRotation, apply_jacobi_on_right


struct BDCSVDImpl:
    var m_naiveU: Mat
    var m_naiveV: Mat
    var m_computed: Mat
    var m_workspace: Vec
    var m_workspaceI: IVec
    var m_baseSvdU: JacobiSVD  # compute_v = False
    var m_baseSvdUV: JacobiSVD  # compute_v = True
    var m_algoswap: Int
    var m_compU: Bool
    var m_compV: Bool
    var m_numIters: Int
    var m_info: ComputationInfo

    @always_inline
    def __init__(out self):
        self.m_naiveU = Mat(0, 0)
        self.m_naiveV = Mat(0, 0)
        self.m_computed = Mat(0, 0)
        self.m_workspace = Vec(0)
        self.m_workspaceI = IVec(0)
        self.m_baseSvdU = JacobiSVD(False)
        self.m_baseSvdUV = JacobiSVD(True)
        self.m_algoswap = 16
        self.m_compU = False
        self.m_compV = False
        self.m_numIters = 0
        self.m_info = INFO_SUCCESS

    @always_inline
    def algoSwap(self) -> Int:
        return self.m_algoswap

    @always_inline
    def setAlgoSwap(mut self, s: Int):
        self.m_algoswap = s

    @always_inline
    def info(self) -> ComputationInfo:
        return self.m_info

    @always_inline
    def numIters(self) -> Int:
        return self.m_numIters

    @always_inline
    def naiveU(mut self) -> Mat:
        return self.m_naiveU.block(0, 0, self.m_naiveU.rows(), self.m_naiveU.cols())

    @always_inline
    def naiveV(mut self) -> Mat:
        return self.m_naiveV.block(0, 0, self.m_naiveV.rows(), self.m_naiveV.cols())

    @always_inline
    def computed(mut self) -> Mat:
        return self.m_computed.block(0, 0, self.m_computed.rows(), self.m_computed.cols())

    def allocate(mut self, diagSize: Int, compU: Bool, compV: Bool):
        self.m_compU = compU
        self.m_compV = compV
        self.m_numIters = 0
        self.m_info = INFO_SUCCESS

        self.m_computed = Mat.zeros(diagSize + 1, diagSize)

        if self.m_compU:
            self.m_naiveU = Mat.zeros(diagSize + 1, diagSize + 1)
        else:
            self.m_naiveU = Mat.zeros(2, diagSize + 1)

        if self.m_compV:
            self.m_naiveV = Mat.zeros(diagSize, diagSize)

        # With U/V requested, `structured_update` needs three matrix-sized
        # packing buffers; values-only needs five vectors: diag, shifts,
        # mus, zhat, diagShifted.
        if self.m_compU or self.m_compV:
            self.m_workspace = Vec((diagSize + 1) * (diagSize + 1) * 3)
        else:
            self.m_workspace = Vec(5 * diagSize)
        self.m_workspaceI = IVec(3 * diagSize)

    # --------------------------------------------------------------------
    # LAPACK's xBDSDC normalizes the bidiagonal by its largest entry and
    # splits wherever a superdiagonal entry falls below eps, so a run of
    # rounding noise never becomes its own sub-problem. This mirrors that:
    # scale by the sub-block's own max magnitude, and zero anything below
    # 0.45 * eps * ||B||_max (0.45 == 0.9 * half-unit-roundoff, matching
    # xBDSDC's 0.9 * DLAMCH('E') threshold).
    @always_inline
    def splitNegligibleSuperdiagonal(mut self, n: Int):
        if n < 2:
            return
        var diag_view = self.m_computed.block(0, 0, n, n).diagonal(0)
        var sub_view = self.m_computed.block(0, 0, n, n).diagonal(-1)
        var norm = max(diag_view.cwiseAbsMax(), sub_view.cwiseAbsMax())
        var threshold = RealScalar(0.45) * REAL_EPSILON * norm
        for i in range(n - 1):
            if abs(self.m_computed[i + 1, i]) < threshold:
                self.m_computed[i + 1, i] = RealScalar(0)

    # --------------------------------------------------------------------
    # A = A * B. Eigen packs the mostly-zero rows of A before the multiply
    # to save flops when A is large; NOT ported here.
    @always_inline
    def structured_update(mut self, mut A: Mat, B: Mat, n1: Int):
        var result = matmul(A, B)
        A.copyFrom(result)

    @always_inline
    def computeBaseCase[V: Bool](
        mut self,
        n: Int,
        firstCol: Int,
        firstRowW: Int,
        firstColW: Int,
        shift: Int,
    ):
        comptime if V:
            self.m_baseSvdUV.compute(self.m_computed.block(firstCol, firstCol, n + 1, n))
            self.m_info = self.m_baseSvdUV.info()
            if self.m_info != INFO_SUCCESS and self.m_info != INFO_NO_CONVERGENCE:
                return
            if self.m_compU:
                var dst = self.m_naiveU.block(firstCol, firstCol, n + 1, n + 1)
                dst.copyFrom(self.m_baseSvdUV.matrixU())
            else:
                var uu = self.m_baseSvdUV.matrixU()
                var row0 = self.m_naiveU.row(0).segment(firstCol, n + 1)
                row0.copyFrom(uu.row(0))
                var row1 = self.m_naiveU.row(1).segment(firstCol, n + 1)
                row1.copyFrom(uu.row(n))
            if self.m_compV:
                var dstv = self.m_naiveV.block(firstRowW, firstColW, n, n)
                dstv.copyFrom(self.m_baseSvdUV.matrixV())
            var view = self.m_computed.block(firstCol + shift, firstCol + shift, n + 1, n)
            view.setZero()
            var dst_diag = self.m_computed.diagonal().segment(firstCol + shift, n)
            dst_diag.copyFrom(self.m_baseSvdUV.singularValues().head(n))
        else:
            self.m_baseSvdU.compute(self.m_computed.block(firstCol, firstCol, n + 1, n))
            self.m_info = self.m_baseSvdU.info()
            if self.m_info != INFO_SUCCESS and self.m_info != INFO_NO_CONVERGENCE:
                return
            if self.m_compU:
                var dst = self.m_naiveU.block(firstCol, firstCol, n + 1, n + 1)
                dst.copyFrom(self.m_baseSvdU.matrixU())
            else:
                var uu = self.m_baseSvdU.matrixU()
                var row0 = self.m_naiveU.row(0).segment(firstCol, n + 1)
                row0.copyFrom(uu.row(0))
                var row1 = self.m_naiveU.row(1).segment(firstCol, n + 1)
                row1.copyFrom(uu.row(n))
            if self.m_compV:
                var dstv = self.m_naiveV.block(firstRowW, firstColW, n, n)
                dstv.copyFrom(self.m_baseSvdU.matrixV())
            var view = self.m_computed.block(firstCol + shift, firstCol + shift, n + 1, n)
            view.setZero()
            var dst_diag = self.m_computed.diagonal().segment(firstCol + shift, n)
            dst_diag.copyFrom(self.m_baseSvdU.singularValues().head(n))

    # --------------------------------------------------------------------
    # The recursive divide-and-conquer driver. Operates in place on
    # sub-blocks of m_computed / m_naiveU / m_naiveV, exactly as in the
    # original: firstCol/lastCol locate the current sub-problem, firstRowW/
    # firstColW locate the corresponding block of V, and shift accumulates
    # by one every time we recurse into a *left* sub-problem (because the
    # merge step promotes the left block's last U-column to be its first).
    def divide(mut self, firstCol: Int, lastCol: Int, firstRowW: Int, firstColW: Int, shift: Int):
        # requires rows == cols + 1
        var n = lastCol - firstCol + 1
        var k = n // 2
        var considerZero = REAL_MIN
        var alphaK: RealScalar
        var betaK: RealScalar
        var r0: RealScalar
        var lambda_: RealScalar
        var phi: RealScalar
        var c0: RealScalar
        var s0: RealScalar

        if n < self.m_algoswap:
            if self.m_compV:
                self.computeBaseCase[V=True](n, firstCol, firstRowW, firstColW, shift)
            else:
                self.computeBaseCase[V=False](n, firstCol, firstRowW, firstColW, shift)
            return

        alphaK = self.m_computed[firstCol + k, firstCol + k]
        betaK = self.m_computed[firstCol + k + 1, firstCol + k]

        # Order matters: the right sub-problem must be solved before the
        # left one, because dividing the left sub-problem reads a column
        # that the right sub-problem's divide() call writes.
        self.divide(k + 1 + firstCol, lastCol, k + 1 + firstRowW, k + 1 + firstColW, shift)
        if self.m_info != INFO_SUCCESS and self.m_info != INFO_NO_CONVERGENCE:
            return
        self.divide(firstCol, k - 1 + firstCol, firstRowW, firstColW + 1, shift + 1)
        if self.m_info != INFO_SUCCESS and self.m_info != INFO_NO_CONVERGENCE:
            return

        if self.m_compU:
            lambda_ = self.m_naiveU[firstCol + k, firstCol + k]
            phi = self.m_naiveU[firstCol + k + 1, lastCol + 1]
        else:
            lambda_ = self.m_naiveU[1, firstCol + k]
            phi = self.m_naiveU[0, lastCol + 1]

        r0 = hypot(alphaK * lambda_, betaK * phi)
        if self.m_compV:
            self.m_naiveV[firstRowW + k, firstColW] = RealScalar(1)
        if r0 < considerZero:
            c0 = RealScalar(1)
            s0 = RealScalar(0)
        else:
            c0 = alphaK * lambda_ / r0
            s0 = betaK * phi / r0

        self.m_computed[firstCol + shift, firstCol + shift] = r0
        if self.m_compU:
            var col_top = self.m_computed.col(firstCol + shift).segment(firstCol + shift + 1, k)
            var src_top = self.m_naiveU.row(firstCol + k).segment(firstCol, k)
            for i in range(k):
                col_top[i] = alphaK * src_top[i]
            var col_bot = self.m_computed.col(firstCol + shift).segment(firstCol + shift + k + 1, n - k - 1)
            var src_bot = self.m_naiveU.row(firstCol + k + 1).segment(firstCol + k + 1, n - k - 1)
            for i in range(n - k - 1):
                col_bot[i] = betaK * src_bot[i]
        else:
            var col_top = self.m_computed.col(firstCol + shift).segment(firstCol + shift + 1, k)
            var src_top = self.m_naiveU.row(1).segment(firstCol, k)
            for i in range(k):
                col_top[i] = alphaK * src_top[i]
            var col_bot = self.m_computed.col(firstCol + shift).segment(firstCol + shift + k + 1, n - k - 1)
            var src_bot = self.m_naiveU.row(0).segment(firstCol + k + 1, n - k - 1)
            for i in range(n - k - 1):
                col_bot[i] = betaK * src_bot[i]

        if self.m_compU:
            var q1 = self.m_workspace.segment(0, k + 1)
            var src_q1 = self.m_naiveU.col(firstCol + k).segment(firstCol, k + 1)
            q1.copyFrom(src_q1)
            # shift Q1 to the right
            var i = firstCol + k - 1
            while i >= firstCol:
                var src = self.m_naiveU.col(i).segment(firstCol, k + 1)
                var dst = self.m_naiveU.col(i + 1).segment(firstCol, k + 1)
                dst.copyFrom(src)
                i -= 1
            # shift q1 to the left, scaled by c0
            var dst0 = self.m_naiveU.col(firstCol).segment(firstCol, k + 1)
            for j in range(k + 1):
                dst0[j] = q1[j] * c0
            # last column = q1 * -s0
            var dstLast = self.m_naiveU.col(lastCol + 1).segment(firstCol, k + 1)
            for j in range(k + 1):
                dstLast[j] = q1[j] * (-s0)
            # first column (lower part) = q2 * s0
            var q2 = self.m_naiveU.col(lastCol + 1).segment(firstCol + k + 1, n - k)
            var dstFirstLower = self.m_naiveU.col(firstCol).segment(firstCol + k + 1, n - k)
            for j in range(n - k):
                dstFirstLower[j] = q2[j] * s0
            # q2 *= c0
            var q2m = self.m_naiveU.col(lastCol + 1).segment(firstCol + k + 1, n - k)
            for j in range(n - k):
                q2m[j] = q2m[j] * c0
        else:
            var q1 = self.m_naiveU[0, firstCol + k]
            var i = firstCol + k - 1
            while i >= firstCol:
                self.m_naiveU[0, i + 1] = self.m_naiveU[0, i]
                i -= 1
            self.m_naiveU[0, firstCol] = q1 * c0
            self.m_naiveU[0, lastCol + 1] = q1 * (-s0)
            self.m_naiveU[1, firstCol] = self.m_naiveU[1, lastCol + 1] * s0
            self.m_naiveU[1, lastCol + 1] = self.m_naiveU[1, lastCol + 1] * c0
            for j in range(firstCol + 1, firstCol + 1 + k):
                self.m_naiveU[1, j] = RealScalar(0)
            for j in range(firstCol + k + 1, firstCol + k + 1 + (n - k - 1)):
                self.m_naiveU[0, j] = RealScalar(0)

        # Second part: deflate singular values in the merged problem.
        self.deflation(firstCol, lastCol, k, firstRowW, firstColW, shift)

        # Third part: solve the SVD of the merged (arrowhead) matrix.
        var UofSVD = Mat(0, 0)
        var VofSVD = Mat(0, 0)
        var singVals = Vec(0)
        self.computeSVDofM(firstCol + shift, n, UofSVD, singVals, VofSVD)

        if self.m_compU:
            var target = self.m_naiveU.block(firstCol, firstCol, n + 1, n + 1)
            self.structured_update(target, UofSVD, (n + 2) // 2)
        else:
            var block = self.m_naiveU.block(0, firstCol, 2, n + 1)
            var result = matmul(block, UofSVD)
            block.copyFrom(result)

        if self.m_compV:
            var targetV = self.m_naiveV.block(firstRowW, firstColW, n, n)
            self.structured_update(targetV, VofSVD, (n + 1) // 2)

        # The recursive children leave this block diagonal outside of the
        # single column this merge step touches; only that column needs
        # clearing before the new (post-secular-equation) diagonal is
        # written in.
        var clear_col = self.m_computed.col(firstCol + shift).segment(firstCol + shift, n)
        clear_col.setZero()
        var dst_diag = self.m_computed.diagonal().segment(firstCol + shift, n)
        dst_diag.copyFrom(singVals)

    # --------------------------------------------------------------------
    # SVD of the n+1 x n arrowhead matrix m_computed.block(firstCol,
    # firstCol, n+1, n): nonzero only in the first column and on the
    # diagonal, already deflated so the diagonal is increasing except
    # possibly the (0,0) entry. Fills U / singVals / (V if m_compV).
    # Singular values come back sorted in decreasing order (callers reverse
    # to increasing).
    @always_inline
    def computeSVDofM(
        mut self, firstCol: Int, n: Int, mut U: Mat, mut singVals: Vec, mut V: Mat
    ):
        var considerZero = REAL_MIN
        var col0 = self.m_computed.col(firstCol).segment(firstCol, n)
        var diag = self.m_workspace.segment(0, n)
        var diag_src = self.m_computed.block(firstCol, firstCol, n, n).diagonal(0)
        diag.copyFrom(diag_src)
        diag[0] = RealScalar(0)

        singVals = Vec(n)
        U = Mat(n + 1, n + 1)
        if self.m_compV:
            V = Mat(n, n)

        # Deflated singular values were moved to the end and are zeroed on
        # the diagonal; skip them when building the active permutation.
        var actual_n = n
        while actual_n > 1 and diag[actual_n - 1] == RealScalar(0):
            actual_n -= 1

        var m = 0
        for k in range(actual_n):
            if abs(col0[k]) > considerZero:
                self.m_workspaceI[m] = k
                m += 1
        var perm = self.m_workspaceI.segment(0, m)

        var shifts = self.m_workspace.segment(1 * n, n)
        var mus = self.m_workspace.segment(2 * n, n)
        var zhat = self.m_workspace.segment(3 * n, n)

        self.computeSingVals(col0, diag, perm, singVals, shifts, mus)
        self.perturbCol0(col0, diag, perm, singVals, shifts, mus, zhat)
        self.computeSingVecs(zhat, diag, perm, singVals, shifts, mus, U, V)

        # Deflation can leave the singular values almost, but not quite,
        # sorted; an O(n) adjacent-swap pass fixes that.
        for i in range(actual_n - 1):
            if singVals[i] > singVals[i + 1]:
                swap(singVals[i], singVals[i+1])
                U.swap_cols(i, i + 1)
                if self.m_compV:
                    V.swap_cols(i, i + 1)

        # Flip to increasing order; deflated zeros are already trailing.
        var sv_head = singVals.head(actual_n)
        sv_head.reverseInPlace()
        reverse_cols(U, actual_n)
        if self.m_compV:
            reverse_cols(V, actual_n)

    def computeSingVals(
        mut self,
        col0: Vec,
        diag: Vec,
        perm: IVec,
        mut singVals: Vec,
        mut shifts: Vec,
        mut mus: Vec,
    ):
        var n = len(col0)
        var actual_n = n
        # Uses col0(i)==0 rather than diag(i)==0: diag(i)==0 implies
        # col0(i)==0, and col0(i)==0 alone already means diag(i) is a
        # singular value on its own.
        while actual_n > 1 and col0[actual_n - 1] == RealScalar(0):
            actual_n -= 1

        for k in range(n):
            if col0[k] == RealScalar(0) or actual_n == 1:
                singVals[k] = col0[0] if k == 0 else diag[k]
                mus[k] = RealScalar(0)
                shifts[k] = col0[0] if k == 0 else diag[k]
                continue

            var left = diag[k]
            var right: RealScalar
            if k == actual_n - 1:
                right = diag[actual_n - 1] + col0.norm()
            else:
                var l = k + 1
                while col0[l] == RealScalar(0):
                    l += 1
                right = diag[l]

            var mid = left + (right - left) / RealScalar(2)
            var fMid = secularEq(mid, col0, diag, perm, diag, RealScalar(0))
            var shift = left if (k == actual_n - 1 or fMid > RealScalar(0)) else right

            var diagShifted = self.m_workspace.segment(4 * len(col0), len(col0))
            for i in range(len(diagShifted)):
                diagShifted[i] = diag[i] - shift

            if k != actual_n - 1:
                var midShifted = (right - left) / RealScalar(2)
                if shift == right:
                    midShifted = -midShifted
                var fMidShifted = secularEq(midShifted, col0, diag, perm, diagShifted, shift)
                if fMidShifted > RealScalar(0):
                    shift = left if fMidShifted > RealScalar(0) else right
                    for i in range(len(diagShifted)):
                        diagShifted[i] = diag[i] - shift

            var muPrev: RealScalar
            var muCur: RealScalar
            if shift == left:
                muPrev = (right - left) * RealScalar(0.1)
                if k == actual_n - 1:
                    muCur = right - left
                else:
                    muCur = (right - left) * RealScalar(0.5)
            else:
                muPrev = -(right - left) * RealScalar(0.1)
                muCur = -(right - left) * RealScalar(0.5)

            var fPrev = secularEq(muPrev, col0, diag, perm, diagShifted, shift)
            var fCur = secularEq(muCur, col0, diag, perm, diagShifted, shift)
            if abs(fPrev) < abs(fCur):
                swap(fPrev, fCur)
                swap(muPrev, muCur)

            # Rational interpolation a/mu + b through the two most recent
            # samples; its zero is the next iterate.
            var useBisection = fPrev * fCur > RealScalar(0)
            while (
                fCur != RealScalar(0)
                and abs(muCur - muPrev)
                > RealScalar(8) * REAL_EPSILON * max(abs(muCur), abs(muPrev))
                and abs(fCur - fPrev) > REAL_EPSILON
                and not useBisection
            ):
                self.m_numIters += 1
                var a = (fCur - fPrev) / (RealScalar(1) / muCur - RealScalar(1) / muPrev)
                var b = fCur - a / muCur
                var muZero = -a / b
                var fZero = secularEq(muZero, col0, diag, perm, diagShifted, shift)

                muPrev = muCur
                fPrev = fCur
                muCur = muZero
                fCur = fZero

                if shift == left and (muCur < RealScalar(0) or muCur > right - left):
                    useBisection = True
                if shift == right and (muCur < -(right - left) or muCur > RealScalar(0)):
                    useBisection = True
                if abs(fCur) > abs(fPrev):
                    useBisection = True

            if useBisection:
                var leftShifted: RealScalar
                var rightShifted: RealScalar
                if shift == left:
                    leftShifted = max(
                        REAL_MIN, RealScalar(2) * abs(col0[k]) / SQRT_REAL_MAX
                    )
                    if k == actual_n - 1:
                        rightShifted = right
                    else:
                        rightShifted = (right - left) * RealScalar(0.51)
                else:
                    leftShifted = -(right - left) * RealScalar(0.51)
                    if k + 1 < n:
                        rightShifted = -max(
                            REAL_MIN, abs(col0[k + 1]) / SQRT_REAL_MAX
                        )
                    else:
                        rightShifted = -REAL_MIN

                var fLeft = secularEq(leftShifted, col0, diag, perm, diagShifted, shift)

                if fLeft < RealScalar(0):
                    while rightShifted - leftShifted > RealScalar(2) * REAL_EPSILON * max(
                        abs(leftShifted), abs(rightShifted)
                    ):
                        var midShifted2 = (leftShifted + rightShifted) / RealScalar(2)
                        fMid = secularEq(midShifted2, col0, diag, perm, diagShifted, shift)
                        if fLeft * fMid < RealScalar(0):
                            rightShifted = midShifted2
                        else:
                            leftShifted = midShifted2
                            fLeft = fMid
                    muCur = (leftShifted + rightShifted) / RealScalar(2)
                else:
                    # Both ends of [left, right] disagree in sign after
                    # shifting; rather than looping forever, fall back to
                    # the midpoint as the best available estimate.
                    muCur = (right - left) * RealScalar(0.5)
                    if shift == right:
                        muCur = -muCur

            singVals[k] = shift + muCur
            shifts[k] = shift
            mus[k] = muCur

    # --------------------------------------------------------------------
    # zhat: the perturbation of col0 that lets singular vectors be computed
    # stably (Gu & Eisenstat section 3.1 / LAPACK's xLASD8).
    @always_inline
    def perturbCol0(
        mut self,
        col0: Vec,
        diag: Vec,
        perm: IVec,
        singVals: Vec,
        shifts: Vec,
        mus: Vec,
        mut zhat: Vec,
    ):
        var n = len(col0)
        var m = len(perm)
        if m == 0:
            zhat.setZero()
            return
        var lastIdx = perm[m - 1]
        for k in range(n):
            if col0[k] == RealScalar(0):
                zhat[k] = RealScalar(0)
            else:
                var dk = diag[k]
                # Materialize (shift - dk) before adding the mu correction:
                # under FP reassociation, `mus + (shift - dk)` could turn
                # into `(mus + shift) - dk`, losing `mus` when shift and dk
                # nearly cancel.
                var diff = shifts[lastIdx] - dk
                var prod = (singVals[lastIdx] + dk) * (mus[lastIdx] + diff)

                for l in range(m):
                    var i = perm[l]
                    if i != k:
                        if i >= k and l == 0:
                            # No valid predecessor to the left of k: flag as
                            # a numerical issue and zero this term, matching
                            # the original's defensive fallback.
                            self.m_info = INFO_NUMERICAL_ISSUE
                            prod = RealScalar(0)
                            break
                        var j = i if i < k else perm[l - 1]
                        diff = shifts[j] - dk
                        prod = prod * productOfQuotients(
                            singVals[j] + dk, diag[i] + dk, mus[j] + diff, diag[i] - dk
                        )
                # Exact arithmetic keeps `prod` nonnegative; as in LAPACK's
                # xLASD8, take abs before sqrt to absorb rounding noise
                # that can push it slightly negative.
                var tmp = sqrt(abs(prod))
                zhat[k] = tmp if col0[k] > RealScalar(0) else -tmp

    def computeSingVecs(
        mut self,
        zhat: Vec,
        diag: Vec,
        perm: IVec,
        singVals: Vec,
        shifts: Vec,
        mus: Vec,
        mut U: Mat,
        mut V: Mat,
    ):
        var n = len(zhat)
        var m = len(perm)

        for k in range(n):
            if zhat[k] == RealScalar(0):
                var uk = U.col(k)
                var unit_u = Vec.Unit(n + 1, k)
                uk.copyFrom(unit_u)
                if self.m_compV:
                    var vk = V.col(k)
                    var unit_v = Vec.Unit(n, k)
                    vk.copyFrom(unit_v)
            else:
                var ucol = U.col(k)
                ucol.setZero()
                if self.m_compV:
                    var vcol = V.col(k)
                    vcol.setZero()
                for l in range(m):
                    var i = perm[l]
                    var diff = diag[i] - shifts[k]
                    diff = diff - mus[k]
                    # Matches Eigen's sequentialQuotient(num, d1, d2) =
                    # (num / d1) / d2, kept as two separate divisions
                    # (rather than one combined denominator) for the same
                    # underflow-avoidance reason as productOfQuotients.
                    U[i, k] = sequentialQuotient(zhat[i], diff, diag[i] + singVals[k])
                    if self.m_compV and l > 0:
                        V[i, k] = sequentialQuotient(diag[i] * zhat[i], diff, diag[i] + singVals[k])
                U[n, k] = RealScalar(0)
                var ucol2 = U.col(k)
                ucol2.stableNormalize()

                if self.m_compV:
                    V[0, k] = RealScalar(-1)
                    var vcol2 = V.col(k)
                    vcol2.stableNormalize()
        var last_u = U.col(n)
        var unit_last = Vec.Unit(n + 1, n)
        last_u.copyFrom(unit_last)

    # --------------------------------------------------------------------
    # i >= 1, d_i ~ 0 and z_i != 0: rotate to zero out z_i, set d_i = 0.
    @always_inline
    def deflation43(mut self, firstCol: Int, shift: Int, i: Int, size: Int):
        var start = firstCol + shift
        var c = self.m_computed[start, start]
        var s = self.m_computed[start + i, start]
        var r = hypot(c, s)
        if r == RealScalar(0):
            self.m_computed[start + i, start + i] = RealScalar(0)
            return
        self.m_computed[start, start] = r
        self.m_computed[start + i, start] = RealScalar(0)
        self.m_computed[start + i, start + i] = RealScalar(0)

        var rot = JacobiRotation(c / r, -s / r)
        if self.m_compU:
            apply_jacobi_on_right(self.m_naiveU, firstCol, size + 1, firstCol, firstCol + i, rot)
        else:
            apply_jacobi_on_right(self.m_naiveU, 0, self.m_naiveU.rows(), firstCol, firstCol + i, rot)

    # --------------------------------------------------------------------
    # i, j >= 1, i > j, |d_i - d_j| < eps * ||M||_2: two rotations make
    # z_i == 0 and d_j == d_i.
    @always_inline
    def deflation44(
        mut self,
        firstColu: Int,
        firstColm: Int,
        firstRowW: Int,
        firstColW: Int,
        i: Int,
        j: Int,
        size: Int,
    ):
        var s = self.m_computed[firstColm + i, firstColm]
        var c = self.m_computed[firstColm + j, firstColm]
        var r = hypot(c, s)
        if r == RealScalar(0):
            self.m_computed[firstColm + j, firstColm + j] = self.m_computed[firstColm + i, firstColm + i]
            return
        c = c / r
        s = s / r
        self.m_computed[firstColm + j, firstColm] = r
        self.m_computed[firstColm + j, firstColm + j] = self.m_computed[firstColm + i, firstColm + i]
        self.m_computed[firstColm + i, firstColm] = RealScalar(0)

        var rot = JacobiRotation(c, -s)
        if self.m_compU:
            apply_jacobi_on_right(self.m_naiveU, firstColu, size + 1, firstColu + j, firstColu + i, rot)
        else:
            apply_jacobi_on_right(self.m_naiveU, 0, self.m_naiveU.rows(), firstColu + j, firstColu + i, rot)
        if self.m_compV:
            apply_jacobi_on_right(self.m_naiveV, firstRowW, size, firstColW + j, firstColW + i, rot)

    # --------------------------------------------------------------------
    # Acts on the block from (firstCol+shift, firstCol+shift) to
    # (lastCol+shift, lastCol+shift) inclusive.
    @always_inline
    def deflation(
        mut self, firstCol: Int, lastCol: Int, k: Int, firstRowW: Int, firstColW: Int, shift: Int
    ):
        var length = lastCol + 1 - firstCol
        var col0 = self.m_computed.col(firstCol + shift).segment(firstCol + shift, length)
        var diag = self.m_computed.diagonal().segment(firstCol + shift, length)

        var considerZero = REAL_MIN
        var tailLen = length - 1 if length - 1 > 1 else 1
        var maxDiag = diag.tail(tailLen).cwiseAbsMax()
        var epsilon_strict = max(considerZero, REAL_EPSILON * maxDiag)
        var epsilon_coarse = RealScalar(8) * REAL_EPSILON * max(col0.cwiseAbsMax(), maxDiag)

        # condition 4.1
        if diag[0] < epsilon_coarse:
            diag[0] = epsilon_coarse

        # condition 4.2
        for i in range(1, length):
            if abs(col0[i]) < epsilon_strict:
                col0[i] = RealScalar(0)

        # condition 4.3
        for i in range(1, length):
            if diag[i] < epsilon_coarse:
                self.deflation43(firstCol, shift, i, length)

        # --- total-deflation check + sorted merge permutation ---
        var total_deflation = True
        for i in range(1, length):
            if abs(col0[i]) >= considerZero:
                total_deflation = False
                break

        var permutation = self.m_workspaceI.segment(0, length)
        permutation[0] = 0
        var p = 1
        for i in range(1, length):
            if diag[i] < considerZero:
                permutation[p] = i
                p += 1
        var ii = 1
        var jj = k + 1
        while p < length:
            if ii > k:
                permutation[p] = jj
                jj += 1
            elif jj >= length:
                permutation[p] = ii
                ii += 1
            elif diag[ii] < diag[jj]:
                permutation[p] = jj
                jj += 1
            else:
                permutation[p] = ii
                ii += 1
            p += 1

        if total_deflation:
            for i in range(1, length):
                var pi = permutation[i]
                if diag[pi] < considerZero or diag[0] < diag[pi]:
                    permutation[i - 1] = permutation[i]
                else:
                    permutation[i - 1] = 0
                    break

        var realInd = self.m_workspaceI.segment(length, length)
        var realCol = self.m_workspaceI.segment(2 * length, length)
        for pos in range(length):
            realCol[pos] = pos
            realInd[pos] = pos

        var start_i = 0 if total_deflation else 1
        for i in range(start_i, length):
            var idx = length - (i + 1) if total_deflation else length - i
            var pi = permutation[idx]
            var J = realCol[pi]

            swap(diag[i], diag[J])
            if i != 0 and J != 0:
                swap(col0[i], col0[J])

            if self.m_compU:
                var a = self.m_naiveU.col(firstCol + i).segment(firstCol, length + 1)
                var b = self.m_naiveU.col(firstCol + J).segment(firstCol, length + 1)
                swap_vecs(a, b)
            else:
                var a = self.m_naiveU.col(firstCol + i).segment(0, 2)
                var b = self.m_naiveU.col(firstCol + J).segment(0, 2)
                swap_vecs(a, b)
            if self.m_compV:
                var a = self.m_naiveV.col(firstColW + i).segment(firstRowW, length)
                var b = self.m_naiveV.col(firstColW + J).segment(firstRowW, length)
                swap_vecs(a, b)

            var realI = realInd[i]
            realCol[realI] = J
            realCol[pi] = i
            realInd[J] = realI
            realInd[i] = pi

        # condition 4.4
        var i2 = length - 1
        while i2 > 0 and (diag[i2] < considerZero or abs(col0[i2]) < considerZero):
            i2 -= 1
        while i2 > 1:
            if (diag[i2] - diag[i2 - 1]) < epsilon_coarse:
                self.deflation44(firstCol, firstCol + shift, firstRowW, firstColW, i2, i2 - 1, length)
            i2 -= 1

# ------------------------------------------------------------------------------
# secularEq / productOfQuotients / sequentialQuotient.
#
# In the original these are static members with no access to `self`; kept as
# free functions here for the same reason.
# ------------------------------------------------------------------------------
@always_inline
def productOfQuotients(
    firstNumerator: RealScalar,
    firstDenominator: RealScalar,
    secondNumerator: RealScalar,
    secondDenominator: RealScalar,
) -> RealScalar:
    # Keep the two divisions separate rather than combining denominators
    # first: the combined denominator can underflow even when the final
    # product is representable.
    var q1 = firstNumerator / firstDenominator
    var q2 = secondNumerator / secondDenominator
    return q1 * q2

@always_inline
def sequentialQuotient(
    numerator: RealScalar, firstDenominator: RealScalar, secondDenominator: RealScalar
) -> RealScalar:
    var q1 = numerator / firstDenominator
    return q1 / secondDenominator

@always_inline
def secularEq(
    mu: RealScalar, col0: Vec, diag: Vec, perm: IVec, diagShifted: Vec, shift: RealScalar
) -> RealScalar:
    var m = len(perm)
    var res = RealScalar(1)
    for i in range(m):
        var j = perm[i]
        res += productOfQuotients(col0[j], diagShifted[j] - mu, col0[j], diag[j] + shift + mu)
    return res
