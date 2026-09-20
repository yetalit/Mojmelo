# ==============================================================================
# Based on libeigen — Eigen/src/SVD/BDCSVD.h, Copyright (C) 2013 Gauthier Brun,
# Nicolas Carre, Jean Ceccato, Pierre Zoppitelli, Jitse Niesen, and
# Copyright (C) 2014-2017 Gael Guennebaud, MPL-2.0.
# ==============================================================================

from .linalg_core import (
    RealScalar,
    REAL_MIN,
    REAL_EPSILON,
    Vec,
    Mat,
    ComputationInfo,
    INFO_SUCCESS,
    INFO_NO_CONVERGENCE,
    INFO_INVALID_INPUT,
    mat_transpose,
    mat_scale,
    mat_identity,
    embed_topleft,
)
from .jacobi_svd import JacobiSVD
from .bdcsvd_impl import BDCSVDImpl
from .bidiagonalization import (
    UpperBidiagonalization,
    make_householder_in_place,
    apply_householder_left,
    apply_compact_wy_block,
)

def prepare_work(A: Mat, do_transpose: Bool) -> Mat:
    """Returns a fresh Mat in both branches (transpose or plain copy) —
    avoids assigning Mat by copy.
    """
    if do_transpose:
        return mat_transpose(A)
    return A.copy()

# ==============================================================================
# Storage convention matches Eigen's HouseholderQR: m_qr's upper triangle
# (including diagonal) holds R itself untouched; m_qr's strict lower holds
# each reflector's essential vector; taus live in the separate m_hCoeffs
# vector rather than being aliased into the matrix.
# ==============================================================================
struct HouseholderQR:
    var m_qr: Mat
    var m_hCoeffs: Vec
    var m_rows: Int
    var m_cols: Int  # number of reflectors == min(original rows, cols)

    @always_inline
    def __init__(out self):
        self.m_qr = Mat(0, 0)
        self.m_hCoeffs = Vec(0)
        self.m_rows = 0
        self.m_cols = 0

    def compute(mut self, A: Mat):
        var rows = A.rows()
        var cols = A.cols()
        self.m_qr = A.copy()
        self.m_rows = rows
        var size = min(rows, cols)
        self.m_cols = size
        self.m_hCoeffs = Vec(size)
        if size == 0:
            return

        comptime block_size = 64
        var k = 0
        while k < size:
            var nb = min(block_size, size - k)
            # Panel: unblocked generation, but each reflector's trailing
            # update is confined to the panel's own nb columns (cheap:
            # O(rows * nb) per column instead of O(rows * cols)) — this
            # is what breaks the generate-then-update dependency into
            # something small enough to keep doing the rank-1 way.
            var V = Mat(rows - k, nb)
            var taus = Vec(nb)
            for jj in range(nb):
                var kk = k + jj
                var remainingRows = rows - kk
                var panelRemainingCols = (k + nb) - kk - 1

                var col_tail = self.m_qr.col(kk).segment(kk, remainingRows)
                var tb = make_householder_in_place(col_tail)
                var tau = tb[0]
                var beta = tb[1]
                self.m_hCoeffs[kk] = tau
                self.m_qr[kk, kk] = beta
                taus[jj] = tau
                V[jj, jj] = RealScalar(1)

                if tau != RealScalar(0):
                    var essential = col_tail.segment(1, remainingRows - 1)
                    V.col(jj).segment(jj + 1, len(essential)).copyFrom(essential)
                    if panelRemainingCols > 0:
                        var sub = self.m_qr.block(kk, kk + 1, remainingRows, panelRemainingCols)
                        apply_householder_left(sub, essential, tau)

            # Flush the whole panel onto the wide trailing block (columns
            # to the right of the panel) in one shot: 2 GEMMs instead of
            # nb more rank-1 updates. Needs Q_panel^T here, not Q_panel:
            # sequential generation applies H_0 first, so the net effect
            # already imposed on later columns is H_{nb-1}*...*H_1*H_0,
            # the transpose of the H_0*...*H_{nb-1} that apply_q_on_left
            # wants when reconstructing Q elsewhere.
            var afterCols = cols - (k + nb)
            if afterCols > 0:
                var trailing = self.m_qr.block(k, k + nb, rows - k, afterCols)
                apply_compact_wy_block(trailing, V, taus, True)

            k += nb

    @always_inline
    def matrixR(self) -> Mat:
        """The cols x cols (== m_cols x m_cols) upper-triangular R factor,
        as a fresh dense copy — matches Eigen's
        `qrDecomp.matrixQR().topRows(diagSize).triangularView<StrictlyLower>().setZero()`.
        """
        var q = self.m_cols
        var R = Mat(q, q)
        for j in range(q):
            for i in range(j + 1):
                R[i, j] = self.m_qr[i, j]
        return R^

    @always_inline
    def apply_q_on_left(self, mut M: Mat):
        """M <- Q * M, i.e. H_0 * H_1 * ... * H_{m_cols-1} * M, applied via
        compact-WY panels of `block_size` reflectors at a time (2 GEMMs per
        panel) instead of one rank-1 update per reflector. M here is the
        full rows x rows (or cols x cols) matrixU/matrixV, so this is the
        single most expensive Householder-apply in the whole solve for
        rectangular inputs.
        """
        comptime block_size = 64
        var k_hi = self.m_cols - 1
        while k_hi >= 0:
            var plen = min(block_size, k_hi + 1)
            var kb = k_hi - plen + 1
            var block_rows = self.m_rows - kb
            var V = Mat(block_rows, plen)
            var taus = Vec(plen)
            for j in range(plen):
                var k = kb + j
                taus[j] = self.m_hCoeffs[k]
                V[j, j] = RealScalar(1)
                var essential = self.m_qr.col(k).segment(k + 1, self.m_rows - k - 1)
                V.col(j).segment(j + 1, len(essential)).copyFrom(essential)
            var C = M.block(kb, 0, block_rows, M.cols())
            apply_compact_wy_block(C, V, taus)
            k_hi = kb - 1

struct BDCSVD:
    var m_impl: BDCSVDImpl
    var m_isTranspose: Bool
    var m_computeU: Bool
    var m_computeV: Bool
    var m_computeThinU: Bool
    var m_computeThinV: Bool
    var m_matrixU: Mat
    var m_matrixV: Mat
    var m_singularValues: Vec
    var m_nonzeroSingularValues: Int
    var m_info: ComputationInfo
    var m_diagSize: Int
    var m_numIters: Int
    # Dense fallback for small problems (cols < algoSwap) and for the
    # base cases inside m_impl.divide().
    var smallSvd: JacobiSVD
    # QR pre-pass ("R-Bidiagonalization"): for very rectangular inputs,
    # bidiagonalizing the small diagSize x diagSize R factor is much
    # cheaper than bidiagonalizing the full tall/wide matrix directly.
    var m_useQrDecomp: Bool
    var qrDecomp: HouseholderQR

    @always_inline
    def __init__(out self):
        self.m_impl = BDCSVDImpl()
        self.m_isTranspose = False
        self.m_computeU = False
        self.m_computeV = False
        self.m_computeThinU = False
        self.m_computeThinV = False
        self.m_matrixU = Mat(0, 0)
        self.m_matrixV = Mat(0, 0)
        self.m_singularValues = Vec(0)
        self.m_nonzeroSingularValues = 0
        self.m_info = INFO_SUCCESS
        self.m_diagSize = 0
        self.m_numIters = 0
        self.smallSvd = JacobiSVD(True)
        self.m_useQrDecomp = False
        self.qrDecomp = HouseholderQR()

    @always_inline
    def setSwitchSize(mut self, s: Int):
        self.m_impl.setAlgoSwap(s)

    @always_inline
    def info(self) -> ComputationInfo:
        return self.m_info

    @always_inline
    def singularValues(self) -> Vec:
        return self.m_singularValues.segment(0, len(self.m_singularValues))

    @always_inline
    def matrixU(self) -> Mat:
        return self.m_matrixU.block(0, 0, self.m_matrixU.rows(), self.m_matrixU.cols())

    @always_inline
    def matrixV(self) -> Mat:
        return self.m_matrixV.block(0, 0, self.m_matrixV.rows(), self.m_matrixV.cols())

    @always_inline
    def nonzeroSingularValues(self) -> Int:
        return self.m_nonzeroSingularValues

    def allocate(
        mut self,
        rows: Int,
        cols: Int,
        computeU: Bool,
        computeV: Bool,
        thinU: Bool = False,
        thinV: Bool = False,
    ):
        self.m_diagSize = min(rows, cols)
        self.m_isTranspose = cols > rows
        self.m_computeU = computeU
        self.m_computeV = computeV
        self.m_computeThinU = thinU and computeU
        self.m_computeThinV = thinV and computeV

        # Same crossover Eigen uses (based on LAPACK dgesdd's 11.0/6.0,
        # widened to avoid a regression for relatively square matrices):
        # for a matrix rectangular enough, QR-then-bidiagonalize-R beats
        # bidiagonalizing the full matrix directly.
        self.m_useQrDecomp = (rows // 4 > cols) or (cols // 4 > rows)

        var compU = computeV
        var compV = computeU
        if self.m_isTranspose:
            var tmp = compU
            compU = compV
            compV = tmp
        self.m_impl.allocate(self.m_diagSize, compU, compV)

    @always_inline
    def extractSingularValues(mut self, scale: RealScalar):
        var considerZero = REAL_MIN
        self.m_singularValues = Vec(self.m_diagSize)
        self.m_nonzeroSingularValues = self.m_diagSize
        for i in range(self.m_diagSize):
            var a = abs(self.m_impl.computed()[i, i])
            self.m_singularValues[i] = a * scale
            if a < considerZero:
                self.m_nonzeroSingularValues = i
                for j in range(i + 1, self.m_diagSize):
                    self.m_singularValues[j] = RealScalar(0)
                break

    # compute_bidiagonal_impl: SVD of a bidiagonal matrix given directly as
    # (diagonal, superdiagonal). No bidiagonalization step needed.
    # No thinU/thinV parameters here: this path is inherently n x n
    # (rows == cols == diagSize == len(diag)), so thin and full always coincide.
    @always_inline
    def compute_bidiagonal(
        mut self, diag: Vec, superdiag: Vec, computeU: Bool, computeV: Bool
    ) -> ComputationInfo:
        var n = len(diag)
        self.allocate(n, n, computeU, computeV)
        self.m_isTranspose = False

        if n == 0:
            self.m_info = INFO_SUCCESS
            self.m_nonzeroSingularValues = 0
            return self.m_info

        var diagScale = diag.cwiseAbsMax()
        var superdiagScale = superdiag.cwiseAbsMax() if n > 1 else RealScalar(0)
        var scale = diagScale if diagScale > superdiagScale else superdiagScale
        if scale == RealScalar(0):
            scale = RealScalar(1)

        # Small problem: skip D&C, hand the dense bidiagonal straight to
        # the base-case solver.
        if n < self.m_impl.algoSwap():
            var B = Mat(n, n)
            for i in range(n):
                B[i, i] = diag[i] / scale
            for i in range(n - 1):
                B[i + 1, i] = superdiag[i] / scale
            self.smallSvd.compute(B)
            self.m_info = self.smallSvd.info()
            if self.m_info == INFO_SUCCESS or self.m_info == INFO_NO_CONVERGENCE:
                var sv = self.smallSvd.singularValues()
                self.m_singularValues = Vec(n)
                for i in range(n):
                    self.m_singularValues[i] = sv[i] * scale
                self.m_nonzeroSingularValues = n
                if computeU:
                    self.m_matrixU = self.smallSvd.matrixU()
                if computeV:
                    self.m_matrixV = self.smallSvd.matrixV()
            return self.m_info

        # D&C operates on B^T: computed(i,i) = d_i / scale, computed(i+1,i) = e_i / scale.
        var nU = self.m_impl.naiveU()
        nU.setZero()
        var nV = self.m_impl.naiveV()
        nV.setZero()
        var cView = self.m_impl.computed()
        cView.setZero()
        for i in range(n):
            cView[i, i] = diag[i] / scale
        for i in range(n - 1):
            cView[i + 1, i] = superdiag[i] / scale

        self.m_impl.splitNegligibleSuperdiagonal(n)
        self.m_impl.divide(0, n - 1, 0, 0, 0)
        self.m_info = self.m_impl.info()
        self.m_numIters = self.m_impl.numIters()
        if self.m_info != INFO_SUCCESS and self.m_info != INFO_NO_CONVERGENCE:
            return self.m_info

        self.extractSingularValues(scale)

        # D&C computes B^T = naiveU * S * naiveV^T, so B = naiveV * S * naiveU^T:
        # U_of_B = naiveV, V_of_B = naiveU. No Householder reflectors to
        # apply here since there was no bidiagonalization step.
        if computeU:
            self.m_matrixU = Mat(n, n)
            embed_topleft(self.m_matrixU, self.m_impl.naiveV(), n)
        if computeV:
            self.m_matrixV = Mat(n, n)
            embed_topleft(self.m_matrixV, self.m_impl.naiveU(), n)

        return self.m_info

    # compute_impl: SVD of a general dense m x n matrix A. Bidiagonalizes
    # via UpperBidiagonalization.
    # Runs the QR pre-pass ("R-Bidiagonalization") for very rectangular
    # inputs, via HouseholderQR.
    @always_inline
    def compute(
        mut self, A: Mat, computeU: Bool, computeV: Bool, thinU: Bool = False, thinV: Bool = False
    ) -> ComputationInfo:
        var rows = A.rows()
        var cols = A.cols()
        self.allocate(rows, cols, computeU, computeV, thinU, thinV)

        # Small problem: fall back to the dense base-case solver directly.
        if cols < self.m_impl.algoSwap():
            self.smallSvd.compute(A, thinU=self.m_computeThinU, thinV=self.m_computeThinV)
            self.m_info = self.smallSvd.info()
            if self.m_info == INFO_SUCCESS or self.m_info == INFO_NO_CONVERGENCE:
                self.m_singularValues = self.smallSvd.singularValues()
                self.m_nonzeroSingularValues = self.m_diagSize
                if computeU:
                    self.m_matrixU = self.smallSvd.matrixU()
                if computeV:
                    self.m_matrixV = self.smallSvd.matrixV()
            return self.m_info

        var scale = A.cwiseAbsMax()
        if scale == RealScalar(0):
            scale = RealScalar(1)

        var work = prepare_work(A, self.m_isTranspose)
        mat_scale(work, scale)

        var bid = UpperBidiagonalization()
        if self.m_useQrDecomp:
            # `work` is always the tall (or square) orientation here.
            # So QR always has rows >= cols, no extra transpose needed.
            self.qrDecomp.compute(work)
            var R = self.qrDecomp.matrixR()
            bid.compute(R)
        else:
            bid.compute(work)

        var nU = self.m_impl.naiveU()
        nU.setZero()
        var nV = self.m_impl.naiveV()
        nV.setZero()
        var cView = self.m_impl.computed()
        cView.setZero()
        var diag = bid.bidiagonal_diagonal()
        var superdiag = bid.bidiagonal_superdiagonal()
        for i in range(self.m_diagSize):
            cView[i, i] = diag[i]
        for i in range(self.m_diagSize - 1):
            cView[i + 1, i] = superdiag[i]

        self.m_impl.splitNegligibleSuperdiagonal(self.m_diagSize)
        self.m_impl.divide(0, self.m_diagSize - 1, 0, 0, 0)
        self.m_info = self.m_impl.info()
        self.m_numIters = self.m_impl.numIters()
        if self.m_info != INFO_SUCCESS and self.m_info != INFO_NO_CONVERGENCE:
            return self.m_info

        self.extractSingularValues(scale)

        # Note the U/V exchange: m_matrixU is built from naiveV (via the
        # *U*-side Householder reflectors) and vice versa — same swap the
        # original's copyUV() performs, then un-swapped again if the input
        # was transposed.
        var naiveU_ = self.m_impl.naiveU()
        var naiveV_ = self.m_impl.naiveV()

        # Thin U/V: build m_matrixU/m_matrixV only diagSize columns wide
        # instead of rows/cols wide. embed_topleft only ever touches the top
        # diagSize x diagSize corner regardless, and apply_u_on_left /
        # apply_v_on_left / apply_q_on_left all size their work off
        # M.cols(), so this alone makes every later apply proportionally cheaper.
        var Ucols = self.m_diagSize if self.m_computeThinU else rows
        var Vcols = self.m_diagSize if self.m_computeThinV else cols
        if not self.m_isTranspose:
            if computeU:
                self.m_matrixU = mat_identity(rows, Ucols)
                embed_topleft(self.m_matrixU, naiveV_, self.m_diagSize)
                bid.apply_u_on_left(self.m_matrixU)
            if computeV:
                self.m_matrixV = mat_identity(cols, Vcols)
                embed_topleft(self.m_matrixV, naiveU_, self.m_diagSize)
                bid.apply_v_on_left(self.m_matrixV)
        else:
            if computeU:
                self.m_matrixU = mat_identity(rows, Ucols)
                embed_topleft(self.m_matrixU, naiveU_, self.m_diagSize)
                bid.apply_v_on_left(self.m_matrixU)
            if computeV:
                self.m_matrixV = mat_identity(cols, Vcols)
                embed_topleft(self.m_matrixV, naiveV_, self.m_diagSize)
                bid.apply_u_on_left(self.m_matrixV)

        # QR pre-pass tail: bid's Householder reflectors above only account
        # for R (diagSize x diagSize), so they've only touched the top
        # diagSize rows/cols of m_matrixU/m_matrixV so far. Lift that back
        # into the full tall/wide space by applying Q (from the QR of
        # `work`, which is rows x rows if !isTranspose, cols x cols if
        # isTranspose — matching whichever of U/V sits on `work`'s row
        # space) on the left.
        if self.m_useQrDecomp:
            if self.m_isTranspose and computeV:
                self.qrDecomp.apply_q_on_left(self.m_matrixV)
            elif not self.m_isTranspose and computeU:
                self.qrDecomp.apply_q_on_left(self.m_matrixU)

        return self.m_info
