"""The manifold of full-rank correlation matrices.

Lead author: Yann Thanwerdas.
"""

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import (
    LieCholeskyMetric,
    SPDAffineMetric,
    SPDMatrices,
)


class FullRankCorrelationMatrices(LevelSet):
    """Class for the manifold of full-rank correlation matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(dim=int(n * (n - 1) / 2), equip=equip)

    def _define_embedding_space(self):
        return SPDMatrices(n=self.n)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return FullRankCorrelationAffineQuotientMetric

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n]
        """
        return Matrices.diagonal(point) - gs.ones(self.n)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n]
        """
        submersed_vector = Matrices.diagonal(vector)
        if point is not None and point.ndim > vector.ndim:
            return gs.broadcast_to(submersed_vector, point.shape[:-1])

        return submersed_vector

    @staticmethod
    def diag_action(diagonal_vec, point):
        r"""Apply a diagonal matrix on an SPD matrices by congruence.

        The action of :math:`D` on :math:`\Sigma` is given by :math:`D
        \Sigma D. The diagonal matrix must be passed as a vector representing
        its diagonal.

        Parameters
        ----------
        diagonal_vec : array-like, shape=[..., n]
            Vector coefficient of the diagonal matrix.
        point : array-like, shape=[..., n, n]
            Symmetric Positive definite matrix.

        Returns
        -------
        mat : array-like, shape=[..., n, n]
            Symmetric matrix obtained by the action of `diagonal_vec` on
            `point`.
        """
        return point * gs.outer(diagonal_vec, diagonal_vec)

    @classmethod
    def from_covariance(cls, point):
        r"""Compute the correlation matrix associated to an SPD matrix.

        The correlation matrix associated to an SPD matrix (the covariance)
        :math:`\Sigma` is given by :math:`D  \Sigma D` where :math:`D` is
        the inverse square-root of the diagonal of :math:`\Sigma`.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Symmetric Positive definite matrix.

        Returns
        -------
        corr : array-like, shape=[..., n, n]
            Correlation matrix obtained by dividing all elements by the
            diagonal entries.
        """
        diag_vec = Matrices.diagonal(point) ** (-0.5)
        return cls.diag_action(diag_vec, point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample full-rank correlation matrices by projecting random SPD mats.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Bound of the interval in which to sample.
            Optional, default: 1.

        Returns
        -------
        cor : array-like, shape=[n_samples, n, n]
            Sample of full-rank correlation matrices.
        """
        spd = self.embedding_space.random_point(n_samples, bound=bound)
        return self.from_covariance(spd)

    def projection(self, point):
        """Project a matrix to the space of correlation matrices.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.
        """
        spd = self.embedding_space.projection(point)
        return self.from_covariance(spd)

    def to_tangent(self, vector, base_point):
        """Project a matrix to the tangent space at a base point.

        The tangent space to the space of correlation matrices is the space of
        symmetric matrices with null diagonal.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to project
        base_point : array-like, shape=[..., n, n]
            Correlation matrix.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Symmetric matrix with 0 diagonal.
        """
        sym = self.embedding_space.to_tangent(vector, base_point)
        mask_diag = gs.ones_like(vector) - gs.eye(self.n)
        return sym * mask_diag


class CorrelationMatricesAffineBundle(FiberBundle):
    """Fiber bundle to construct the quotient metric on correlation matrices.

    Correlation matrices are obtained as the quotient of the space of SPD
    matrices by the action by congruence of diagonal matrices.
    Metric-dependent methods implemented for affine-invariant metric.

    References
    ----------
    .. [TP21] Thanwerdas, Yann, and Xavier Pennec. “Geodesics and Curvature of
        the Quotient-Affine Metrics on Full-Rank CorrelationMatrices.”
        In Proceedings of Geometric Science of Information.
        Paris, France, 2021.
        https://hal.archives-ouvertes.fr/hal-03157992.
    """

    def __init__(self, total_space):
        super().__init__(
            total_space=total_space,
            group_dim=total_space.n,
            group_action=FullRankCorrelationMatrices.diag_action,
        )

    @staticmethod
    def riemannian_submersion(point):
        """Compute the correlation matrix associated to an SPD matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            SPD matrix.

        Returns
        -------
        cor : array_like, shape=[..., n, n]
            Full rank correlation matrix.
        """
        diagonal = Matrices.diagonal(point) ** (-0.5)
        return point * gs.outer(diagonal, diagonal)

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        """Compute the differential of the submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        result : array-like, shape=[..., n, n]
        """
        diagonal_bp = Matrices.diagonal(base_point)
        diagonal_tv = Matrices.diagonal(tangent_vec)

        diagonal = diagonal_tv / diagonal_bp
        aux = base_point * (diagonal[..., None, :] + diagonal[..., :, None])
        mat = tangent_vec - 0.5 * aux
        return self.group_action(diagonal_bp ** (-0.5), mat)

    def vertical_projection(self, tangent_vec, base_point):
        """Compute the vertical projection wrt the affine-invariant metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        ver : array-like, shape=[..., n, n]
            Vertical projection.
        """
        n = self.total_space.n
        inverse_base_point = GeneralLinear.inverse(base_point)
        operator = gs.eye(n) + base_point * inverse_base_point
        inverse_operator = GeneralLinear.inverse(operator)
        vector = gs.einsum("...ij,...ji->...i", inverse_base_point, tangent_vec)
        diagonal = gs.einsum("...ij,...j->...i", inverse_operator, vector)
        return base_point * (diagonal[..., None, :] + diagonal[..., :, None])

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Compute the horizontal lift wrt the affine-invariant metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector of the manifold of full-rank correlation matrices.
        fiber_point : array-like, shape=[..., n, n]
            SPD matrix in the fiber above point.
        base_point : array-like, shape=[..., n, n]
            Full-rank correlation matrix.

        Returns
        -------
        hor_lift : array-like, shape=[..., n, n]
            Horizontal lift of tangent_vec from point to base_point.
        """
        if base_point is not None:
            return self.horizontal_projection(tangent_vec, base_point)
        diagonal_point = Matrices.diagonal(fiber_point) ** 0.5
        lift = self.group_action(diagonal_point, tangent_vec)
        return self.horizontal_projection(lift, base_point=fiber_point)


class CorrelationMatricesBundle(FiberBundle):
    """Fiber bundle to construct the quotient metric on correlation matrices.

    Correlation matrices are obtained as the quotient of the space of SPD
    matrices by the action by congruence of diagonal matrices.
    Metric-dependent methods implemented for any total space metric.

    References
    ----------
    .. [thanwerdas2022] Yann Thanwerdas. Riemannian and stratified
    geometries on covariance and correlation matrices. Differential
    Geometry [math.DG]. Université Côte d'Azur, 2022.
    """

    def __init__(self, total_space):
        super().__init__(
            total_space=total_space,
            group_dim=total_space.n,
            group_action=FullRankCorrelationMatrices.diag_action,
        )
        self._diagonal_matrices_basis_ = None

    def _generate_diagonal_matrices_basis(self):
        """Generate basis for diagonal matrices."""
        n = self.total_space.n
        indices = [(i, i, i) for i in range(n)]
        return gs.array_from_sparse(indices, gs.ones(n), (n, n, n))

    @property
    def _diagonal_matrices_basis(self):
        """Basis of the diagonal matrices space."""
        if self._diagonal_matrices_basis_ is None:
            self._diagonal_matrices_basis_ = self._generate_diagonal_matrices_basis()
        return self._diagonal_matrices_basis_

    @staticmethod
    def riemannian_submersion(point):  # correlation map
        """Compute the correlation matrix associated to an SPD matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            SPD matrix.

        Returns
        -------
        cor : array_like, shape=[..., n, n]
            Full rank correlation matrix.
        """
        diagonal = Matrices.diagonal(point) ** (-0.5)
        return point * gs.outer(diagonal, diagonal)

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        """Compute the differential of the submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        result : array-like, shape=[..., n, n]
        """
        diagonal_bp = Matrices.diagonal(base_point)
        diagonal_tv = Matrices.diagonal(tangent_vec)

        diagonal = diagonal_tv / diagonal_bp
        aux = base_point * (diagonal[..., None, :] + diagonal[..., :, None])
        mat = tangent_vec - 0.5 * aux
        return self.group_action(diagonal_bp ** (-0.5), mat)

    def _generate_vertical_space_basis(self, base_point):
        """Generate basis for vertical space."""
        basis_list = []
        for elt in self._diagonal_matrices_basis:
            X_i = elt @ base_point + base_point @ elt
            basis_list.append(X_i)
        vertical_basis = gs.array(basis_list)

        if gs.ndim(vertical_basis) == 4:
            return gs.moveaxis(vertical_basis, 0, 1)

        return vertical_basis

    def _compute_vertical_space_orthogonal_basis(self, point):
        """Compute vertical space orthogonal basis.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in the total space.
        """
        n_o_basis = self._generate_vertical_space_basis(point)
        return self._gram_schmidt_orthonormalize(basis=n_o_basis, base_point=point)

    def _gram_schmidt_orthonormalize(self, basis, base_point):
        """Gram-Schmidt orthonormalization vectorized."""
        if gs.ndim(base_point) == 2:
            return self._gram_schmidt_orthonormalize_single(basis, base_point)

        return gs.array(
            [
                self._gram_schmidt_orthonormalize_single(basis_, base_point_)
                for basis_, base_point_ in zip(basis, base_point)
            ]
        )

    def _gram_schmidt_orthonormalize_single(self, basis, base_point):
        """Gram-Schmidt orthonormalization for a single base point."""
        orthonormal_basis = []
        for vector in basis:
            for o_vector in orthonormal_basis:
                num = self.total_space.metric.inner_product(
                    tangent_vec_a=vector, tangent_vec_b=o_vector, base_point=base_point
                )
                dem = self.total_space.metric.inner_product(
                    tangent_vec_a=o_vector,
                    tangent_vec_b=o_vector,
                    base_point=base_point,
                )
                projection = num / dem
                vector -= projection * o_vector
            squared_norm = self.total_space.metric.inner_product(
                tangent_vec_a=vector, tangent_vec_b=vector, base_point=base_point
            )
            norm = gs.sqrt(squared_norm)
            orthonormal_basis.append(vector / norm)
        return gs.array(orthonormal_basis)

    def vertical_projection(self, tangent_vec, base_point):
        """Compute the vertical projection wrt the total space metric.

        Algorithm is independent of total space metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        ver : array-like, shape=[..., n, n]
            Vertical projection.
        """

        def _get_coefficients(o_basis_, tangent_vec_, base_point_):
            coefficients = []
            for basis_vector in o_basis_:
                coefficient = self.total_space.metric.inner_product(
                    tangent_vec_a=tangent_vec_,
                    tangent_vec_b=basis_vector,
                    base_point=base_point_,
                )
                coefficients.append(coefficient)
            return gs.array(coefficients)

        o_basis = self._compute_vertical_space_orthogonal_basis(base_point)
        if o_basis.ndim == 3:
            coefficients = _get_coefficients(o_basis, tangent_vec, base_point)
            return gs.einsum("i...,ijk->...jk", coefficients, o_basis)

        coefficients = gs.stack(
            [
                _get_coefficients(o_basis_, tangent_vec_, base_point_)
                for o_basis_, tangent_vec_, base_point_ in zip(
                    o_basis, tangent_vec, base_point
                )
            ]
        )
        return gs.einsum("...i,...ijk->...jk", coefficients, o_basis)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Compute the horizontal lift wrt the total space metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector of the manifold of full-rank correlation matrices.
        fiber_point : array-like, shape=[..., n, n]
            SPD matrix in the fiber above point.
        base_point : array-like, shape=[..., n, n]
            Full-rank correlation matrix.

        Returns
        -------
        hor_lift : array-like, shape=[..., n, n]
            Horizontal lift of tangent_vec from point to base_point.
        """
        if fiber_point is None and base_point is not None:
            return self.horizontal_projection(
                tangent_vec=tangent_vec, base_point=base_point
            )
        diagonal_point = Matrices.diagonal(fiber_point) ** 0.5
        lift = FullRankCorrelationMatrices.diag_action(diagonal_point, tangent_vec)
        return self.horizontal_projection(tangent_vec=lift, base_point=fiber_point)


class FullRankCorrelationAffineQuotientMetric(QuotientMetric):
    """Class for the quotient of the affine-invariant metric.

    The affine-invariant metric on SPD matrices is invariant under the
    action of diagonal matrices, thus it induces a quotient metric on the
    manifold of full-rank correlation matrices.
    """

    def __init__(self, space, total_space=None):
        if total_space is None:
            total_space = SPDMatrices(space.n, equip=False)
            total_space.equip_with_metric(SPDAffineMetric)

        super().__init__(
            space=space,
            fiber_bundle=CorrelationMatricesAffineBundle(total_space),
        )


class FullRankCorrelationQuotientLieCholeskyMetric(QuotientMetric):
    """Class for the quotient of the Lie-Cholesky metric."""

    def __init__(self, space, total_space=None):
        if total_space is None:
            total_space = SPDMatrices(space.n, equip=False)
            total_space.equip_with_metric(LieCholeskyMetric)

        super().__init__(
            space=space,
            fiber_bundle=CorrelationMatricesBundle(total_space),
        )
