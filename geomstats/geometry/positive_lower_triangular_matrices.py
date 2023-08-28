"""The manifold of lower triangular matrices with positive diagonal elements.

Lead author: Saiteja Utpala.
"""

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.open_hemisphere import ProductOpenHemispheres
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import check_is_batch


class PositiveLowerTriangularMatrices(OpenSet):
    """Manifold of lower triangular matrices with >0 diagonal.

    This manifold is also called the cholesky space.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.

    References
    ----------
    .. [TP2019] Z Lin. "Riemannian Geometry of Symmetric
        Positive Definite Matrices Via Cholesky Decomposition"
        SIAM journal on Matrix Analysis and Applications , 2019.
        https://arxiv.org/abs/1908.09326
    """

    def __init__(self, n, equip=True):
        super().__init__(
            dim=int(n * (n + 1) / 2),
            embedding_space=LowerTriangularMatrices(n, equip=False),
            equip=equip,
        )
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return CholeskyMetric

    def belongs(self, point, atol=gs.atol):
        """Check if mat is lower triangular with >0 diagonal.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat belongs to cholesky space.
        """
        is_lower_triangular = self.embedding_space.belongs(point, atol)
        diagonal = Matrices.diagonal(point)
        is_positive = gs.all(diagonal > 0, axis=-1)
        return gs.logical_and(is_lower_triangular, is_positive)

    def projection(self, point):
        """Project a matrix to the PLT space.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        projected: array-like, shape=[..., n, n]
        """
        vec_diag = gs.abs(Matrices.diagonal(point))
        vec_diag = gs.where(vec_diag < gs.atol, gs.atol, vec_diag)
        diag = gs.vec_to_diag(vec_diag)
        strictly_lower_triangular = gs.tril(point, k=-1)
        return diag + strictly_lower_triangular


class CholeskyMetric(RiemannianMetric):
    """Class for Cholesky metric on Cholesky space.

    References
    ----------
    .. [TP2019] . "Riemannian Geometry of Symmetric
        Positive Definite Matrices Via Cholesky Decomposition"
        SIAM journal on Matrix Analysis and Applications , 2019.
        https://arxiv.org/abs/1908.09326
    """

    @staticmethod
    def diag_inner_product(tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product using only diagonal elements.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        ip_diagonal : array-like, shape=[...]
            Inner-product.
        """
        inv_sqrt_diagonal = gs.power(Matrices.diagonal(base_point), -2)
        tangent_vec_a_diagonal = Matrices.diagonal(tangent_vec_a)
        tangent_vec_b_diagonal = Matrices.diagonal(tangent_vec_b)
        prod = tangent_vec_a_diagonal * tangent_vec_b_diagonal * inv_sqrt_diagonal
        return gs.sum(prod, axis=-1)

    @staticmethod
    def strictly_lower_inner_product(tangent_vec_a, tangent_vec_b):
        """Compute the inner product using only strictly lower triangular elements.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.

        Returns
        -------
        ip_sl : array-like, shape=[...]
            Inner-product.
        """
        sl_tagnet_vec_a = gs.tril_to_vec(tangent_vec_a, k=-1)
        sl_tagnet_vec_b = gs.tril_to_vec(tangent_vec_b, k=-1)
        ip_sl = gs.dot(sl_tagnet_vec_a, sl_tagnet_vec_b)
        return ip_sl

    @classmethod
    def inner_product(cls, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the cholesky Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...]
            Inner-product.
        """
        diag_inner_product = cls.diag_inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        strictly_lower_inner_product = cls.strictly_lower_inner_product(
            tangent_vec_a, tangent_vec_b
        )
        return diag_inner_product + strictly_lower_inner_product

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Cholesky exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Cholesky metric.
        This gives a lower triangular matrix with positive elements.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        sl_base_point = Matrices.to_strictly_lower_triangular(base_point)
        sl_tangent_vec = Matrices.to_strictly_lower_triangular(tangent_vec)
        diag_base_point = Matrices.diagonal(base_point)
        diag_tangent_vec = Matrices.diagonal(tangent_vec)
        diag_product_expm = gs.exp(gs.divide(diag_tangent_vec, diag_base_point))

        sl_exp = sl_base_point + sl_tangent_vec
        diag_exp = gs.vec_to_diag(diag_base_point * diag_product_expm)
        return sl_exp + diag_exp

    def log(self, point, base_point, **kwargs):
        """Compute the Cholesky logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Cholesky metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm.
        """
        sl_base_point = Matrices.to_strictly_lower_triangular(base_point)
        sl_point = Matrices.to_strictly_lower_triangular(point)
        diag_base_point = Matrices.diagonal(base_point)
        diag_point = Matrices.diagonal(point)
        diag_product_logm = gs.log(gs.divide(diag_point, diag_base_point))

        sl_log = sl_point - sl_base_point
        diag_log = gs.vec_to_diag(diag_base_point * diag_product_logm)
        return sl_log + diag_log

    def squared_dist(self, point_a, point_b, **kwargs):
        """Compute the Cholesky Metric squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Point.
        point_b : array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        _ : array-like, shape=[...]
            Riemannian squared distance.
        """
        log_diag_a = gs.log(Matrices.diagonal(point_a))
        log_diag_b = gs.log(Matrices.diagonal(point_b))
        diag_diff = log_diag_a - log_diag_b
        squared_dist_diag = gs.sum((diag_diff) ** 2, axis=-1)

        sl_a = Matrices.to_strictly_lower_triangular(point_a)
        sl_b = Matrices.to_strictly_lower_triangular(point_b)
        sl_diff = sl_a - sl_b
        squared_dist_sl = Matrices.frobenius_product(sl_diff, sl_diff)
        return squared_dist_sl + squared_dist_diag


class UnitNormedPLTMatrices(OpenSet):
    """Space of positive lower triangular matrices with unit-normed rows.

    Rows are unit normed for the Euclidean norm.
    """

    # TODO: notice is is not an OpenSet of a vector space

    def __init__(self, n, equip=True):
        super().__init__(
            dim=int(n * (n + 1) / 2),
            embedding_space=PositiveLowerTriangularMatrices(n),
            equip=equip,
        )
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return UnitNormedPLTMatricesPullbackMetric

    def belongs(self, point, atol=gs.atol):
        """Check if a point belongs to the unit normed plt matrices."""
        is_plt = self.embedding_space.belongs(point)
        row_norms = gs.linalg.norm(point, axis=-1)
        is_unit_normed = gs.logical_and(
            gs.all(row_norms < 1.0 + atol, axis=-1),
            gs.all(1.0 - atol < row_norms, axis=-1),
        )
        return gs.logical_and(is_plt, is_unit_normed)

    def projection(self, point):
        """Project a point on the unit normed plt matrices.

        It is the same as the projection on the sphere.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding plt space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected on the open unit normed plt matrices.
        """
        point = self.embedding_space.projection(point)
        norm = gs.linalg.norm(point, axis=-1)
        return gs.einsum("...,...i->...i", 1.0 / norm, point)

    # def to_tangent(self, vector, base_point=None):
    #     # this is completely stupid but if not there the to_tangent method doesn't project well
    #     """Project a vector to a tangent space of the vector space.

    #     This method is for compatibility and returns vector.

    #     Parameters
    #     ----------
    #     vector : array-like, shape=[..., *point_shape]
    #         Vector.
    #     base_point : array-like, shape=[..., *point_shape]
    #         Point in the vector space

    #     Returns
    #     -------
    #     tangent_vec : array-like, shape=[..., *point_shape]
    #         Tangent vector at base point.
    #     """
    #     tangent_vec = self.projection(vector)
    #     if base_point is not None and base_point.ndim > vector.ndim:
    #         return gs.broadcast_to(tangent_vec, base_point.shape)
    #     return tangent_vec

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.embedding_space.is_tangent(vector, base_point, atol)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        return self.embedding_space.to_tangent(vector, base_point)


class UnitNormedPLTMatricesPullbackMetric(PullbackDiffeoMetric):
    def _define_embedding_space(self):
        return ProductOpenHemispheres(self._space.embedding_space.n, equip=True)

    def diffeomorphism(self, base_point):
        return gs.concatenate(
            [
                gs.flip(base_point[..., i, : i + 1], axis=-1)
                for i in range(1, self._space.n)
            ],
            axis=-1,
        )

    def inverse_diffeomorphism(self, image_point):
        def _inverse_diffeomorphism_single(image_point):
            image_point_ = []
            i = 0
            for row in range(1, n):
                image_point_.append(gs.flip(image_point[i : i + (row + 1)]))
                i += row + 1

            image_point_ = gs.hstack(image_point_)

            return gs.array_from_sparse(
                indices,
                gs.hstack([gs.array([1.0]), image_point_]),
                (
                    n,
                    n,
                ),
            )

        n = self._space.n
        indices = [(index_0, index_1) for index_0, index_1 in zip(*gs.tril_indices(n))]
        if check_is_batch(self.embedding_space, image_point):
            return gs.stack(
                [
                    _inverse_diffeomorphism_single(image_point_)
                    for image_point_ in image_point
                ]
            )
        return _inverse_diffeomorphism_single(image_point)

    def tangent_diffeomorphism(self, tangent_vec, base_point):
        # the diffeo is linear, so its pushforward is itself
        return self.diffeomorphism(tangent_vec)

    def inverse_tangent_diffeomorphism(self, image_tangent_vec, image_point):
        # same for the inverse diffeo
        return self.inverse_diffeomorphism(image_point=image_tangent_vec)
