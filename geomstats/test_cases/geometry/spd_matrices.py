import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase


class SPDMatricesTestCase(VectorSpaceOpenSetTestCase):
    pass


class SPDEuclideanMetricTestCase(MatricesMetricTestCase):
    def test_exp_domain(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp_domain(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)


def polar_differential_2(tangent_vec, base_point):
    """Differential of polar decomposition for n=2."""
    group_elem, sigma = gs.linalg.polar(base_point, side="left")

    aux = Matrices.mul(tangent_vec, Matrices.transpose(group_elem)) - Matrices.mul(
        group_elem, Matrices.transpose(tangent_vec)
    )

    return Matrices.mul(tangent_vec, Matrices.transpose(group_elem)) - gs.einsum(
        "...,...ij->...ij", 1 / gs.trace(sigma), Matrices.mul(sigma, aux)
    )


def polar_differential_3(tangent_vec, base_point):
    """Differential of polar decomposition for n=3."""
    n = base_point.shape[-1]
    group_elem, sigma = gs.linalg.polar(base_point, side="left")

    z = gs.einsum("...,...ij->...ij", gs.trace(sigma), gs.eye(n)) - sigma
    aux = Matrices.mul(tangent_vec, Matrices.transpose(group_elem)) - Matrices.mul(
        group_elem, Matrices.transpose(tangent_vec)
    )

    return Matrices.mul(tangent_vec, Matrices.transpose(group_elem)) - gs.einsum(
        "...,...ij->...ij", 1 / gs.linalg.det(z), Matrices.mul(sigma, z, aux, z)
    )
