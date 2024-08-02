import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.spd_matrices import (
    CholeskyMap,
    LieCholeskyMetric,
    MatrixPower,
    PolarDecompositionBundle,
    SPDAffineMetric,
    SPDBuresWassersteinMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
    SPDPowerMetric,
    SymMatrixLog,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.geometry.spd_matrices import (
    SPDEuclideanMetricTestCase,
    SPDMatricesTestCase,
    polar_differential_2,
    polar_differential_3,
)

from .data.diffeo import DiffeoTestData
from .data.spd_matrices import (
    CholeskyMapSmokeTestData,
    LieCholeskyMetricTestData,
    MatrixPower05TestData,
    PolarDecompositionBundleTestData,
    SPD2AffineMetricTestData,
    SPD2BuresWassersteinMetricTestData,
    SPD2EuclideanMetricTestData,
    SPD2LogEuclideanMetricTestData,
    SPD3AffineMetricPower05TestData,
    SPD3BuresWassersteinMetricTestData,
    SPD3EuclideanMetricPower05TestData,
    SPD3EuclideanMetricTestData,
    SPD3LogEuclideanMetricTestData,
    SPDAffineMetricTestData,
    SPDBuresWassersteinMetricTestData,
    SPDEuclideanMetricTestData,
    SPDLogEuclideanMetricTestData,
    SPDMatrices2TestData,
    SPDMatrices3TestData,
    SPDMatricesTestData,
    SymMatrixLogSmokeTestData,
)


class TestSymMatrixLog(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False)
    image_space = SymmetricMatrices(n=_n, equip=False)
    diffeo = SymMatrixLog()
    testing_data = DiffeoTestData()


@pytest.mark.smoke
class TestSymMatrixLogSmoke(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = SymMatrixLog()
    testing_data = SymMatrixLogSmokeTestData()


class TestMatrixPower(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = image_space = SPDMatrices(n=_n, equip=False)
    diffeo = MatrixPower(power=gs.random.uniform(low=0.5, high=1.5, size=1))
    testing_data = DiffeoTestData()


@pytest.mark.smoke
class TestMatrixPower05(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = MatrixPower(power=0.5)
    testing_data = MatrixPower05TestData()


class TestCholeskyMap(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False)
    image_space = PositiveLowerTriangularMatrices(n=_n, equip=False)
    diffeo = CholeskyMap()
    testing_data = DiffeoTestData()


@pytest.mark.smoke
class TestCholeskyMapSmoke(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = CholeskyMap()
    testing_data = CholeskyMapSmokeTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spaces(request):
    request.cls.space = SPDMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestSPDMatrices(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPDMatricesTestData()


@pytest.mark.smoke
class TestSPDMatrices2(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=2, equip=False)
    testing_data = SPDMatrices2TestData()


@pytest.mark.smoke
class TestSPDMatrices3(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=3, equip=False)
    testing_data = SPDMatrices3TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spd_with_affine_metric(request):
    n = request.param
    request.cls.space = SPDMatrices(n=n)


@pytest.mark.usefixtures("spd_with_affine_metric")
class TestSPDAffineMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPDAffineMetricTestData()


@pytest.mark.smoke
class TestSPD2AffineMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPD2AffineMetricTestData()
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDAffineMetric)


@pytest.mark.smoke
class TestSPD3AffineMetricPower05(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD3AffineMetricPower05TestData()

    _power = 0.5
    _scale = 1 / _power**2

    image_space = SPDMatrices(n=3, equip=False)
    image_space.equip_with_metric(SPDAffineMetric)
    image_space.equip_with_metric(ScalarProductMetric(image_space, scale=_scale))

    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDPowerMetric, image_space=image_space)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spd_with_bw_metric(request):
    space = request.cls.space = SPDMatrices(n=request.param, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=2.0)


@pytest.mark.redundant
@pytest.mark.usefixtures("spd_with_bw_metric")
class TestSPDBuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinMetricTestData()


@pytest.mark.smoke
class TestSPD2BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD2BuresWassersteinMetricTestData()
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)


@pytest.mark.smoke
class TestSPD3BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD3BuresWassersteinMetricTestData()
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)


@pytest.fixture(
    scope="class",
    params=[random.randint(2, 5)],
)
def spd_with_euclidean(request):
    n = request.param

    space = request.cls.space = SPDMatrices(n=n, equip=False)
    space.equip_with_metric(SPDEuclideanMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=4.0)


@pytest.mark.redundant
@pytest.mark.usefixtures("spd_with_euclidean")
class TestSPDEuclideanMetric(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDEuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD2EuclideanMetric(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDEuclideanMetric)
    testing_data = SPD2EuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3EuclideanMetric(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDEuclideanMetric)
    testing_data = SPD3EuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3EuclideanMetricPower05(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    _power = 0.5
    _scale = 1 / _power**2

    image_space = SPDMatrices(n=3, equip=False)
    image_space.equip_with_metric(SPDEuclideanMetric)
    image_space.equip_with_metric(ScalarProductMetric(image_space, scale=_scale))

    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDPowerMetric, image_space=image_space)

    testing_data = SPD3EuclideanMetricPower05TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spd_with_log_euclidean(request):
    n = request.param
    space = request.cls.space = SPDMatrices(n=n, equip=False)
    space.equip_with_metric(SPDLogEuclideanMetric)


@pytest.mark.redundant
@pytest.mark.usefixtures("spd_with_log_euclidean")
class TestSPDLogEuclideanMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDLogEuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD2LogEuclideanMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDLogEuclideanMetric)
    testing_data = SPD2LogEuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3LogEuclideanMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDLogEuclideanMetric)
    testing_data = SPD3LogEuclideanMetricTestData()


@pytest.mark.slow
@pytest.mark.redundant
class TestLieCholeskyMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False).equip_with_metric(LieCholeskyMetric)
    testing_data = LieCholeskyMetricTestData()


class TestPolarDecompositionBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    total_space = GeneralLinear(n=_n, equip=True)

    total_space.equip_with_group_action(SpecialOrthogonalComposeAction(_n))
    total_space.fiber_bundle = PolarDecompositionBundle(total_space)
    base = SPDMatrices(n=_n, equip=False)

    testing_data = PolarDecompositionBundleTestData()

    @pytest.mark.random
    def test_tangent_riemmanian_submersion_against_alt(self, n_points, atol):
        n = self.total_space.n
        if n > 3:
            return

        if n == 2:
            tangent_polar = polar_differential_2
        else:
            tangent_polar = polar_differential_3

        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        quotient_tangent_vec = (
            self.total_space.fiber_bundle.tangent_riemannian_submersion(
                tangent_vec, base_point
            )
        )
        quotient_tangent_vec_ = tangent_polar(tangent_vec, base_point)

        self.assertAllClose(quotient_tangent_vec, quotient_tangent_vec_, atol=atol)
