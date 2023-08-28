import random

import pytest

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.open_hemisphere import (
    OpenHemisphere,
    OpenHemispherePullbackMetric,
    OpenHemisphereToHyperboloidDiffeo,
    ProductOpenHemispheres,
    ProductOpenHemispheresMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import OpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.open_hemisphere import OpenHemispherePullbackMetricTestData
from .data.product_manifold import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        1,
        random.randint(2, 4),
    ],
)
def diffeos(request):
    dim = request.param
    request.cls.space = OpenHemisphere(dim, equip=False)
    request.cls.image_space = Hyperboloid(dim, equip=False)


@pytest.mark.usefixtures("diffeos")
class TestOpenHemisphereToHyperboloidDiffeo(
    DiffeoTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    diffeo = OpenHemisphereToHyperboloidDiffeo()
    testing_data = DiffeoTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        random.randint(2, 4),
    ],
)
def open_hemispheres(request):
    dim = request.param
    request.cls.space = OpenHemisphere(dim, equip=False)


@pytest.mark.usefixtures("open_hemispheres")
class TestOpenHemisphere(OpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = OpenSetTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        random.randint(2, 4),
    ],
)
def equipped_open_hemispheres(request):
    dim = request.param
    space = request.cls.space = OpenHemisphere(dim, equip=False)
    space.equip_with_metric(OpenHemispherePullbackMetric)


@pytest.mark.usefixtures("equipped_open_hemispheres")
class TestOpenHemispherePullbackMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = OpenHemispherePullbackMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        3,
    ],
)
def product_open_hemispheres(request):
    n = request.param
    request.cls.space = ProductOpenHemispheres(n, equip=False)


@pytest.mark.usefixtures("product_open_hemispheres")
class TestProductOpenHemispheres(
    ProductManifoldTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductManifoldTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        3,
    ],
)
def equipped_product_open_hemispheres(request):
    n = request.param
    space = request.cls.space = ProductOpenHemispheres(n, equip=False)
    space.equip_with_metric(ProductOpenHemispheresMetric)


@pytest.mark.usefixtures("equipped_product_open_hemispheres")
class TestProductOpenHemispheresMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductRiemannianMetricTestData()
