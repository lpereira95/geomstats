from geomstats.geometry.base import MatrixVectorSpace, VectorSpaceOpenSet
from geomstats.geometry.lie_group import MatrixLieGroup


class DiagonalMatrices(MatrixVectorSpace):
    pass


class PositiveDiagonalMatrices(MatrixLieGroup, VectorSpaceOpenSet):
    pass
    # TODO: relate metric with positive reals
