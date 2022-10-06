from abc import ABCMeta, abstractmethod

import geomstats


class LadderStep(metaclass=ABCMeta):
    def __init__(self, save_geodesics=False):
        self.save_geodesics = save_geodesics

        self.geodesics_ = []

    @abstractmethod
    def step(self, metric, base_point, next_point, base_shoot):
        pass

    def reset_geodesics(self):
        self.geodesics_ = []

    def post_proc_transported(self, parallel_transporter, transported):
        return transported


class PoleLadderStep(LadderStep):
    """Pole Ladder step."""

    def step(self, metric, base_point, next_point, base_shoot):
        """Compute one Pole Ladder step.

        One step of pole ladder scheme [LP2013a]_ using the geodesic to
        transport along as main_geodesic of the parallelogram.

        Parameters
        ----------
        metric: Connection
        base_point : array-like, shape=[..., dim]
            Point on the manifold, from which to transport.
        next_point : array-like, shape=[..., dim]
            Point on the manifold, to transport to.
        base_shoot : array-like, shape=[..., dim]
            Point on the manifold, end point of the geodesics starting
            from the base point with initial speed to be transported.

        Returns
        -------
        next_tangent_vec : array-like, shape=[..., dim]
            Tangent vector at end point.


        References
        ----------
        .. [LP2013a] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transport
            of Deformations in Time Series of Images: from Schild's to
            Pole Ladder. Journal of Mathematical Imaging and Vision, Springer
            Verlag, 2013,50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩
        """
        mid_tangent_vector_to_shoot = (
            1.0 / 2.0 * metric.log(base_point=base_point, point=next_point)
        )

        mid_point = metric.exp(
            base_point=base_point, tangent_vec=mid_tangent_vector_to_shoot
        )

        tangent_vector_to_shoot = -metric.log(
            base_point=mid_point,
            point=base_shoot,
        )

        end_shoot = metric.exp(
            base_point=mid_point,
            tangent_vec=tangent_vector_to_shoot,
        )

        if self.save_geodesics:
            self.geodesics_.append(
                self._compute_geodesics(
                    metric, base_point, next_point, base_shoot, mid_point, end_shoot
                )
            )

        return end_shoot

    def _compute_geodesics(
        self, metric, base_point, next_point, base_shoot, mid_point, end_shoot
    ):
        main_geodesic = metric.geodesic(initial_point=base_point, end_point=next_point)
        diagonal = metric.geodesic(initial_point=mid_point, end_point=base_shoot)
        final_geodesic = metric.geodesic(initial_point=next_point, end_point=end_shoot)
        return [main_geodesic, diagonal, final_geodesic]

    def post_proc_transported(self, parallel_transporter, transported_tangent_vec):
        if parallel_transporter.n_rungs % 2 == 1:
            transported_tangent_vec *= -1.0

        return transported_tangent_vec


class SchildLadderStep(LadderStep):
    """Schild's Ladder step."""

    def step(self, metric, base_point, next_point, base_shoot):
        """Compute one Schild's Ladder step.

        One step of the Schild's ladder scheme [LP2013a]_ using the geodesic to
        transport along as one side of the parallelogram.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold, from which to transport.
        next_point : array-like, shape=[..., dim]
            Point on the manifold, to transport to.
        base_shoot : array-like, shape=[..., dim]
            Point on the manifold, end point of the geodesics starting
            from the base point with initial speed to be transported.

        Returns
        -------
        end_point : array-like, shape=[..., dim]
            Point on the manifold, closes the geodesic parallelogram of the
            construction.

        References
        ----------
        .. [LP2013a] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transport
            of Deformations in Time Series of Images: from Schild's to
            Pole Ladder. Journal of Mathematical Imaging and Vision, Springer
            Verlag, 2013,50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩
        """
        mid_tangent_vector_to_shoot = (
            1.0 / 2.0 * metric.log(base_point=base_shoot, point=next_point)
        )

        mid_point = metric.exp(
            base_point=base_shoot, tangent_vec=mid_tangent_vector_to_shoot
        )

        tangent_vector_to_shoot = -metric.log(
            base_point=mid_point,
            point=base_point,
        )

        end_shoot = metric.exp(
            base_point=mid_point,
            tangent_vec=tangent_vector_to_shoot,
        )

        if self.save_geodesics:
            self.geodesics_.append(
                self._compute_geodesics(
                    metric, base_point, next_point, base_shoot, end_shoot
                )
            )

        return end_shoot

    def _compute_geodesics(self, metric, base_point, next_point, base_shoot, end_shoot):
        main_geodesic = metric.geodesic(initial_point=base_point, end_point=next_point)
        diagonal = metric.geodesic(initial_point=base_point, end_point=end_shoot)
        second_diagonal = metric.geodesic(
            initial_point=base_shoot, end_point=next_point
        )
        final_geodesic = metric.geodesic(initial_point=next_point, end_point=end_shoot)
        return [main_geodesic, diagonal, second_diagonal, final_geodesic]


class ParallelTransport(metaclass=ABCMeta):
    def _manipulate_input(self, metric, base_point, direction, end_point):
        if direction is None and end_point is None:
            raise ValueError(
                "Either an end_point or a tangent_vec_b must be given to define the"
                " geodesic along which to transport."
            )

        if direction is not None:
            return direction

        return metric.log(end_point, base_point)

    @abstractmethod
    def parallel_transport(
        self, metric, tangent_vec, base_point, direction=None, end_point=None
    ):
        pass


class LadderParallelTransport(ParallelTransport):
    def __init__(self, n_rungs, alpha=1, scheme="pole"):

        geomstats.errors.check_integer(n_rungs, "n_rungs")
        if alpha < 1:
            raise ValueError("alpha must be greater or equal to one")

        self.n_rungs = n_rungs
        self.alpha = alpha
        self.scheme = scheme

        # TODO: add ladder stepper

    def parallel_transport(
        self, metric, tangent_vec, base_point, direction=None, end_point=None
    ):
        # TODO: reset geodesics
        # TODO: save trajectory
        # TODO: save endpoint?

        direction = self._manipulate_input(metric, base_point, direction, end_point)

        current_point = base_point
        next_tangent_vec = tangent_vec / (self.n_rungs**self.alpha)
        base_shoot = metric.exp(base_point=current_point, tangent_vec=next_tangent_vec)

        for i_point in range(self.n_rungs):
            frac_tan_vector_b = (i_point + 1) / self.n_rungs * direction

            next_point = self.exp(base_point=base_point, tangent_vec=frac_tan_vector_b)
            base_shoot = self.ladder_stepper.step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot,
            )
            current_point = next_point

        transported_tangent_vec = self.log(base_shoot, current_point)
        transported_tangent_vec *= self.n_rungs**self.alpha
        transported_tangent_vec = self.ladder_stepper.post_proc_transported(
            self, transported_tangent_vec
        )

        return transported_tangent_vec
