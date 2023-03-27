from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.numerics.bvp_solvers import ScipySolveBVP
from geomstats.numerics.ivp_solvers import GSIntegrator
from geomstats.numerics.optimizers import ScipyMinimize


class ExpSolver(ABC):
    @abstractmethod
    def exp(self, space, tangent_vec, base_point):
        pass

    @abstractmethod
    def geodesic_ivp(self, space, tangent_vec, base_point, t):
        pass


class ExpIVPSolver(ExpSolver):
    def __init__(self, integrator=None):
        if integrator is None:
            integrator = GSIntegrator()

        self.integrator = integrator

    def _solve(self, space, tangent_vec, base_point, t_eval=None):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        if self.integrator.state_is_raveled:
            initial_state = gs.hstack([base_point, tangent_vec])
        else:
            initial_state = gs.stack([base_point, tangent_vec])

        force = self._get_force(space)
        if t_eval is None:
            return self.integrator.integrate(force, initial_state)

        return self.integrator.integrate_t(force, initial_state, t_eval)

    def exp(self, space, tangent_vec, base_point):
        result = self._solve(space, tangent_vec, base_point)
        return self._simplify_exp_result(result, space)

    def geodesic_ivp(self, space, tangent_vec, base_point):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)
        t_axis = int(tangent_vec.ndim > space.point_ndim)

        def path(t):
            # TODO: likely unwanted behavior
            squeeze = False
            if not gs.is_array(t):
                t = gs.array([t])
                squeeze = True

            result = self._solve(space, tangent_vec, base_point, t_eval=t)
            result = self._simplify_result_t(result, space)
            if squeeze:
                return gs.squeeze(result, axis=t_axis)

            return result

        return path

    def _get_force(self, space):
        if self.integrator.state_is_raveled:
            force_ = lambda state, t: self._force_raveled_state(state, t, space=space)
        else:
            force_ = lambda state, t: self._force_unraveled_state(state, t, space=space)

        if self.integrator.tfirst:
            return lambda t, state: force_(state, t)

        return force_

    def _force_raveled_state(self, raveled_initial_state, _, space):
        # input: (n,)

        # assumes unvectorize
        state = gs.reshape(raveled_initial_state, (space.dim, space.dim))

        # TODO: remove dependency on time in `geodesic_equation`?
        eq = space.metric.geodesic_equation(state, _)

        return gs.flatten(eq)

    def _force_unraveled_state(self, initial_state, _, space):
        return space.metric.geodesic_equation(initial_state, _)

    def _simplify_exp_result(self, result, space):
        y = result.y[-1]

        if self.integrator.state_is_raveled:
            return y[..., : space.dim]

        return y[0]

    def _simplify_result_t(self, result, space):
        # TODO: need to verify

        # assumes several t
        y = result.y

        if self.integrator.state_is_raveled:
            y = y[..., : space.dim]
            if gs.ndim(y) > 2:
                return gs.moveaxis(y, 0, 1)
            return y

        y = y[:, 0, :, ...]
        if gs.ndim(y) > 2:
            return gs.moveaxis(y, 1, 0)
        return y


class LogSolver(ABC):
    # TODO: check private methods in children
    @abstractmethod
    def log(self, space, point, base_point):
        pass


class LogShootingSolver(LogSolver):
    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.flatten(gs.random.rand(*base_point.shape))

    def _objective(self, velocity, space, point, base_point):
        velocity = gs.reshape(velocity, base_point.shape)
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def log(self, space, point, base_point):
        # TODO: are we sure optimizing together is a good idea?
        # TODO: create alternative vectorization case

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec)

        tangent_vec = gs.reshape(res.x, base_point.shape)

        return tangent_vec


class LogShootingSolverUnflatten(LogSolver):
    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.random.rand(*base_point.shape)

    def _objective(self, velocity, space, point, base_point):
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def _log(self, space, point, base_point):
        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec)

        return res.x

    def log(self, space, point, base_point):
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            return self._log(space, point, base_point)

        return gs.stack(
            [self._log(space, point_, base_point_) for point_, base_point_ in zip(point, base_point)]
        )


class LogBVPSolver(LogSolver):
    def __init__(self, n_segments=1000, integrator=None, initialization=None):
        # TODO: add more control on the discretization?
        if integrator is None:
            integrator = ScipySolveBVP()

        if initialization is None:
            initialization = self._default_initialization

        # TODO: rename (more segments than nodes) #
        self.n_segments = n_segments
        self.integrator = integrator
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point, n_segments):
        # TODO: receive discretization instead?
        dim = space.dim
        point_0, point_1 = base_point, point

        # TODO: need to update torch linspace #
        # TODO: need to avoid assignment #

        mesh = gs.transpose(gs.linspace(point_0, point_1, n_segments))
        predictions = n_segments * (mesh[:,1:] - mesh[:,:-1])
        predictions = gs.hstack((predictions, gs.array(predictions[:,-2:-1])))
        lin_init = gs.vstack((mesh,predictions))
        return lin_init

    def boundary_condition(self, state_0, state_1, space, point_0, point_1):
        pos_0 = state_0[:space.dim]
        pos_1 = state_1[:space.dim]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    def bvp(self, _, raveled_state, space):
        # inputs: n (2*dim) , n_nodes

        # assumes unvectorized

        state = gs.moveaxis(
            gs.reshape(raveled_state, (2, space.dim, -1)), -2, -1
        )

        eq = space.metric.geodesic_equation(state, _)

        eq = gs.reshape(gs.moveaxis(eq, -2, -1), (2 * space.dim, -1))
        
        return eq

    def log(self, space, point, base_point):
        # TODO: vectorize #
        # TODO: assume known jacobian

        all_results = []

        point, base_point = gs.broadcast_arrays(point, base_point)
        if point.ndim == 1:
            point = gs.expand_dims(point, axis=0)
            base_point = gs.expand_dims(base_point, axis=0)

        for i in range(point.shape[0]):
            bvp = lambda t, state: self.bvp(t, state, space)
            bc = lambda state_0, state_1: self.boundary_condition(
                state_0, state_1, space, base_point[i], point[i]
            )

            x = gs.linspace(0.0, 1.0, self.n_segments)
            y = self.initialization(space, point[i], base_point[i], self.n_segments)
            
            result = self.integrator.integrate(bvp, bc, x, y)
            all_results.append(result)

        return gs.squeeze(gs.vstack([self._simplify_result(result, space) for result in all_results]), axis=0)

    def geodesic_bvp(self, space, point, base_point):
        all_results = []

        point, base_point = gs.broadcast_arrays(point, base_point)
        if point.ndim == 1:
            point = gs.expand_dims(point, axis=0)
            base_point = gs.expand_dims(base_point, axis=0)

        for i in range(point.shape[0]):
            bvp = lambda t, state: self.bvp(t, state, space)
            bc = lambda state_0, state_1: self.boundary_condition(
                state_0, state_1, space, base_point[i], point[i]
            )

            x = gs.linspace(0.0, 1.0, self.n_segments)
            y = self.initialization(space, point[i], base_point[i], self.n_segments)
            
            result = self.integrator.integrate(bvp, bc, x, y)
            all_results.append(result)

        def path(t):
            y_t = gs.array([result.sol(t)[:1] for result in all_results])
            return gs.expand_dims(gs.squeeze(y_t, axis=-2), axis=-1)
            
        return path

    def _simplify_result(self, result, space):
        _, tangent_vec = gs.reshape(gs.transpose(result.y)[0], (2, space.dim))

        return tangent_vec
    
# class LogPolynomialSolver(LogSolver):
# use minimize in optimize

