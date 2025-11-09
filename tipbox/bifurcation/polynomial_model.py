"""
Random polynomial dynamical system generator.

Following the approach from:
Bury et al. (2021) "Deep learning for early warning signals of tipping points"
PNAS 118 (39) e2106140118

Generates 2D polynomial systems of the form:
    dx/dt = a1 + a2*x + a3*y + a4*x^2 + a5*xy + a6*y^2 + a7*x^3 + a8*x^2*y + a9*xy^2 + a10*y^3
    dy/dt = b1 + b2*x + b3*y + b4*x^2 + b5*xy + b6*y^2 + b7*x^3 + b8*x^2*y + b9*xy^2 + b10*y^3
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PolynomialModel:
    """Container for polynomial, bifurcation-induced dynamical system"""
    pars_a: jnp.ndarray  # 10 parameters for x dynamics
    pars_b: jnp.ndarray  # 10 parameters for y dynamics
    s0: jnp.ndarray      # Initial condition [x, y]
    equi: jnp.ndarray    # Equilibrium point [x, y]
    recovery_rate: float  # Magnitude of dominant eigenvalue real part
    
    def get_parameters(self) -> jnp.ndarray:
        """Get all 20 parameters as single array"""
        return jnp.concatenate([self.pars_a, self.pars_b])


@jax.jit
def polynomial_basis(state: jnp.ndarray) -> jnp.ndarray:
    """
    Compute polynomial basis functions up to 3rd order.
    
    Parameters
    ----------
    state : array [x, y]
        Current state
        
    Returns
    -------
    array of 10 basis functions: [1, x, y, x^2, xy, y^2, x^3, x^2*y, xy^2, y^3]
    """
    x, y = state[0], state[1]
    return jnp.array([1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3])


@jax.jit
def compute_jacobian(state: jnp.ndarray, pars_a: jnp.ndarray, pars_b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Jacobian matrix at given state.
    
    Parameters
    ----------
    state : array [x, y]
    pars_a : parameters for x dynamics (10 elements)
    pars_b : parameters for y dynamics (10 elements)
    
    Returns
    -------
    2x2 Jacobian matrix
    """
    x, y = state[0], state[1]
    
    # df/dx 
    j11 = pars_a[1] + 2*pars_a[3]*x + pars_a[4]*y + 3*pars_a[6]*x**2 + 2*pars_a[7]*x*y + pars_a[8]*y**2
    # df/dy
    j12 = pars_a[2] + pars_a[4]*x + 2*pars_a[5]*y + pars_a[7]*x**2 + 2*pars_a[8]*x*y + 3*pars_a[9]*y**2
    # dg/dx
    j21 = pars_b[1] + 2*pars_b[3]*x + pars_b[4]*y + 3*pars_b[6]*x**2 + 2*pars_b[7]*x*y + pars_b[8]*y**2
    # dg/dy
    j22 = pars_b[2] + pars_b[4]*x + 2*pars_b[5]*y + pars_b[6]*x**2 + 2*pars_b[8]*x*y + 3*pars_b[9]*y**2
    
    return jnp.array([[j11, j12], [j21, j22]])


@jax.jit
def evaluate_dynamics(state: jnp.ndarray, pars_a: jnp.ndarray, pars_b: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate dynamics [dx/dt, dy/dt] at given state.
    
    Parameters
    ----------
    state : array [x, y]
    pars_a : parameters for x dynamics (10 elements)
    pars_b : parameters for y dynamics (10 elements)
    
    Returns
    -------
    array [dx/dt, dy/dt]
    """
    polys = polynomial_basis(state)
    dxdt = jnp.dot(pars_a, polys)
    dydt = jnp.dot(pars_b, polys)
    return jnp.array([dxdt, dydt])


def simulate_to_equilibrium(
    pars_a: jnp.ndarray,
    pars_b: jnp.ndarray,
    s0: jnp.ndarray,
    dt: float = 0.01,
    max_time: float = 100.0,
    convergence_threshold: float = 1e-8
) -> Tuple[jnp.ndarray, bool, jnp.ndarray]:
    """
    Simulate system to find equilibrium point.
    
    Parameters
    ----------
    pars_a, pars_b : parameter arrays
    s0 : initial condition [x, y]
    dt : time step
    max_time : maximum simulation time
    convergence_threshold : threshold for convergence detection
    
    Returns
    -------
    equilibrium : final state [x, y]
    converged : whether system converged
    trajectory : full trajectory for inspection
    """
    n_steps = int(max_time / dt)
    
    @jax.jit
    def integrate(traj):
        """Run integration loop - JIT compiled for speed"""
        def body_fun(i, traj):
            state = traj[i]
            dynamics = evaluate_dynamics(state, pars_a, pars_b)
            new_state = state + dynamics * dt
            return traj.at[i + 1].set(new_state)
        
        return jax.lax.fori_loop(0, n_steps - 1, body_fun, traj)
    
    # Simulate
    traj = jnp.zeros((n_steps, 2))
    traj = traj.at[0].set(s0)
    traj = integrate(traj)
    
    # Check convergence
    # Difference between max and min of last 10 data points
    last_points = traj[-10:]
    diff = jnp.max(last_points, axis=0) - jnp.min(last_points, axis=0)
    norm = jnp.sqrt(jnp.sum(jnp.square(diff)))

    # Check amplitude doesn't blow up
    max_amplitude = jnp.max(jnp.abs(traj))
    converged = (norm < convergence_threshold) and (max_amplitude < 1e3)
    
    # Return equilibrium point, convergence status, and trajectory
    return traj[-1], converged, traj


def generate_random_polynomial_model(
    key: jax.random.PRNGKey,
    sparsity: float = 0.5,
    param_scale: float = 1.0,
    max_attempts: int = 100
) -> Optional[PolynomialModel]:
    """
    Main entry point to generate a random polynomial model with stable equilibrium.
    
    This follows the Bury et al. (2021) paper approach:
        1. Generate random parameters from N(0, 1)
        2. Set random subset to zero (sparsity)
        3. Negate high-order terms (x^3, y^3 terms) for boundedness
        4. Simulate to find equilibrium given a set of checks: convergence, boundedness, and eigenvalues
    
    Parameters
    ----------
    key : JAX random key
    sparsity : fraction of parameters to set to zero
    param_scale : scale of normal distribution for parameters
    max_attempts : maximum number of random models to try
    
    Returns
    -------
    PolynomialModel or None if no suitable model found
    """
    
    for attempt in range(max_attempts):
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)
        
        # Generate parameters from normal distribution
        pars = jax.random.normal(subkey1, (20,)) * param_scale
        
        # Apply sparsity: randomly set parameters to zero
        n_zero = int(20 * sparsity)
        zero_indices = jax.random.choice(subkey2, 20, (n_zero,), replace=False)
        pars = pars.at[zero_indices].set(0.0)
        
        # Negate high-order terms for boundedness (indices 6-9 for a, 16-19 for b)
        # These correspond to x^3, x^2*y, xy^2, y^3 terms
        pars = pars.at[6:10].set(-jnp.abs(pars[6:10]))
        pars = pars.at[16:20].set(-jnp.abs(pars[16:20]))
        
        pars_a = pars[:10]
        pars_b = pars[10:]
        
        # Random initial condition
        s0 = jax.random.normal(subkey3, (2,)) * 2.0
        
        # Simulate to equilibrium
        equi, converged, traj = simulate_to_equilibrium(pars_a, pars_b, s0)
        
        if not converged:
            continue
        
        # Check for NaN or Inf
        if jnp.any(jnp.isnan(equi)) or jnp.any(jnp.isinf(equi)):
            continue
        
        # Check if equilibrium is bounded
        if jnp.max(jnp.abs(equi)) > 1e3:
            continue
        
        # Compute Jacobian and eigenvalues at equilibrium using helper function
        jac = compute_jacobian(equi, pars_a, pars_b)
        evals = jnp.linalg.eigvals(jac)
        
        # Compute recovery rate (magnitude of real part of dominant eigenvalue)
        real_parts = jnp.real(evals)
        dom_eval_re = jnp.max(real_parts)
        recovery_rate = float(jnp.abs(dom_eval_re))
        
        # Check if stable (all real parts negative)
        if jnp.all(real_parts < 0):
            return PolynomialModel(
                pars_a=pars_a,
                pars_b=pars_b,
                s0=s0,
                equi=equi,
                recovery_rate=recovery_rate
            )
    
    # No suitable model found
    return None
