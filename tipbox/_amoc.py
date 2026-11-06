"""Shared physical parameters and numerical kernels for the two-state AMOC model."""

from dataclasses import dataclass

import jax.numpy as jnp

from .utils import parameter, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class AMOCPhysics:
    """Calibrated 2×CO₂ box-model parameters; salinities are mass fractions.

    The states are 100 times salinity anomalies. Their drift therefore includes
    100 times the seconds-per-year conversion. Model time is years; box volumes
    use m³ and transports use m³/s.
    """

    S0: float = parameter(0.035, "Reference salinity mass fraction.", "mass fraction")
    alpha: float = parameter(0.12, "Thermal-density coefficient.", "kg / m^3 / K")
    beta: float = parameter(
        790.0, "Haline-density coefficient per salinity mass fraction.", "kg / m^3"
    )
    seconds_per_year: float = parameter(
        31500000.0,
        "Seconds per model year; the separate factor100 converts salinity fractions to anomaly coordinates.",
        "s / year",
    )
    V_N: float = parameter(3.683e16, "North Atlantic box volume.", "m^3")
    V_T: float = parameter(5.418e16, "Tropical Atlantic box volume.", "m^3")
    V_S: float = parameter(6.097e16, "Southern Ocean box volume.", "m^3")
    V_IP: float = parameter(1.486e17, "Indo-Pacific box volume.", "m^3")
    V_B: float = parameter(9.925e17, "Bottom-water box volume.", "m^3")
    T_S: float = parameter(
        7.919, "Southern reference temperature in the transport closure.", "degC"
    )
    T_0: float = parameter(
        3.87, "North Atlantic reference temperature in the transport closure.", "degC"
    )
    total_salt: float = parameter(
        4.4735e16, "Conserved salinity-volume integral.", "m^3"
    )
    lambda_q: float = parameter(
        16200000.0, "Density-to-volume-transport coefficient.", "m^6 / kg / s"
    )
    gamma: float = parameter(
        0.36, "Southern contribution to northward advective inflow.", "dimensionless"
    )
    mu: float = parameter(
        2.2e-07,
        "Temperature-transport coupling in the transport denominator.",
        "K s / m^3",
    )
    K_N: float = parameter(1762000.0, "Northern gyre exchange transport.", "m^3 / s")
    K_S: float = parameter(1872000.0, "Southern gyre exchange transport.", "m^3 / s")
    S_S: float = parameter(0.034427, "Fixed Southern Ocean salinity.", "mass fraction")
    S_B: float = parameter(0.034538, "Fixed bottom-water salinity.", "mass fraction")
    F_N0: float = parameter(0.486, "Northern freshwater flux at zero hosing.", "Sv")
    F_NH: float = parameter(
        0.1311, "Northern freshwater-flux response to hosing.", "dimensionless"
    )
    F_T0: float = parameter(-0.997, "Tropical freshwater flux at zero hosing.", "Sv")
    F_TH: float = parameter(
        0.6961, "Tropical freshwater-flux response to hosing.", "dimensionless"
    )


def strength_sv(SN, p: AMOCPhysics):
    """AMOC transport in Sverdrups from North Atlantic salinity anomaly."""
    south = p.S_S - p.S0
    return (
        p.lambda_q
        * (p.alpha * (p.T_S - p.T_0) + p.beta * (SN / 100 - south))
        / (1 + p.lambda_q * p.alpha * p.mu)
        * 1e-6
    )


def drift(state, H, p: AMOCPhysics):
    """Salinity-anomaly drift, preserving the calibrated sign-dependent transport."""
    SN, ST = state
    SS, SB = (p.S_S - p.S0) * 100, (p.S_B - p.S0) * 100
    SIP = (
        100
        * (
            p.total_salt
            - (p.V_N * SN + p.V_T * ST + p.V_S * SS + p.V_B * SB) / 100
            - p.S0 * (p.V_B + p.V_N + p.V_T + p.V_IP + p.V_S)
        )
        / p.V_IP
    )
    q = strength_sv(SN, p) * 1e6
    aq = jnp.abs(q)
    FN, FT = (p.F_N0 + p.F_NH * H) * 1e6, (p.F_T0 + p.F_TH * H) * 1e6
    north_exchange = p.K_N * (ST - SN) / 100
    tropical_exchange = p.K_S * (SS - ST) / 100 + p.K_N * (SN - ST) / 100
    north_adv = jnp.where(q >= 0, q * (ST - SN) / 100, aq * (SB - SN) / 100)
    tropical_adv = jnp.where(
        q >= 0,
        q * (p.gamma * SS + (1 - p.gamma) * SIP - ST) / 100,
        aq * (SN - ST) / 100,
    )
    return jnp.array(
        [
            100 * p.seconds_per_year / p.V_N * (north_adv + north_exchange - FN * p.S0),
            100
            * p.seconds_per_year
            / p.V_T
            * (tropical_adv + tropical_exchange - FT * p.S0),
        ]
    )


def rk4_step(state, h_cur, h_mid, h_next, dt, physics):
    """RK4 drift update; diffusion is applied separately by the caller."""
    k1 = drift(state, h_cur, physics)
    k2 = drift(state + 0.5 * dt * k1, h_mid, physics)
    k3 = drift(state + 0.5 * dt * k2, h_mid, physics)
    k4 = drift(state + dt * k3, h_next, physics)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def observe(path, physics):
    """Observed transport followed by the two physical salinity-anomaly states."""
    return jnp.column_stack([strength_sv(path[:, 0], physics), path])
