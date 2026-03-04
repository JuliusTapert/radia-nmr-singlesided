import numpy as np


# ------------------------------------------------------------------
# Equation (2): Compute B1c (perpendicular circular component vector)
# ------------------------------------------------------------------

def compute_B1c(B0, B1):
    """
    Compute B1c according to:

    B1c = 1/2 [ B1 - B0 (B1·B0)/(B0·B0) ]

    Parameters
    ----------
    B0 : ndarray (..., 3)
    B1 : ndarray (..., 3)

    Returns
    -------
    B1c : ndarray (..., 3)
    """

    B0_dot_B0 = np.sum(B0 * B0, axis=-1)
    B0_dot_B0_safe = np.where(B0_dot_B0 == 0, 1.0, B0_dot_B0) # Had a zero division error oops, what are the odds honestly

    B1_dot_B0 = np.sum(B1 * B0, axis=-1)

    projection = (B1_dot_B0[..., None] / B0_dot_B0_safe[..., None]) * B0
    B1c = 0.5 * (B1 - projection)

    return B1c


# ------------------------------------------------------------------
# Equation (4): m_asy term (CPMG response)
# ------------------------------------------------------------------

def compute_masy(Domega0, omega1, Big_omega, t90, t180, te):
    """
    Compute masy according to Eq. (4)
    """

    #eps = 1e-20

    #omega1_safe = np.where(np.abs(omega1) < eps, eps, omega1)
    #Big_omega_safe = np.where(np.abs(Big_omega) < eps, eps, Big_omega)

    term1 = (Big_omega / omega1) * (
        np.sin(Domega0 * te / 2) *
        1/(np.tan(Big_omega * t180 / 2))
    )

    term2 = (Domega0 / omega1) * np.cos(Domega0 * te / 2)

    denominator = 1 + (term1 + term2) ** 2

    masy = (omega1 / Big_omega) * np.sin(Big_omega * t90) / denominator

    return masy


# ------------------------------------------------------------------
# Equation (3): Sensitivity / Signal Map
# ------------------------------------------------------------------

def compute_cpmg_signal(
    B0,
    B1,
    I=4.0,
    t90=None,
    t180=None,
    te=150e-6,
    gamma_bar=42.58e6,   # Hz/T
    chi=4e-9,
    Q=20,
    voxel_size=2.5e-7,  # m^2
):
    """

    Parameters
    ----------
    B0 : ndarray (..., 3)
    B1 : ndarray (..., 3)
    I : float
        Current used to generate B1 field
    t90, t180 : float
        Pulse durations (seconds)
    te : float
        Echo spacing (seconds)
    gamma_bar : float
        Gyromagnetic ratio in Hz/T
    chi : float
        Magnetic susceptibility
    Q : float
        Coil quality factor
    voxel_size : float
        Voxel size for spatial integration

    Returns
    -------
    signal_map : ndarray (complex)
    """

    mu0 = 4e-7 * np.pi
    gamma = 2 * np.pi * gamma_bar  # rad/s/T

    # Magnitude of B0
    B0_mag = np.linalg.norm(B0, axis=-1)

    # Compute B1c vector and magnitude
    B1c_vec = compute_B1c(B0, B1)
    B1c_mag = np.linalg.norm(B1c_vec, axis=-1)

    # Reference RF frequency
    f_rf = gamma_bar * np.median(B0_mag)

    # Default pulse durations
    if t180 is None:
        B1_ref = np.median(B1c_mag)
        t180 = np.pi / (gamma * B1_ref)
    if t90 is None:
        t90 = t180 / 2

    # Frequency offsets
    Domega0 =  gamma * B0_mag - (2 * np.pi * f_rf)
    omega1 = gamma * B1c_mag
    Big_omega = np.sqrt(Domega0**2 + omega1**2)

    # Compute masy
    masy = compute_masy(Domega0, omega1, Big_omega, t90, t180, te)
    #masy = omega1 * t90

    # Resonator transfer function F(Δω0)
    f = gamma_bar * B0_mag
    F = Q / (1 + 1j * Q * ((f / f_rf) - (f_rf / f)))

    # Equation (3) scalar components:
    # γ B0 · (χ/μ0) B0 · (B1c/I) · F · masy

    signal_map = (
        chi / mu0
        * B0_mag**2
        * gamma
        * (B1c_mag / I)
        * F
        * masy
        * voxel_size
    )

    signal_map = np.nan_to_num(signal_map)
    #signal_map = np.sum(signal_map)
    print("abs(signal_map) range:",
        np.min(np.abs(signal_map)),
        np.max(np.abs(signal_map)))
    print("abs(masy) range:",
        np.min(np.abs(masy)),
        np.max(np.abs(masy)))
    return signal_map
