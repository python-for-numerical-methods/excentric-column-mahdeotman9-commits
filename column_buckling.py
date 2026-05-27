import numpy as np
from scipy import optimize

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    Calculates the critical load P for a column using the Secant Formula
    and numerical root-finding methods (Newton-Raphson with bisection fallback).

    Args:
        L (float): Column length in mm.
        E (float): Modulus of Elasticity in MPa.
        A (float): Cross-sectional area in mm^2.
        r (float): Radius of gyration in mm.
        c (float): Distance from neutral axis to extreme fiber in mm.
        e (float): Load eccentricity in mm.
        sigma_allow (float): Allowable stress in MPa.

    Returns:
        float: The critical load P in Newtons.
    """

    # Create an objective function for the root-finding algorithm.
    # This function takes only P as a variable, with other parameters fixed.
    objective_function = lambda P_val: column_stress_error_assignment(P_val, L, E, A, r, c, e, sigma_allow)

    # Calculate Euler buckling load (P_euler) for an ideal column.
    # P_euler = (pi^2 * E * I) / L^2, where I = A * r^2
    # This serves as a theoretical upper bound for the critical load.
    P_euler = (np.pi**2 * E * A * r**2) / (L**2)

    # Initial guess for Newton's method.
    # A common approach is to use a value slightly less than the Euler buckling load,
    # as the critical load for eccentric loading will generally be lower than P_euler.
    initial_guess_P = P_euler * 0.8

    # Ensure the initial guess is positive. Fallback to a small positive value if P_euler is non-positive or very small.
    if initial_guess_P <= 0:
        initial_guess_P = 1.0

    try:
        # Attempt to use Newton-Raphson method for faster convergence.
        P_critical = optimize.newton(objective_function, initial_guess_P)
    except RuntimeError:
        # If Newton's method fails to converge, fall back to the more robust Bisection method.
        # Bisection requires an interval [a, b] where f(a) and f(b) have opposite signs.

        # Lower bound 'a' can be 0 (load cannot be negative for this problem).
        a = 0.0

        # Upper bound 'b' is set generously above P_euler to ensure it brackets the root.
        # The true critical load for the secant formula is always less than or equal to P_euler,
        # but giving a slightly larger range ensures robustness in edge cases or numerical inaccuracies.
        b = P_euler * 1.5 # Start with 150% of Euler load as a generous upper bound for bisection

        # Verify that objective_function(a) is negative and objective_function(b) is positive
        # for bisection to work. objective_function(0) should be -sigma_allow (if sigma_allow > 0).
        if objective_function(a) >= 0 and sigma_allow > 0:
             raise ValueError("Lower bound 'a' for bisection does not result in negative stress error. Check parameters or function definition.")
        elif sigma_allow <= 0:
            # If allowable stress is non-positive, P=0 might already meet or exceed it.
            # This case might need special handling or implies invalid input for structural design.
            return 0.0

        # Adjust 'b' dynamically if the initial upper bound is not high enough
        # to make objective_function(b) positive. This can happen for very specific parameter combinations.
        while objective_function(b) <= 0:
            b *= 2
            # Prevent infinite loop by setting a practical upper limit, e.g., 100 times Euler load
            # or a very large absolute number (1e12).
            if b > P_euler * 100 and b > 1e12:
                raise ValueError("Could not find an appropriate upper bound for bisection method. Consider a wider range or different initial P.")

        # Perform bisection to find the root within the determined interval.
        P_critical = optimize.bisect(objective_function, a, b)

    return P_critical

