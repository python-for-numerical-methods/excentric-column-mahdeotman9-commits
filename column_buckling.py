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

    # Initial guess for Newton's method. This is crucial for convergence.
    # Based on examples, 500,000 N seems like a reasonable starting point for many cases.
    initial_guess_P = 500000.0

    try:
        # Attempt to use Newton-Raphson method for faster convergence.
        P_critical = optimize.newton(objective_function, initial_guess_P)
    except RuntimeError:
        # If Newton's method fails to converge, fall back to the more robust Bisection method.
        # Bisection requires an interval [a, b] where f(a) and f(b) have opposite signs.

        # We know objective_function(0) = -sigma_allow (assuming sigma_allow > 0),
        # so the lower bound 'a' can be 0.
        a = 0.0

        # Find an upper bound 'b' where the objective_function returns a positive value.
        # Start with the initial_guess_P and double it until f(b) > 0 or a very large limit is reached.
        b = initial_guess_P
        # Ensure the lower bound is actually lower than the upper bound.
        # If initial_guess_P leads to a positive error, we need to adjust b down.
        # However, it's more common for a first guess to be too low or result in a negative error.
        while objective_function(b) <= 0:
            b *= 2
            if b > 1e12:  # Prevent infinite loop in extreme cases
                raise ValueError("Could not find an appropriate upper bound for bisection method.")

        # Perform bisection to find the root within the determined interval.
        P_critical = optimize.bisect(objective_function, a, b)

    return P_critical

