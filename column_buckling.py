%%writefile column_buckling.py
import numpy as np
from scipy import optimize

def column_stress_error(P, L, E, A, r, c, e, sigma_allow):
    arg = (L / (2 * r)) * np.sqrt(P / (E * A))
    # Handle cases where P is too large and the argument to cos approaches pi/2 + n*pi
    # which would lead to an overflow or division by zero.
    # Check if arg is close to an odd multiple of pi/2
    if np.isclose(np.cos(arg), 0):
        return np.inf # Stress is effectively infinite

    sec_term = 1 / np.cos(arg)
    sigma_max = (P / A) * (1 + (e * c / r**2) * sec_term)
    return sigma_max - sigma_allow

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    func_to_solve = lambda P: column_stress_error(P, L, E, A, r, c, e, sigma_allow)
    
    # Use a range for bisection to find a safe starting point for newton, if needed
    # For simplicity and given newton's robustness with a good guess, we'll stick to a direct call
    # An initial guess of 500,000 N is used as it worked well in the previous examples.
    try:
        P_critical = optimize.newton(func_to_solve, 500000.0) # Initial guess
    except RuntimeError as e:
        # If newton fails, try bisection for a more robust (but slower) solution
        # Need to define a search bracket where the function changes sign
        # A common approach is to search from a small positive load up to a theoretical max (e.g., Euler load)
        # For this example, I will define a default broad range.
        # The lower bound must be > 0 because P=0 would cause division by zero in the `arg` calculation.
        # The upper bound should be sufficiently high to encompass the expected critical load.
        # If `newton` fails, `bisect` needs a bracket where signs are different.
        # This can be tricky to define universally without prior knowledge of the function behavior.
        # For this problem, let's assume P should be positive and less than some large number.
        
        # A safer strategy for `newton`'s failure is to perhaps refine the initial guess
        # or to provide better bounds based on the problem's physical constraints.
        print(f"Newton's method failed: {e}. Attempting bisection...")
        try:
            # Try to find a bracket using some heuristic or a wider range
            # For example, from 1N to A * sigma_allow * some_factor
            p_low = 1.0
            # A * sigma_allow is the load for pure compression without buckling
            # We need a higher upper bound for the secant formula which can have higher loads initially
            p_high = A * sigma_allow * 10 # heuristic upper bound
            if func_to_solve(p_low) * func_to_solve(p_high) < 0:
                P_critical = optimize.bisect(func_to_solve, p_low, p_high)
            else:
                print("Could not find a valid bracket for bisection. Returning NaN.")
                return np.nan # Or raise an error
        except Exception as bisect_e:
            print(f"Bisection also failed: {bisect_e}. Returning NaN.")
            return np.nan # Or raise an error

    # Ensure the critical load is positive. A physical load cannot be negative.
    if P_critical < 0:
        return np.nan # Or handle as an error
    
    return P_critical

