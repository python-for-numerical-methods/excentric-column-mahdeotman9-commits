import numpy as np
from scipy import optimize

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    L: אורך במ"מ
    E: מודול אלסטיות ב-MPa
    A: שטח חתך בממ"ר
    r: רדיוס אינרציה במ"מ
    c: מרחק לסיב קיצוני במ"מ
    e: אקסצנטריות במ"מ
    sigma_allow: מאמץ מותר ב-MPa

    Return: העומס P בניוטון (float)
    """

    def stress_error(P):
        # Calculate the argument for np.cos
        # Add a small epsilon to the denominator to prevent division by zero or very small numbers
        # especially when P is zero. Also, handle potential negative values under sqrt
        sqrt_term_arg = P / (E * A + 1e-10)
        if sqrt_term_arg < 0: # Ensure argument for sqrt is non-negative
            return np.inf # Return a very large error if P is negative

        cos_arg = (L / (2 * r)) * np.sqrt(sqrt_term_arg)

        # Handle potential overflow in np.cos or when cos_arg is close to pi/2 + k*pi
        # If cos_arg is too close to an odd multiple of pi/2, cos will be near zero,
        # leading to a very large sec_term. This implies buckling or a very high stress.
        # We can cap the sec_term to a large but finite value or return a large error.
        # A common practice is to check for values close to critical points.
        if np.isclose(np.cos(cos_arg), 0):
            return np.inf # Indicate an infinite stress if cos is zero (buckling)

        sec_term = 1 / np.cos(cos_arg)
        
        # Calculate sigma_max
        sigma_max = (P / A) * (1 + (e * c / r**2) * sec_term)
        
        return sigma_max - sigma_allow

    # A good initial guess is crucial for Newton's method. 
    # For column buckling, an initial guess can be based on Euler buckling load for first approximation,
    # or a reasonable value based on typical loads. Let's start with a generic positive value.
    # We need to ensure that the initial guess does not fall into problematic regions (e.g., P=0 or too high).
    # Given the previous examples, 500000 N seems like a reasonable starting point if no other info is given.
    # Using bisect is more robust, but newton is faster if the initial guess is good.
    # Let's try to use newton with a reasonable initial guess.
    try:
        # The range for bisect method (lower_bound, upper_bound)
        # The lower bound can be 0, but for stress_error to be defined, P cannot be exactly 0 (or sqrt issue).
        # The upper bound should be high enough to contain the root, e.g., Euler critical load or a very high load.
        # For this problem, assuming P is positive, a lower bound slightly > 0 and a large upper bound is needed.
        # Let's use a safe range, similar to what was used in the notebook for bisect.
        # A heuristic for initial guess for newton or bounds for bisect could be (A * sigma_allow * 0.1) and (A * sigma_allow * 10)
        # However, for demonstration, 500000 seems to work for the previous example.
        # Let's refine the initial guess for Newton's method. A simple one could be `A * sigma_allow * 0.5`
        initial_guess = A * sigma_allow * 0.5 # A more informed initial guess based on material strength
        if initial_guess <= 0: # Ensure initial guess is positive
            initial_guess = 1.0 # Fallback to a small positive value

        P_critical = optimize.newton(stress_error, initial_guess)
        
        # Check if P_critical is negative, which is not physically meaningful
        if P_critical < 0:
            # If newton converges to a negative value, it indicates an issue or no positive root.
            # We can try bisect if Newton fails or gives non-physical results.
            # For now, let's just return 0 or raise an error.
            # Or, if a search range is known, bisect is safer.
            # For this problem, we expect a positive critical load.
            raise ValueError("Newton's method converged to a negative load. Try different initial guess or bisect method.")

    except RuntimeError as e:
        # Newton's method might fail to converge. Fallback to bisect if possible.
        print(f"Newton's method failed: {e}. Trying bisection method...")
        # For bisect, we need a bracket [a, b] where f(a) and f(b) have opposite signs.
        # We need to find a suitable range where stress_error changes sign.
        # A common approach is to find a lower bound where stress_error is negative (P is too low) and 
        # an upper bound where stress_error is positive (P is too high, or buckling). 
        # Let's use a broad range based on the context's example (0 to 1,000,000 or even higher)
        # We need to ensure stress_error(lower_bound) < 0 and stress_error(upper_bound) > 0
        
        # Attempt to find a suitable bracket for bisect:
        lower_bound = 1.0 # P must be positive
        upper_bound = 2.0 * A * sigma_allow # A very high load, usually higher than expected critical load
        
        # Make sure the function changes sign in this range.
        # If stress_error(lower_bound) is already positive, then the root might be very close to 0.
        # If stress_error(upper_bound) is negative, then the upper bound is too low.
        
        # Let's find a more robust upper bound if the initial one doesn't work
        if stress_error(lower_bound) >= 0:
             # If stress is already too high at lower_bound, the root is even lower (or no positive root).
            # This case might indicate an issue with problem setup or very small P_critical.
            # For this problem, P should be > 0. Let's assume a small positive value is the root for this edge case.
            # Or, we can expand the search for a negative stress_error.
            pass # This case needs careful handling or domain specific knowledge.
        
        # Try to find an upper bound where stress_error is positive (exceeds sigma_allow)
        # Start from initial_guess and double it until stress_error becomes positive or a max is reached.
        current_upper_bound = initial_guess * 2.0
        max_search_upper_bound = 1e9 # Prevent infinite loop
        while stress_error(current_upper_bound) < 0 and current_upper_bound < max_search_upper_bound:
            current_upper_bound *= 2.0
            
        if stress_error(lower_bound) < 0 and stress_error(current_upper_bound) > 0:
            P_critical = optimize.bisect(stress_error, lower_bound, current_upper_bound)
        else:
            raise ValueError("Could not find a suitable bracket for bisection, or no positive root exists.")
            
    return P_critical

