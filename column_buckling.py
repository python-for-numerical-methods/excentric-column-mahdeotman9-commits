
def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    L: אורך במ"מ
    E: מודול אלסטיות ב-MPa
    A: שטח חתך בממ"ר
    r: רדיוס אינרציה במ"מ
    c: מרחק לסיב קיצוני בממ"מ
    e: אקסצנטריות במ"מ
    sigma_allow: מאמץ מותר ב-MPa

    Return: העומס P בניוטון (float)
    """
    # A good initial guess for P is crucial for newton's method.
    # For structural problems, a fraction of the Euler buckling load can be a starting point.
    # Or, we can use a reasonable estimate like P=1 N to ensure positive values, or P_allow * A, etc.
    # The problem description provided an initial guess of 500000 N earlier, which can be adapted.

    # Define the function for scipy.optimize to find the root.
    # It fixes all parameters except P, which is the variable to be found.
    func_to_solve = lambda P: column_stress_error(P, L, E, A, r, c, e, sigma_allow)

    # Find the root using Newton's method. Initial guess is important.
    # A simple positive guess like 1.0 or a more informed guess can be used.
    # Given the previous context, 500000 N seems to be a reasonable starting point for many cases.
    P_critical = optimize.newton(func_to_solve, 500000.0) # Using 500000.0 as initial guess as in prior example

    return P_critical
