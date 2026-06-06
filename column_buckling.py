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
    # כתבו כאן את הקוד
import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    def f(P):
        if P <= 0:
            return -sigma_allow
        
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # sec(x) = 1 / cos(x)
        sigma_max = (P / A) * (1 + (e * c / (r**2)) * (1 / np.cos(angle)))
        
        return sigma_max - sigma_allow

    # חסם עליון מבוסס על עומס אוילר התיאורטי
    p_euler = (np.pi*2 * E * (A * r2)) / (L*2)
    
    return float(bisect(f, 0.01, p_euler * 0.99, xtol=1e-4))
