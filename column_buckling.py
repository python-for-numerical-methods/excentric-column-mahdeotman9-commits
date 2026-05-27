import numpy as np
from scipy.optimize import bisect

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
    
    # 1. הגדרת פונקציית העזר שאת השורש שלה נחפש
    def f(P):
        if P <= 0:
            return -sigma_allow
        
        # חישוב הארגומנט שבתוך הקוסינוס (ברדיאנים)
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # חישוב המאמץ המקסימלי בסיב הקיצוני (החלפת secant ב- 1/cos)
        sigma_max = (P / A) * (1 + (e * c / (r**2)) * (1.0 / np.cos(angle)))
        
        # החזרת ההפרש מהמאמץ המותר
        return sigma_max - sigma_allow

    # 2. חישוב חסם עליון מבוסס על עומס אוילר התיאורטי (I = A * r^2)
    p_euler = (np.pi**2 * E * (A * r**2)) / (L**2)
    
    # לוקחים 99.9% מעומס אוילר כדי להימנע מאינסוף/חלוקה באפס בתוך הקוסינוס
    p_min = 0.0
    p_max = 0.999 * p_euler
    
    # 3. הפעלת שיטת החצייה. xtol=1e-4 מבטיח דיוק גבוה בהרבה מהנדרש בטסטים (10^-3)
    p_critical = bisect(f, p_min, p_max, xtol=1e-4)
    
    return float(p_critical)
