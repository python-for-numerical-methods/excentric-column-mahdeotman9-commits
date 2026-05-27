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
    
    # 1. הגדרת פונקציית המטרה שאת השורש שלה נרצה למצוא
    def objective_function(P):
        # חישוב האיבר בתוך ה-cos (ברדיאנים)
        # שימו לב: הארגומנט הוא (L / (2 * r)) * sqrt(P / (E * A))
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # חישוב המאמץ המקסימלי לפי נוסחת הסקנט
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(angle)))
        
        # נרצה ש-sigma_max יהיה שווה ל-sigma_allow, לכן נחזיר את ההפרש
        return sigma_max - sigma_allow

    # 2. קביעת גבולות לחיפוש הנומרי (Bracketing)
    # הגבול התחתון הוא עומס קרוב לאפס (אך לא אפס מוחלט כדי למנוע בעיות חלוקה באפס)
    p_min = 1e-5
    
    # הגבול העליון התיאורטי הוא עומס אוילר לקריסה (עמוד אידיאלי ללא אקסצנטריות)
    # P_euler = (pi^2 * E * I) / L^2  כאשר  I = A * r^2
    p_max = (np.pi**2 * E * A * r**2) / L**2
    
    # ליתר ביטחון, אם עומס המעיכה הפשוט (sigma * A) קטן מעומס אוילר, נשתמש בו כחסם
    p_max = min(p_max, sigma_allow * A)

    # 3. הרצת שיטת החצייה (Bisection) למציאת השורש בדיוק הנדרש
    # הפרמטר xtol מגדיר את הדיוק של ה-X (העומס P) שנדרש לעמוד בטולרנס של 10^-3
    critical_load = bisect(objective_function, p_min, p_max, xtol=1e-4)
    
    return float(critical_load)
