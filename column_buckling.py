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
    
    # 1. הגדרת פונקציית המטרה
    def objective_function(P):
        if P <= 0:
            return -sigma_allow
            
        # חישוב הארגומנט לקוסינוס ברדיאנים
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # הגנה מתמטית מפני הגעה לאסימפטוטה של הקוסינוס (pi/2)
        if angle >= np.pi / 2:
            return float('inf')  # מאמץ שואף לאינסוף מעבר לנקודת הקריסה
            
        # נוסחת הסקנט
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(angle)))
        
        return sigma_max - sigma_allow

    # 2. קביעת גבולות דינמיים ובטוחים
    p_min = 1e-8
    
    # החסם העליון הפיזיקלי המוחלט שבו הקוסינוס מתאפס
    p_euler = (np.pi**2 * E * A * r**2) / L**2
    
    # נתחיל מחסם עליון שמרני: עומס המעיכה הפשוט (ללא אקסצנטריות)
    p_max = sigma_allow * A
    
    # ודואים ש-p_max לא עובר או מתקרב מדי לעומס אוילר כדי למנוע חריגה מהתחום
    if p_max >= p_euler:
        p_max = 0.99 * p_euler

    # בדיקה דינמית: שיטת החצייה דורשת שסימני הקצוות יהיו הפוכים f(p_min)*f(p_max) < 0
    # אם f(p_max) עדיין שלילי, זה אומר ש-p_max קטן מדי בשביל להגיע למאמץ המותר,
    # לכן נקרב את p_max לעומס אוילר עד שהפונקציה תשנה סימן לפלוס.
    iterations = 0
    while objective_function(p_max) < 0 and iterations < 10:
        p_max = p_max + 0.5 * (p_euler - p_max)
        iterations += 1

    # 3. הרצת שיטת החצייה
    # xtol=1e-4 מבטיח עמידה בטולרנס הנדרש של 10^-3 בדיקות האוטומטיות
    critical_load = bisect(objective_function, p_min, p_max, xtol=1e-4)
    
    return float(critical_load)
