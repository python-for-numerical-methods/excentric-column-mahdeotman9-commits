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
    
    Return: העומס P בניונטון (float)
    """
    
    # 1. הגדרת פונקציית המטרה
    def objective_function(P):
        if P <= 0:
            return -sigma_allow
            
        # חישוב הארגומנט לקוסינוס
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # נוסחת הסקנט
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(angle)))
        
        return sigma_max - sigma_allow

    # 2. הגדרת הגבולות בצורה מתמטית קשיחה
    p_min = 1e-8  # גבול תחתון קרוב מאוד לאפס
    
    # עומס אוילר הוא הגבול שבו הקוסינוס מתאפס (angle = pi/2)
    p_euler = (np.pi**2 * E * A * r**2) / L**2
    
    # החסם העליון חייב להיות קרוב מאוד לאוילר כדי לא לפספס פתרונות,
    # אך מעט פחות ממנו כדי למנוע חלוקה באפס או מעבר לסינגולריות.
    p_max = 0.9999 * p_euler

    # 3. חישוב השורש בעזרת שיטת החצייה
    critical_load = bisect(objective_function, p_min, p_max, xtol=1e-2)
    
    return float(critical_load)
