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
    
    # 1. הגדרת פונקציית העזר שהשורש שלה הוא הפתרון המבוקש
    def f(P):
        # חישוב הארגומנט בתוך הקוסינוס (ברדיאנים)
        # שימו לב ש- sec(x) שווה ל- 1 / cos(x)
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # נוסחת הסקנט למאמץ המקסימלי
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(angle)))
        
        # החזרת ההפרש מהמאמץ המותר
        return sigma_max - sigma_allow

    # 2. הגדרת חסמים לשיטת החצייה (Bisection)
    # הגבול התחתון הוא עומס אפסי
    p_min = 1e-5 
    
    # הגבול העליון הוא עומס אוילר התיאורטי (חציון עליון מוחלט לקריסה)
    p_max = (np.pi*2 * E * (A * r2)) / L*2
    
    # במקרה קיצוני שבו החסם העליון של אוילר עובר את גבול המאמץ הישיר
    # נגביל אותו לעומס המקסימלי ממאמץ לחיצה פשוט (P = sigma * A)
    p_max = min(p_max, sigma_allow * A)

    # 3. הרצת שיטת החצייה למציאת השורש בדיוק הנדרש
    # הדיוק כברירת מחדל ב-bisect הוא גבוה מאוד (מעל ומעבר ל-10^-3 הנדרש)
    P_critical = bisect(f, p_min, p_max)
    
    return float(P_critical)
