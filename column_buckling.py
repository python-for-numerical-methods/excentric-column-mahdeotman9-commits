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
    
    def f(P):
        if P <= 0:
            return -sigma_allow
        
        # חישוב הארגומנט בתוך הקוסינוס
        angle = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # הגנה הנדסית: אם הזווית מתקרבת או עוברת את פאי חלקי 2, 
        # המשמעות היא שהעמוד עבר את נקודת הקריסה התיאורטית שלו.
        if angle >= np.pi / 2:
            return 1e10  # החזרת ערך חיובי גבוה מאוד כדי ששיטת החצייה תדע לרדת בעומס
            
        cos_val = np.cos(angle)
        
        # חישוב המאמץ המקסימלי לפי נוסחת הסקנט
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1.0 / cos_val))
        
        return sigma_max - sigma_allow

    # חישוב מדויק של העומס שמאפס את הקוסינוס (הגבול הפיזיקלי העליון המוחלט)
    # angle = pi/2 => (L / 2r) * sqrt(P / EA) = pi/2 => sqrt(P / EA) = pi * r / L
    p_upper_bound = E * A * (np.pi * r / L)**2
    
    # נקבע חסמים בטוחים לשיטת החצייה
    p_min = 0.0
    p_max = 0.9999 * p_upper_bound

    # הרצת שיטת החצייה בדיוק גבוה
    p_critical = bisect(f, p_min, p_max, xtol=1e-5)
    
    return float(p_critical)
