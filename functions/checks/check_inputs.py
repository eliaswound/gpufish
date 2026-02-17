check inputs for the functions 
def check_sigma(sigma):
    """
    Check that sigma is either a single number or a tuple of 3 numbers (z, y, x)
    with z >= y and z >= x. Single numbers are converted to (sigma, sigma, sigma).
    
    Parameters
    ----------
    sigma : int, float, or tuple
        Sigma value(s). If a single number, converted to (sigma, sigma, sigma).
    
    Returns
    -------
    tuple
        Tuple of 3 numbers (z, y, x) to use.
    
    Raises
    ------
    ValueError
        If sigma is not a number or a 3-tuple, or if (z, y, x) fails z >= y, z >= x.
    """
    # Single number → (sigma, sigma, sigma)
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)
        print("single sigma input detected, using the same number on 3 axis")
    
    if not isinstance(sigma, tuple):
        raise ValueError("sigma must be a number or a tuple of 3 numbers (z, y, x)")
    
    if len(sigma) != 3:
        raise ValueError("sigma must be a single number or a tuple of 3 numbers (z, y, x)")
    
    z, y, x = sigma
    
    if not all(isinstance(v, (int, float)) for v in (z, y, x)):
        raise ValueError("sigma must contain only numbers")
    
    if z < y or z < x:
        raise ValueError(
            f"sigma (z, y, x) must have z >= y and z >= x; got ({z}, {y}, {x})"
        )
    
    return sigma