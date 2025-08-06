import numpy as np

def round_away(x):

    """ Function that rounds away from 0 to mimic the Matlab 
    round function.

    Examples: 2.3 -> 2 ; 2.7 -> 3 ; 2.5 -> 3

    Parameters
    ----------
    x : number to round

    Returns
    -------
    rounded value

    Function found at
    https://stackoverflow.com/questions/59142749/how-to-round-away-from-zero
    """
    a = np.abs(x)
    b = np.floor(a) + np.floor(2*(a%1))
    return np.sign(x)*b

def beta(x):

    """
    Beta function used in Meyer and Littlewood-Paley mother wavelets definitions
    """
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return np.power(x,4) * (35 - 84*x + 70*np.power(x,2) - 20*np.power(x,3))