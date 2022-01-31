import numpy as np

def removeExtremeRad(rad, theta, min, max):
    theta = theta[rad < max]
    rad = rad[rad < max]
    theta = theta[rad > min]
    rad = rad[rad > min]
    return rad, theta

def removeExtremeAng(rad, theta, min, max):
    rad = rad[theta < max]
    theta = theta[theta < max]
    rad = rad[theta > min]
    theta = theta[theta > min]
    return rad, theta
