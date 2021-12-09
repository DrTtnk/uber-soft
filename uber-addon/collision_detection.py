import numpy as np


# http://www.cs.ubc.ca/labs/imager/tr/2012/ExactContinuousCollisionDetection/beb2012.pdf

def point_triangle_test(v_start, p1_start, p2_start, p3_start, v_end, p1_end, p2_end, p3_end):
    F = ()