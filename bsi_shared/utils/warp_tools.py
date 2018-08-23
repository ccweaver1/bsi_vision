import numpy as np
import cv2

def normalized_points(points, shape):
    return np.array([(float(x)/shape[1], float(y)/shape[0]) for x,y in points], dtype=np.float32)

def denormalized_points(points, shape):
    return np.array([(float(x)*shape[1], float(y)*shape[0]) for x,y in points], dtype=np.float32)

def get_homography(src_points, dest_points, src_shape=None, dest_shape=None, normalized=False):
    if normalized:
        return get_normalized_Homography(src_points, dest_points, src_shape, dst_shape)
    else:
        H, status = cv2.findHomography(src_points, dest_points)
    return H

def get_normalized_homography(src_points, dst_points, src_shape, dst_shape):
    '''
    Generates a homography matrix between a source and destination image.  This homography
    is based on translation/rotation between normalized points (x,y in range(0,1)).

    src_points, dst_points: list or np.array of at least 4 corresponding planar image locations
    src_shape, dst_shape: shape of input and output images.  shape = (rows, cols)
    '''

    # src_points_norm = np.array([(float(x)/src_shape[1], float(y)/src_shape[0]) for x,y in src_points], dtype=np.float32)
    # dst_points_norm = np.array([(float(x)/dst_shape[1], float(y)/dst_shape[0]) for x,y in dst_points], dtype=np.float32)
    src_points_norm = normalized_points(src_points, src_shape)
    dst_points_norm = normalized_points(dst_points, dst_shape)
    H, status = cv2.findHomography(src_points_norm, dst_points_norm)
    return H

def scale_homography(H, scale_x, scale_y):
    '''
    Scale a homography matrix, H, such that it can be applied to an image of new size.

    H: input homography matrix
    scale_x, scale_y: scale factors in x and y direction.  If H was designed for an image of size 'x'
        but now must be used with the same image of scale '2x' then 'scale_x' = 2
    '''
    S = np.eye(3,3)
    S[0,0] = scale_x
    S[1,1] = scale_y
    S_inv = np.linalg.inv(S)
    s_x_h = np.matmul(S,H)
    return np.matmul(s_x_h, S_inv)


def src_xy_to_dst_xy(xy, normH, src_size=(1,1), dsize=(1,1)):
    '''
    Note: warpPerspective inverts the H matrix and then maps dest x,y to src points.
    Here there is no inversion, and thus the H matrix maps src x,y to dest points 

    xy: x, y coords or point in src image
    normH: normalized homography matrix
    src_size: size of src image (width, height)
    dsize: size of dest space (weight, height)
    '''
    x = float(xy[0]) / src_size[0]
    y = float(xy[1]) / src_size[1] 
    dst_x = (normH[0,0]*x + normH[0,1]*y + normH[0,2]) / (normH[2,0]*x + normH[2,1]*y + normH[2,2])
    dst_y = (normH[1,0]*x + normH[1,1]*y + normH[1,2]) / (normH[2,0]*x + normH[2,1]*y + normH[2,2])    
    return (dst_x * dsize[0], dst_y * dsize[1])

def dst_xy_to_src_xy(xy, normH, src_size=(1,1), dsize=(1,1)):
    '''
    xy: x, y coords or point in src image
    normH: normalized homography matrix
    src_size: size of src image (width, height)
    dsize: size of dest space (weight, height)
    '''
    normH = cv2.invert(normH)[1]
    x = float(xy[0]) / src_size[0]
    y = float(xy[1]) / src_size[1] 
    dst_x = (normH[0,0]*x + normH[0,1]*y + normH[0,2]) / (normH[2,0]*x + normH[2,1]*y + normH[2,2])
    dst_y = (normH[1,0]*x + normH[1,1]*y + normH[1,2]) / (normH[2,0]*x + normH[2,1]*y + normH[2,2])    
    return (dst_x * dsize[0], dst_y * dsize[1])


