import numpy as np
import cv2

def warp_image_with_normalized_homography(H, src_image, dest_image):

    Wm = self.scale_homography(H,dest_image.shape[:2])
    resized_src = cv2.resize(src_image, (dest_image.shape[1],dest_image.shape[0]))
    warped_image = cv2.warpPerspective(resized_src, Wm, (dest_image.shape[1],dest_image.shape[0]))
    return warped_image

def get_homography(src_points, dest_points, src_shape, dest_shape, normalized=False):
    '''
    Generates a homography matrix between a source and destination image.  If normalized, this homography
    is based on translation/rotation between normalized points (x,y in range(0,1)).

    src_points, dst_points: list or np.array of at least 4 corresponding planar image locations
    src_shape, dst_shape: shape of input and output images.  shape = (rows, cols)
    '''

    if normalized:
        H, status = cv2.findHomography(normalized_points(src_points, src_shape), normalized_points(dest_points, dest_shape))
    else:
        H, status = cv2.findHomography(src_points, dest_points)
    return H

def normalized_points(points, shape):
    return np.array([(float(x)/shape[1], float(y)/shape[0]) for x,y in points], dtype=np.float32)

def denormalized_points(points, shape):
    return np.array([(float(x)*shape[1], float(y)*shape[0]) for x,y in points], dtype=np.float32)

def scale_homography(H, scale_y, scale_x):
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

