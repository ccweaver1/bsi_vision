import matplotlib.patches as patches
import numpy as np
import cv2
from datatools.draggables import DraggablePoint, DraggableLine, DraggableQuadrangle
# from homography import get_homography,normalized_points,denormalized_points
from utils.warp_tools import normalized_points, denormalized_points, get_homography, \
                    get_normalized_homography


class WarpManager:

    def __init__(self, motion_callback, release_callback, name, start, end, size=10.0, color='green', picker=25.0):

        self.parent_on_motion = motion_callback
        self.parent_on_release = release_callback

        size = 10.0
        picker = 25.0
        self.warp_pairs = []
        self.warp_pairs.append(DraggableLine(self.on_motion, self.on_release, 'upper_left', start=(50., 220.), end=(172.,465.), size=size,picker=picker))
        self.warp_pairs.append(DraggableLine(self.on_motion, self.on_release, 'upper_left', start=(50.,0.), end=(990.,188.), size=size,picker=picker))
        self.warp_pairs.append(DraggableLine(self.on_motion, self.on_release, 'upper_left', start=(152.,240.), end=(835.,650.), size=size,picker=picker))
        self.warp_pairs.append(DraggableLine(self.on_motion, self.on_release, 'upper_left', start=(138.,116.), end=(1241.,400.), size=size,picker=picker))

    def add_artist(self,parent_ax):

        for c in self.warp_pairs:
            c.add_artist(parent_ax)

    def draw_artist(self,parent_ax):

        for c in self.warp_pairs:
            c.draw_artist(parent_ax)

    def remove_artist(self,parent_ax):

        for c in self.warp_pairs:
            c.remove_artist(parent_ax)

    def on_motion(self, event):

        self.parent_on_motion(event)

    def on_release(self, event):

        self.parent_on_release(event)

    def add_warp_pair(self, im_shape):
        init_start = (min(self.warp_pairs[-1].start_point.x, im_shape[1]), \
            min(self.warp_pairs[-1].start_point.y + 25, im_shape[0]))

        init_end = (min(self.warp_pairs[-1].end_point.x, im_shape[1]), \
            min(self.warp_pairs[-1].end_point.y + 25, im_shape[0]))
        size = 10.0
        picker = 25.0
        new_pair_name = "extra_pair_{}".format(len(self.warp_pairs))
        self.warp_pairs.append(DraggableLine(self.on_motion, self.on_release, new_pair_name, start=init_start, end=init_end, size=size,picker=picker))
    def delete_warp_pair(self):
        self.warp_pairs.pop()

    def only_four_warp_pairs(self):
        while len(self.warp_pairs) > 4:
            self.warp_pairs.pop()

    def set_warp_points(self,pairs,src_image,dest_image,reversed=False):

        # src image is RINK....intended to be warped to dest_image the FRAME
        # reversed = TRUE if input pairs are from image to rink

        if reversed:
            src = denormalized_points(pairs[:,1,:],src_image.shape)
            dest = denormalized_points(pairs[:,0,:],dest_image.shape)
        else :
            src = denormalized_points(pairs[:,0,:],src_image.shape)
            dest = denormalized_points(pairs[:,1,:],dest_image.shape)
        for wp,s,d in zip(self.warp_pairs, src, dest) :
            wp.set_location((s[0], s[1]),(d[0], d[1]))

    def warp_points(self,src_image,dest_image,reversed=False):

        src = normalized_points(np.float32([[warp.start_point.x,warp.start_point.y] for warp in self.warp_pairs]),
            src_image.shape)
        dest = normalized_points(np.float32([[warp.end_point.x,warp.end_point.y] for warp in self.warp_pairs]),
            dest_image.shape)
        pairs = []
        if reversed:
            for s,d in zip(dest, src) :
                pairs.append([[s[0],s[1]],[d[0],d[1]]])
        else:
            for s,d in zip(src, dest) :
                pairs.append([[s[0],s[1]],[d[0],d[1]]])
        return np.float32(pairs)

    def homography(self, src_image, dest_image, normalized=False, reversed=False):

        src = np.float32([[warp.start_point.x,warp.start_point.y] for warp in self.warp_pairs])
        dest = np.float32([[warp.end_point.x,warp.end_point.y] for warp in self.warp_pairs])
        if not reversed :
            # H = get_homography(src, dest, src_image.shape, dest_image.shape, normalized)
            H = get_normalized_homography(src, dest, src_image.shape, dest_image.shape)
            return H
        else :
            H = get_normalized_homography(dest, src, dest_image.shape, src_image.shape)
            
            # H = get_homography(dest, src, dest_image.shape, src_image.shape, normalized)
            return H

    def warp_image(self, src_image, dest_image, reversed=False):

        src = np.float32([[warp.start_point.x,warp.start_point.y] for warp in self.warp_pairs])
        dest = np.float32([[warp.end_point.x,warp.end_point.y] for warp in self.warp_pairs])
        if not reversed :
            H = get_homography(src, dest, src_image.shape, dest_image.shape, normalized=False)
            img = cv2.warpPerspective(src_image, H, (dest_image.shape[1],dest_image.shape[0]))
            return img
        else :
            H = get_homography(dest, src, dest_image.shape, src_image.shape, normalized=False)
            return cv2.warpPerspective(dest_image, H, (src_image.shape[1],src_image.shape[0]))
