import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import warp_tools
from rink_specs import HockeyRink


class Display:
    def __init__(self):
        self.rink = HockeyRink()

    def load_image(self, filename):
        return cv2.imread(filename)

    # def display_image(self, im, title='Image'):
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     cv2.imshow(title, im)
    #     k = cv2.waitKey(0)
    #     if k == ord('q'):
    #         exit()
    #     cv2.destroyAllWindows()
    
    def display_image(self, im, title='Image'):
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # cv2.imshow(title, im)
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     exit()
        # cv2.destroyAllWindows()
        plt.imshow(im)
        plt.show()


    def display_warp(self, im, norm_H, dsize=(1200,600), title='Warped'):
        warped = self.get_warp(im, norm_H, dsize)
        self.display_image(warped, title=title)

    def display_warp_overlay(self, im, norm_H, dsize=(1200,600), title='Warped Overlay'):
        overlay = self.get_warp_overlay(im, norm_H, dsize)
        self.display_image(overlay, title=title)

    def get_warp_overlay(self, im, norm_H, dsize=(1200,600)):
        warped = self.get_warp(im, norm_H, dsize)
        rink_resized = cv2.resize(self.rink.rink, dsize)
        overlay = cv2.addWeighted(rink_resized, 0.5, warped, 0.5, 0, dtype=-1)
        return overlay

    def get_warp(self, im, norm_H, dsize=(1200, 600)):
        scaled_H = warp_tools.scale_homography(norm_H, dsize[0], dsize[1])
        im_resize = cv2.resize(im, dsize)
        warped = cv2.warpPerspective(im_resize, scaled_H, dsize=dsize, borderValue=(255,255,255))
        return warped

    def get_vstacked(self, im1, im2, dsize=(600,600), title='Stacked Images'):
        im1 = cv2.resize(im1, (dsize[0], dsize[1]/2))
        im2 = cv2.resize(im2, (dsize[0], dsize[1]/2))        
        stacked = np.vstack(((im1, im2)))
        return stacked

    def display_vstacked(self, im1, im2, dsize=(600, 600), title='Stacked Images'):
        stacked = self.get_vstacked(im1, im2, dsize, title)
        self.display_image(stacked, title=title)

    def put_text(self, im, text, text_loc=(100,100), text_size=0.7, color=(0,0,0), thickness=1):
        cv2.putText(im, text, text_loc, cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness)

    def put_rectangle(self, im, v1, v2, color=(255,255,255)):
        cv2.rectangle(im, v1, v2, color, -1)

    def save_image(self, filepath, im):
        plt.imshow(im)
        plt.savefig(filepath)