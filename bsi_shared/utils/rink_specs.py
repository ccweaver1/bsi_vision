import cv2
import os
import sys
from warp_tools import *

class HockeyRink:
    def __init__(self):
        shared_path = filter(lambda x: x.endswith('bsi_shared'), sys.path)[0]
        static_path = os.path.join(shared_path, 'static')

        rink = cv2.imread(os.path.join(static_path, 'hockey_rink.jpg'))
        self.rink = cv2.cvtColor(rink, cv2.COLOR_BGR2RGB)
        
        self.height, self.width, channels = rink.shape
        self.xmin, self.xmax, self.ymin, self.ymax = -100, 100, -42.5, 42.5
        self.rink_size = ()
    
    def rink_image(self):
        return self.rink

    def rink_dims(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    def image_pos_to_nhl_pos(self,pos):
        '''
        converts between image coordinate system (0,0) -> (200,85) and NHL (+/-100,+/-42.5)
        '''
        nhlx = self.xmin + (self.xmax - self.xmin) * pos[0]/self.width
        nhly = self.ymin + (self.ymax - self.ymin) * pos[1]/self.height
        return (nhlx,nhly)

    def im_xy_to_rink_xy(self, xy, normH, im_size):
        '''
        Takes an xy point in an image and returns its position on a hockey rink map
        based on a normalized homography matrix.        
        '''
        return src_xy_to_dst_xy(xy, normH, im_size, dsize=(200,85))