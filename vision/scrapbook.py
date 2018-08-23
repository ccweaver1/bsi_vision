from utils.display import Display
from utils.file_manager import FileManager
from utils.warp_tools import *
from utils.rink_specs import HockeyRink
import random
import cv2
fm = FileManager('bsivisiondata')
d = Display()

annotations = fm.get_folder_list('PHI-PIT_6m-8m/annotations', extension_filter='json')
# random.shuffle(annotations)
for f in annotations:
    print f
    im_dict = fm.read_image_dict('PHI-PIT_6m-8m/annotations', f)
    if not 'warp' in im_dict:
        continue
    
    imname = f.split('.')[0] + '.png'
    im = fm.read_image_file('PHI-PIT_6m-8m/frames', imname)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    H = np.array(im_dict['warp']['M'])
    hr = HockeyRink()

    scaled_H = scale_homography(H, 600, 300)
    H1280 = scale_homography(H, 1280, 720)
    '''
    NEEDED TO BE RESIZING IMAGES BEFORE CALLING WARP!!!
    '''

    # dstxy = (30, 150)
    # normed = (dstxy[0]/float(600), dstxy[1]/float(300))
    # invH = cv2.invert(H)[1]
    # print "For dst xy of {}: ".format(normed)
    # srcx, srcy = get_src_xy_for_dst_xy(normed, invH)
    # print "norm src = {}".format((srcx, srcy))
    # sx, sy = 600*srcx, 300*srcy
    # print "src = {}".format((sx, sy))
    # print "1280720 src = {}".format((srcx * 1280, srcy*720))
    
    im_resize = cv2.resize(im, (600,300))

    print src_xy_to_dst_xy((120,185), H, src_size=(600,300), dsize=(600,300))

    # dst = normH_warp(im_resize, H, (600,300))
    
    # print(dst[dstxy[1], dstxy[0]])
    # print(im_resize[int(srcy*300), int(srcx*600)])

    # crease = dst[100:200, 10:50,:]
    
    # d.display_image(crease)
    # dst_2 = cv2.warpPerspective(im, scaled_H, (600,300))
    # d.display_image(im_resize)
    # d.display_image(im)
    # d.display_image(cv2.warpPerspective(im, scaled_H, (600,300)))
    # warp = cv2.warpPerspective(im, scaled_H, (600,300))
    warp = d.get_warp(im, H, (600,300))
    d.display_warp(im, H, (600,300))
    invwarp = cv2.warpPerspective(warp, scaled_H, (600,300), flags=cv2.WARP_INVERSE_MAP)
    d.display_vstacked(im, invwarp)

    # dst_2 = campbell_speedupP(im, scaled_H, (600,300))
    # dst_2 = cv2.warpPerspective(hr.rink, scaled_H, (1280,720), borderValue=(255,255,255), flags=cv2.WARP_INVERSE_MAP)
    # im = cv2.resize(im, (1280,720))
    # overlay = cv2.addWeighted(im, 0.5, dst_2, 0.5, 0)
    # dst_2 = 

    # d.display_vstacked(dst, im_resize)
    # d.display_warp_overlay(im, H, dsize=(1280,720))
    # d.display_warp_overlay(im, H, dsize=(1280/2,720/2))
    
    # d.display_warp_overlay(im, H, dsize=(1000,500))
    # d.display_warp_overlay(im, H, dsize=(500,500))
    # d.display_warp_overlay(im, H, dsize=(500,1000))
    
