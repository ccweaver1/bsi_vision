
import os
import imghdr
from tqdm import tqdm
import cv2
import json

# top_dir = 'DET-NSH_0h7m45s-0h8m5s'
top_dir = 'BUF-EDM_1h53m26s-1h53m46s'

base_path = '/Users/campbellweaver/Documents/VisionData/'
frames_path = os.path.join(base_path, top_dir, 'frames')
annotation_path = os.path.join(base_path, top_dir, 'annotations')

if not os.path.exists(annotation_path):
    os.makedirs(annotation_path)

basic_annotations = {}
image_exts = ['png', 'jpg', 'tiff', 'giff']

def filter_images(im):
    return imghdr.what(os.path.join(frames_path, im)) in image_exts

for f in tqdm(filter(filter_images, os.listdir(frames_path))):
    frame_path = os.path.join(frames_path, f)
    output_filename = f.split('.')[0] + '.json'
    output_path = os.path.join(annotation_path, output_filename)
    im = cv2.imread(frame_path)
    h, w, c = im.shape
    annotations = {'annotation': {'path': frame_path, 'filename': f, \
                'size': {'width': w, 'height': h, 'depth': c}}}
    with open(output_path, 'w') as f:
        json.dump(annotations, f)
    

