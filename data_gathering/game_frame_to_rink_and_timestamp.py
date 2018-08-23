from optparse import OptionParser
import matplotlib.pyplot as plt
from utils.rink_specs import HockeyRink
from utils.data_extractor import DataExtractor
from datatools.draggables import DraggablePoint, DraggableLine, DraggableQuadrangle
from warp_manager import WarpManager
from matplotlib.widgets import Button
from matplotlib import gridspec
import matplotlib.patches as patches
import sys

import boto3
import json
import numpy as np
import os
import re
import cv2


class Game_Frame_To_Rink:


	def __init__(self, parent_dir, s3_bucket, randomize, check_warps):

		self.plt_first_pass = True

		self.extractor = DataExtractor(parent_dir, s3_bucket)
		self.data_generator = self.extractor.get_data_generator(randomize)

		self.check_warps = check_warps

		self.rink = HockeyRink()

		self.fig = plt.figure(figsize=(12, 8))
		plt.subplots_adjust(bottom=0.2)

		self.ax = plt.subplot()

		self.ax.spines['left'].set_position('center')
		self.ax.spines['bottom'].set_position('center')

		self.status_name = plt.axes([0.1, 0.1, 0.3, 0.04],frameon=False)
		self.status_name.get_xaxis().set_visible(False)	
		self.status_name.get_yaxis().set_visible(False)

		self.status_time_warp = plt.axes([0.10, 0.05, 0.3, 0.04],frameon=False)
		self.status_time_warp.get_xaxis().set_visible(False)	
		self.status_time_warp.get_yaxis().set_visible(False)

		self.modes = ['Setup','Rink','Frame']
		self.mode = 0 #initial mode
		self.axmode_toggle = plt.axes([0.31, 0.05, 0.2, 0.075])
		self.bmode_toggle = Button(self.axmode_toggle, 'Setup | Rink | Frame')
		self.bmode_toggle.on_clicked(self.on_click_toggle_mode)

		self.axsave_next = plt.axes([0.51, 0.05, 0.1, 0.075])
		self.bsave_next = Button(self.axsave_next, 'Save & Next')
		self.bsave_next.on_clicked(self.on_click_save_and_next)

		self.axnext = plt.axes([0.61, 0.05, 0.1, 0.075])
		self.bnext = Button(self.axnext, 'Next')
		self.bnext.on_clicked(self.on_click_next)

		self.axmark_all_clock = plt.axes([0.71, 0.05, 0.1, 0.075])
		self.bmark_all_clock = Button(self.axmark_all_clock, 'Save All TS')
		self.bmark_all_clock.on_clicked(self.save_all_timestamps)

		self.axadd_warp_pair = plt.axes([0.81, 0.05, 0.05, 0.075])
		self.badd_warp_pair = Button(self.axadd_warp_pair, 'Add WP')
		self.badd_warp_pair.on_clicked(self.add_warp_pair)
		self.axdel_warp_pair = plt.axes([0.86, 0.05, 0.05, 0.075])
		self.bdel_warp_pair = Button(self.axdel_warp_pair, 'Del WP')
		self.bdel_warp_pair.on_clicked(self.del_warp_pair)

		self.ax.set_title("", fontsize=12)

		size = 10.0
		picker = 25.0

		self.warp_manager = WarpManager(self.on_motion_callback, self.on_release_callback, 'upper_left', start=(50., 220.), end=(172.,465.), size=size,picker=picker)

		init_corners = [(725, 40), (925, 40), (925, 125), (725, 125)]
		self.quad = DraggableQuadrangle(self.on_motion_callback, self.on_release_callback, 'quad', init_corners, size=size,picker=picker, color='red')

		self.period, self.time, self.prev_period, self.prev_time = None, None, None, None
		self.time_subdivisions	= 0 # the number of times we've seen the same time in consecutive frames
		self.client = boto3.client('rekognition')

		self.background = None


	def launch(self):
		
		self.next_frame()
		self.draw_setup()


	def on_click_toggle_mode(self, event):

		if self.mode == 0 :
			self.mode = 1
			self.draw_warp_rink()
		elif self.mode == 1 :
			self.mode = 2
			self.draw_warp_frame()
		else :
			self.mode = 0
			self.draw_setup()

	def add_warp_pair(self, event):
		self.warp_manager.add_warp_pair(self.im_data['image'].shape)
		self.draw_setup()
	def del_warp_pair(self, event):
		self.warp_manager.delete_warp_pair()
		self.draw_setup()

	def draw_setup(self):

		self.ax.cla()
		self.ax.set_title("Set Warp Points", fontsize=12)
		self.ax.imshow(self.rink.rink, alpha=1.0)
		self.ax.imshow(self.im_data['image'],alpha=0.6)


		self.status_name.cla()
		self.status_name.text(0.01,0.75,self.im_data['data']['annotation']['filename'], fontsize=12)

		tstatus = None
		if self.im_data.has_key('data') : 
			if self.im_data['data'].has_key('annotation') :
				period = self.im_data['data']['annotation'].get('period',"not found")
				if not period:
					period = '?'
				time = self.im_data['data']['annotation'].get('time',"not found")
				if not time:
					time = '?'
				tstatus = period + ' ' + time
		wstatus = None
		if self.im_data.has_key('data') : 
				wstatus = 'WARP' if self.im_data['data'].has_key('warp') else None

		status_string = 'No timestamp' if tstatus is None else tstatus
		status_string += ' : '
		status_string += 'No warp' if wstatus is None else 'WARP'
		self.status_time_warp.cla()
		c = 'red' if wstatus is None else 'black'
		self.status_time_warp.text(0,0.45,status_string, fontsize=12, color=c)

		self.fig.canvas.draw() #must do this before adding artists
		self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

		self.warp_manager.add_artist(self.ax)
		self.quad.add_artist(self.ax)

		if self.plt_first_pass :
			plt.show()
			self.plt_first_pass = False
		else:
			self.fig.canvas.draw()

	def draw_warp_rink(self):

		self.warp_manager.remove_artist(self.ax)
		self.quad.remove_artist(self.ax)

		warp_rink = self.warp_manager.warp_image(self.rink.rink, self.im_data['image'], reversed=False)

		self.ax.cla()
		self.ax.set_title("Warp Speed Scotty", fontsize=12)
		self.ax.imshow(warp_rink, alpha=0.5)
		self.ax.imshow(self.im_data['image'],alpha=0.6)

		plt.draw()


	def draw_warp_frame(self):

		warp_frame = self.warp_manager.warp_image(self.rink.rink, self.im_data['image'], reversed=True)

		self.ax.cla()
		self.ax.set_title("Warp Speed Scotty", fontsize=12)
		self.ax.imshow(self.rink.rink, alpha=0.5)
		self.ax.imshow(warp_frame,alpha=0.6)

		plt.draw()

	def on_click_save_and_next(self, event):

		M = self.warp_manager.homography(self.rink.rink, self.im_data['image'], normalized=True, reversed=True)
		pairs = self.warp_manager.warp_points(self.rink.rink, self.im_data['image'],reversed=True).tolist()
		self.im_data['data']['warp'] = {'M' : M.tolist(),'pairs' : pairs}
		self.extractor.save_imdata(self.im_data['filename'],self.im_data['data'])

		self.prev_period, self.prev_time = self.period, self.time

		self.next_frame()

		self.mode = 0
		self.draw_setup()

	def on_click_next(self, event):

		self.next_frame()
		self.mode = 0
		self.draw_setup()

	def save_all_timestamps(self, event):
		while self.im_data:
			self.get_timestamp()
			self.extractor.save_imdata(self.im_data['filename'],self.im_data['data'])
			self.prev_period, self.prev_time = self.period, self.time
			self.next_frame()
		
	def get_timestamp(self):
		score_bug = self.quad.order_points()
		x1, y1 = int(score_bug[0][0]), int(score_bug[0][1])
		x2, y2 = int(score_bug[2][0]), int(score_bug[2][1])

		clock = self.im_data['image'][y1:y2, x1:x2]
		words = self.parse_score_bug(clock)
		words.sort(key=lambda x: x[1], reverse=True)
		words = [x[0] for x in words]
		period = [x for x in words if x in ['1st', '2nd', '3rd', 'OT', 'OT2']]
		self.period = period[0] if len(period) > 0 else None
		time = [re.findall("\d{1,2}:\d\d|$", x)[0] for x in words if re.match("\d{1,2}:\d\d", x)]
		if len(time) == 0 :
			time = [re.findall("\d{1,2}:\d|$", x)[0] for x in words if re.match("\d{1,2}:\d", x)]
			self.time = None if len(time) == 0 else time[0] + '1'
			print('missing second digit: ',self.time)
		else:
			self.time = time[0]
		if self.period is None or self.time is None :
			print('WORDS:',words)
			# print('PERIOD_TIME: ',period_time)
			print('PERIOD TIME: ',period, time)
		if self.time == self.prev_time:
			self.time_subdivisions += 1  
		else:
			self.time_subdivisions = 0  
		self.im_data['data']['annotation']['period'] = self.period
		self.im_data['data']['annotation']['time'] = "{} {}".format(self.time, self.time_subdivisions)


	def parse_score_bug(self, img):
		temp_filename = 'temp_output_file.png'
		plt.imsave(temp_filename, img)

		with open(temp_filename, 'rb') as i:
			f = i.read()
			b = bytearray(f)
#        os.remove(temp_filename)

		image = {'Bytes': b}
		response = self.client.detect_text(Image=image)
		words = [(detection['DetectedText'], detection['Confidence']) for detection in response['TextDetections']]        
		return words
		
	def on_motion_callback(self, event):

		self.fig.canvas.restore_region(self.background)
		self.warp_manager.draw_artist(self.ax)
		self.quad.draw_artist(self.ax)
		self.fig.canvas.blit(self.ax.bbox)

	def on_release_callback(self, event):
		self.fig.canvas.restore_region(self.background)
		self.warp_manager.draw_artist(self.ax)
		self.quad.draw_artist(self.ax)
		self.fig.canvas.blit(self.ax.bbox)
			   
	def next_frame(self):

	    try:
			self.warp_manager.only_four_warp_pairs() #if we've added extra pairs, remove them
			
			self.im_data = next(self.data_generator)
			warp_points = self.im_data.get('data', {}).get('warp',{}).get('pairs')
			if self.check_warps:
				while not warp_points:
					self.im_data = next(self.data_generator)
					warp_points = self.im_data.get('data', {}).get('warp',{}).get('pairs')

			if warp_points is not None :
				self.warp_manager.set_warp_points(np.float32(warp_points),self.rink.rink, self.im_data['image'],reversed=True)
	    except StopIteration:
	    	sys.exit()

	def test(self):


		# f, axarr = plt.subplot()
		img = cv2.imread('2000px-New_York_Rangers.png')
		self.ax.axis('off')
		# self.ax.imshow(img,alpha=1.0)

		# axarr[1].imshow(self.im_data['image'],alpha=0.6)
		# axarr[0].imshow(self.rink.rink,  alpha=0.6)
		# self.fig.figimage(self.im_data['image'], 100, self.fig.bbox.ymax-img.shape[0],zorder=10)
		# self.fig.figimage(self.im_data['image'], 0, 100, extent=(0.0, 0.0, 1, 1), zorder=10, resize=True)
#aspect='auto'
		# self.ax.imshow(self.im_data['image'], extent=(0.0, 1.0, 0.0, 1.0), zorder=-1)

		ax = self.fig.add_axes([0,0,1,1], label='ax')
		ax.axis('off')

		axs = self.fig.subplots(2)
		# ax1 = self.fig.add_axes([0,.5,1,0.5], label='ax1')
		axs[0].axis('off')
		axs[0].imshow(img,alpha=1.0)
#			, zorder=10,aspect=1.0,origin='upper',extent=[0.0,1.0,0.5,1.0])
#		ax1.imshow(img,alpha=1.0, zorder=10,aspect=1.0,origin='upper',extent=[0.0,1.0,0.5,1.0])

		# ax2 = self.fig.add_axes([0,0,1,.5],label='ax2')
		axs[1].axis('off')
		axs[1].imshow(self.im_data['image'])
# alpha=1.0, zorder=5,origin='upper',extent=[0.0,1.0,0.5,1.0])

# im = image.imread('debian-swirl.png')
# fig, ax = plt.subplots()
# ax.imshow(im, aspect='auto', extent=(0.4, 0.6, .5, .7), zorder=-1)



		self.fig.canvas.draw()
		plt.show()

		# print(self.rink.rink.shape)
		# for x in range(475, 1000, 200):
		# 	for y in range(203, 500, 100):
		# 		pos = (x,y)
		# 		nhl = self.rink.image_pos_to_nhl_pos(pos)
		# 		print('x,y: {} nhl: {}'.format(pos,nhl))


if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-p", "--path", dest="path", 
		default='/Users/douglasweaver/BSI/VisionData/', 
		help="Path to dir containing annotations/ and frames/")
	parser.add_option("-s", "--s3_bucket", dest="s3_bucket", 
		default=None,
		help="S3 bucket")
	parser.add_option("-n", "--unrandomize", dest="unrandomize", 
		default=False, action="store_true",
		help="If set, frames will be in sequential order")
	parser.add_option("--check_warps", dest="check_warps",
		default=False, action="store_true",
		help="If set, shows only frames with warps already generated, for checking purposes.")
	(options, args) = parser.parse_args()

	randomize = not options.unrandomize
	# Valid paths: 
	# NYR-BOS_22m12s-22m30s
	# PHI-PIT_6m-8m

	dd = Game_Frame_To_Rink(options.path, options.s3_bucket, randomize, options.check_warps)
	# dd.test()
	dd.launch()
