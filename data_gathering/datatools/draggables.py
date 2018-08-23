import matplotlib.patches as patches
import numpy as np


class DraggablePoint:

    # http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    lock = None #  only one can be animated at a time
    in_event = None

    def __init__(self, motion_callback, release_callback, name, pointxy, size=10.0, color='green', picker=25.0):

        self._motion_observer = motion_callback
        self._release_observer = release_callback

        self.point = patches.Ellipse(pointxy, size, size, fc=color, alpha=0.5, edgecolor=color, fill=False, linewidth=size,
             picker=picker)

        self.x = pointxy[0]
        self.y = pointxy[1]

        self.name = name    


    def add_artist(self,parent_ax):
        parent_ax.add_artist(self.point)
        self.connect()        

    def draw_artist(self,parent_ax):
        parent_ax.draw_artist(self.point)

    def remove_artist(self,parent_ax):
        self.disconnect()
        self.point.remove()

    def connect(self):

        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidonpick = self.point.figure.canvas.mpl_connect('pick_event', self.on_pick)

    def disconnect(self):

        'disconnect all the stored connection ids'

        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        self.point.figure.canvas.mpl_disconnect(self.cidonpick)

    def on_pick(self, event):


        if DraggablePoint.lock is not None: return
        if event.artist != self.point:return

        if DraggablePoint.in_event is not None: return

        DraggablePoint.lock = self

    def set_location(self, loc) :
        self.point.center = loc
        self.x = loc[0]
        self.y = loc[1]

    def get_location(self):
        return int(self.x), int(self.y)
 
    def on_motion(self, event):

        if DraggablePoint.in_event is not None: return
        if DraggablePoint.lock is not self: return
        if event.inaxes != self.point.axes: return

        DraggablePoint.in_event = self
        self.set_location((event.xdata,event.ydata))
        self._motion_observer(event)
        DraggablePoint.in_event = None


    def on_release(self, event):

        if DraggablePoint.in_event is not None: return
        if DraggablePoint.lock is not self: return
        if event.inaxes != self.point.axes: return
        DraggablePoint.lock = None

        DraggablePoint.in_event = self
        self._release_observer(event)
        DraggablePoint.in_event = None


class DraggableLine:

    def __init__(self, motion_callback, release_callback, name, start, end, size=10.0, color='green', picker=25.0):

        self.parent_on_motion = motion_callback
        self.parent_on_release = release_callback
        self.start_point = DraggablePoint(self.on_motion, self.on_release, 'start', start, size, color, picker)
        self.end_point = DraggablePoint(self.on_motion, self.on_release, 'end', end, size, 'magenta', picker)
        self.arrow = patches.FancyArrowPatch((self.start_point.x, self.start_point.y), (self.end_point.x,self.end_point.y), color='green')

    def add_artist(self,parent_ax):

        self.start_point.add_artist(parent_ax)
        self.end_point.add_artist(parent_ax)
        parent_ax.add_artist(self.arrow)

    def draw_artist(self,parent_ax):

        self.start_point.draw_artist(parent_ax)
        if self.end_point is not None:
            self.end_point.draw_artist(parent_ax)

        if self.arrow is not None:
            parent_ax.draw_artist(self.arrow)

    def remove_artist(self,parent_ax):

        self.start_point.remove_artist(parent_ax)
        self.end_point.remove_artist(parent_ax)
        self.arrow.remove()

    def set_location(self,start,end) :
        self.start_point.set_location(start)
        self.end_point.set_location(end)
        self.arrow.set_positions(start,end)

    def on_motion(self, event):
        self.arrow.set_positions((self.start_point.x, self.start_point.y), (self.end_point.x,self.end_point.y))
        self.parent_on_motion(event)

    def on_release(self, event):

        self.arrow.set_positions((self.start_point.x, self.start_point.y), (self.end_point.x,self.end_point.y))
        self.parent_on_release(event)


class DraggableQuadrangle:

    def __init__(self, motion_callback, release_callback, name, xy, size=10.0, color='green', picker=25.0):

        self.parent_on_motion = motion_callback
        self.parent_on_release = release_callback

        self.points = [DraggablePoint(self.on_motion, self.on_release, name, pt, size, color, picker) for pt in xy]
        self.quad = patches.Polygon(self.order_points(), closed=True, color=color, fill=False)

    def order_points(self):

        # sort the points based on their x-coordinates
        pts = np.float32([[p.x,p.y] for p in self.points])
        pts = pts[np.argsort(pts[:, 0])]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = pts[:2, :][np.argsort(pts[:2, :][:, 1])]
        rightMost = pts[2:, :]
     
        # use Euclidean distance between the top-left and right-most points; by the Pythagorean
        if np.linalg.norm([leftMost[0],rightMost[0]]) > np.linalg.norm([leftMost[0],rightMost[1]]) :
            result = np.float32([leftMost[0],rightMost[1],rightMost[0],leftMost[1]])
        else:
            result = np.float32([leftMost[0],rightMost[0],rightMost[1],leftMost[1]])
        return result

    def add_artist(self,parent_ax):

        for point in self.points :
            point.add_artist(parent_ax)
        parent_ax.add_artist(self.quad)

    def draw_artist(self,parent_ax):

        for point in self.points :
            point.draw_artist(parent_ax) 
        if self.quad is not None:
            parent_ax.draw_artist(self.quad)

    def remove_artist(self,parent_ax):

        for point in self.points :
            point.remove_artist(parent_ax)
        self.quad.remove()

    def on_motion(self, event):

        self.quad.set_xy(self.order_points())
        self.parent_on_motion(event)

    def on_release(self, event):

        self.quad.set_xy(self.order_points())
        self.parent_on_release(event)


