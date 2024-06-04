import numpy as np

import cv2


def select_bounding_box(impath=None, frame=None):
    image = cv2.imread(impath) if impath is not None else frame
    window_name = "Selection"
    size = (800, 600)
    scale_down = 0.5
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, size[0], size[1])
    # cv2.resize(window_name, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)

    x, y, w, h = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True, )
    # cv2.resizeWindow("Selection", 1000, 1000)
    print(f"x = {x}, y = {y}, w = {w}, h = {h}")
    return Point(x, y), Size(w, h)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video")
        return None
    coords = select_bounding_box(frame=frame)
    return coords


def compute_bounding_boxes(frame):
    """
    Dummy function to simulate bounding box computation.
    Replace this with your actual implementation.
    """
    # Assuming bounding boxes are in the format [x, y, width, height]
    bounding_boxes = [[100, 100, 200, 300], [300, 200, 150, 250]]
    return bounding_boxes


class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return Point(int(self.x * other), int(self.y * other))

    def __str__(self):
        return f'Point({self.x}, {self.y})'


class Size:

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

    def __str__(self):
        return f'Size({self.w}, {self.h})'


class MyTracker():
    def __init__(self):
        super().__init__()
        self._tracking_active: bool = False
        self._point_to_track: Point = None
        self._size_to_track: Size = None
        self._old_frame: np.ndarray = None
        self._tracker = None

    def _update(self, frame):
        (success, box) = self._tracker.update(frame)
        if success:
            (x1, y1, width, height) = [int(v) for v in box]
            return x1, y1, width, height
        else:
            return 0, 0, 0, 0

    def _xywh_toroi(self, point, size):
        x, y = point.x, point.y
        width, height = size.w, size.h
        return [int(x), int(y), int(width * 0.5), int(height * 0.5)]

    def startTracking(self, point_to_track: Point, size_to_track: Size):
        self._point_to_track = point_to_track
        self._size_to_track = size_to_track
        self._tracking_active = True
        print("Starting tracking...")

    def updateTracking(self, frame: np.ndarray, tracked_point_result: Point, object_size: Size) -> bool:
        # self._size_to_track = object_size
        ready_to_track = self._old_frame is not None and self._point_to_track and self._point_to_track.y != -1

        if not ready_to_track:
            self._old_frame = frame
            self._tracker = cv2.legacy.TrackerMedianFlow_create()
            new_roi = self._xywh_toroi(self._point_to_track, self._size_to_track)
            self._tracker.init(frame, new_roi)
            return False

        # cols, rows = frame.shape[:2]
        x1, y1, width, height = self._update(frame)
        frame_center: Point = Point(x1, y1)
        obj_size: Size = Size(width * 2, height * 2)
        # direction: Point = frame_center - self._point_to_track
        # step_size_factor = 0.05

        # tracked_point_result: Point = self._point_to_track + direction * step_size_factor
        # tracked_point_result: Point = frame_center

        self._point_to_track = frame_center
        self._size_to_track = obj_size
        self._old_frame = frame
        return True

    def stopTracking(self):
        self._old_frame = None
        self._point_to_track = (-1, -1)
        self._tracking_active = False
        print("Stop tracking")
        print("old_frame: ", self._old_frame)


def update(frame):
    (success, box) = tracker.update(frame)
    if success:
        (x1, y1, width, height) = [int(v) for v in box]
        return x1, y1, width, height
    else:
        return 0, 0, 0, 0


def process_video(input_path, output_path, tracker: MyTracker) -> None:
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    while True:
        ret, frame = cap.read()
        if ret:
            # Compute bounding boxes for the current frame
            if tracker.updateTracking(frame, None, None):
                x, y, w, h = (
                    tracker._point_to_track.x, tracker._point_to_track.y, tracker._size_to_track.w,
                    tracker._size_to_track.h)

                # Draw bounding boxes on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # # Write the processed frame to the output video
                # out.write(frame)
                cv2.imshow("Selection", frame)
                cv2.waitKey(1)
        else:
            print("Writing finished.")
            # Release the video capture and writer objects
            cap.release()
    # out.release()


if __name__ == '__main__':
    tracker = MyTracker()
    video_path = "white_cup.mp4"
    output_path = "output_video.MP4"

    point, size = get_first_frame(video_path)
    tracker.startTracking(point, size)
    process_video(video_path, output_path, tracker)
    tracker.stopTracking()
