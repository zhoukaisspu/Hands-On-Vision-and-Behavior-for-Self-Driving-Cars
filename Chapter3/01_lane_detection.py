import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
sys.path.append('../')

from utils import set_save_files, save_dir, ensure_dir, get_save_files

perspective_correction = None
perspective_correction_inv = None
perspective_trapezoid = None
warp_size = None
orig_size = None
left_fit_avg = None
right_fit_avg = None
MIN_DETECTIONS = 8
MAX_DETECTIONS = 10
current_left_fitx = None
current_right_fitx = None
g_left_line = None
g_right_line = None
# pt1, pt2, ptr3, and pt4 and four points defining a trapezoid used for the perspective correction
def compute_perspective(width, height, pt1, pt2, pt3, pt4):
    global perspective_trapezoid, perspective_dest

    perspective_trapezoid = [(pt1[0], pt1[1]), (pt2[0], pt2[1]), (pt3[0], pt3[1]), (pt4[0], pt4[1])]
    src = np.float32([pt1, pt2, pt3, pt4])
    # widest side on the trapezoid
    x1 = pt1[0]
    x2 = pt4[0]
    # height of the trapezoid
    y1 = pt1[1]
    y2 = pt2[1]
    h = y1 - y2
    # The destination is a rectangle with the height of the trapezoid and the width of the widest side
    dst = np.float32([[x1, h], [x1, 0], [x2, 0], [x2, h]])
    perspective_dest = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

    global perspective_correction, perspective_correction_inv
    global warp_size, orig_size

    perspective_correction = cv2.getPerspectiveTransform(src, dst)
    perspective_correction_inv = cv2.getPerspectiveTransform(dst, src)
    warp_size = (width, h)
    orig_size = (width, height)


def warp(img, filename):
    img_persp = img.copy()
    save_dir(img_persp, "original_", filename)
    cv2.line(img_persp, perspective_dest[0], perspective_dest[1], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[1], perspective_dest[2], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[2], perspective_dest[3], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[3], perspective_dest[0], (255, 255, 255), 3)

    cv2.line(img_persp, perspective_trapezoid[0], perspective_trapezoid[1], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[1], perspective_trapezoid[2], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[2], perspective_trapezoid[3], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[3], perspective_trapezoid[0], (0, 192, 0), 3)

    save_dir(img_persp, "persp_", filename)

    return save_dir(cv2.warpPerspective(img, perspective_correction, warp_size, flags=cv2.INTER_LANCZOS4), "warp_",
                    filename)


def edge_detection(channel, filename):
    edge_x = cv2.Scharr(channel, cv2.CV_64F, 1, 0)  # Edge detection using the Scharr operator
    edge_x = np.absolute(edge_x)

    return save_dir(np.uint8(255 * edge_x / np.max(edge_x)), "edge_", filename)


def threshold(channel_threshold, channel_edge, filename):
    # Gradient threshold
    binary = np.zeros_like(channel_edge)
    height = binary.shape[0]

    threshold_up = 15
    threshold_down = 60
    threshold_delta = threshold_down - threshold_up

    for y in range(height):
        binary_line = binary[y, :]
        edge_line = channel_edge[y, :]
        threshold_line = threshold_up + threshold_delta * y / height
        binary_line[edge_line >= threshold_line] = 255

    save_dir(binary, "threshold_edge_only_", filename)
    save_dir(channel_threshold, "channel_only_", filename)

    binary[(channel_threshold >= 180) & (channel_threshold <= 255)] = 255

    binary_threshold = np.zeros_like(channel_threshold)
    binary_threshold[(channel_threshold >= 220) & (channel_threshold <= 255)] = 255

    return (save_dir(binary, "threshold_", filename), save_dir(binary_threshold, "threshold_other", filename))


def histogram(img, prefix, filename):
    partial_img = img[img.shape[0] * 2 // 3:, :]  # Select the bottom part (one third of the image)
    hist = np.sum(partial_img, axis=0)

    if get_save_files() and filename:
        plt.plot(hist)
        plt.savefig(ensure_dir(filename) + "/" + prefix + filename.replace(".jpg", ".png"))
        plt.clf()

    return hist


class HistLanes:
    def __init__(self, x_left, x_right, left_confidence, right_confidence):
        self.x_left = x_left
        self.x_right = x_right
        self.left_confidence = left_confidence
        self.right_confidence = right_confidence


def avg_x(hist_lanes):
    return sum([lane.hist_x for lane in hist_lanes]) // len(hist_lanes)


# Single lane line
class Line:
    def __init__(self):
        self.lane_indexes = None
        # pixel positions
        self.x = None
        self.y = None

        # Fit a second order polynomial to each
        self.fit = None
        # Plotting parameters
        self.fitx = None

        self.derivate = None
        # Histogram
        self.hist_x = None

        self.ondeg_fitx = None

        self.is_left = False
        self.is_right = False
        self.has_line = True

# Data collected during the sliding windows phase
class SlideWindow:
    # left = Line()
    # right = Line()
    # hist = None

    # left_avg = None
    # right_avg = None

    #ploty = None

    def __init__(self, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y):
        self.left = Line()
        self.right = Line()
        self.hist = None
        self.ploty = None
        self.left_avg = None
        self.right_avg = None
        self.left.lane_indexes = np.concatenate(left_lane_indexes)
        self.right.lane_indexes = np.concatenate(right_lane_indexes)
        print("left.lane_indexes:",self.left.lane_indexes.shape)
        print("right.lane_index:",self.right.lane_indexes.shape)
        self.left.hist_x = hist.x_left
        self.right.hist_x = hist.x_right
        if self.left.lane_indexes.shape[0] < 100:
            self.left.has_line = False
        if self.right.lane_indexes.shape[0] < 100:
            self.right.has_line = False
        # Extract left and right positions
        if self.left.has_line is True:
            self.left.x = non_zero_x[self.left.lane_indexes]
            self.left.y = non_zero_y[self.left.lane_indexes]
        if self.right.has_line is True:
            self.right.x = non_zero_x[self.right.lane_indexes]
            self.right.y = non_zero_y[self.right.lane_indexes]

    def plot_lines(self, img, color_left=(0, 255, 255), color_right=(0, 255, 255)):
        left = []
        right = []
        left_one = []
        right_one = []
        myleft = []
        for i in range(0, len(self.ploty)):
            if self.left.has_line is True:
                left.append((self.left.fitx[i], self.ploty[i]))
                left_one.append((self.left.ondeg_fitx[i],self.ploty[i]))
                myleft.append((self.ploty[i],self.left.fitx[i]))
            if self.right.has_line is True:
                right.append((self.right.fitx[i], self.ploty[i]))
                right_one.append((self.right.ondeg_fitx[i],self.ploty[i]))
        #plotarray = np.array(myleft)
        #plotx = plotarray[:,1]
        #ploty = plotarray[:,0]
        #print(plotx.shape)
        #print(ploty.shape)
        # plt.plot(plotx,ploty)
        # plt.gca().invert_yaxis()
        # plt.show()
        #print(myleft)
        cv2.polylines(img, np.int32([left]), False, color_left)
        cv2.polylines(img, np.int32([right]), False, color_right)
        cv2.polylines(img, np.int32([left_one]),False, (14,155,33))
        cv2.polylines(img, np.int32([right_one]),False, (14,155,33))
        return img


def slide_window(img, binary_warped, hist, num_windows, filename=None, prefix=None):
    img_height = binary_warped.shape[0]
    window_height = int(img_height / num_windows)
    # Indices (e.g. coordinates) of the pixels that are not zero
    non_zero = binary_warped.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])
    # Current positions, to be updated while sliding the window; we start with the ones identified with the histogram
    left_x = hist.x_left
    right_x = hist.x_right
    margin = 80
    # Set minimum number of pixels found to recenter window
    min_pixels = 25
    left_lane_indexes = []
    right_lane_indexes = []

    out_img = img.copy() if (filename and prefix) else None

    for idx_window in range(num_windows):
        # X range where we expect the left lane to land
        win_x_left_min = left_x - margin
        win_x_left_max = left_x + margin
        # X range where we expect the right lane to land
        win_x_right_min = right_x - margin
        win_x_right_max = right_x + margin
        # Y range that we are analyzing
        win_y_top = img_height - idx_window * window_height
        win_y_bottom = win_y_top - window_height

        # Show the windows
        if (filename and prefix):
            cv2.rectangle(out_img, (win_x_left_min, win_y_bottom), (win_x_left_max, win_y_top), (255, 255, 255), 2)
            cv2.rectangle(out_img, (win_x_right_min, win_y_bottom), (win_x_right_max, win_y_top), (255, 255, 255), 2)

        # Non zero pixels in x and y
        non_zero_left = ((non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top) & (non_zero_x >= win_x_left_min) & (
                non_zero_x < win_x_left_max)).nonzero()[0]
        non_zero_right = ((non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top) & (non_zero_x >= win_x_right_min) & (
                non_zero_x < win_x_right_max)).nonzero()[0]

        left_lane_indexes.append(non_zero_left)
        right_lane_indexes.append(non_zero_right)
        # If you found > min_pixels pixels, recenter next window on the mean position
        if len(non_zero_left) > min_pixels:
            left_x = int(np.mean(non_zero_x[non_zero_left]))

        if len(non_zero_right) > min_pixels:
            right_x = int(np.mean(non_zero_x[non_zero_right]))
    #print("left_lane_index:",len(left_lane_indexes))
    #print(left_lane_indexes)
    valid, sw = fit_slide_window(binary_warped, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y)

    if valid and filename and prefix and get_save_files():
        out_img[non_zero_y[sw.left.lane_indexes], non_zero_x[sw.left.lane_indexes]] = [0, 0, 255]
        out_img[non_zero_y[sw.right.lane_indexes], non_zero_x[sw.right.lane_indexes]] = [0, 0, 255]
        img_plot = sw.plot_lines(out_img)

        cv2.imwrite(ensure_dir(filename) + "/" + prefix + filename, img_plot)

    return valid, sw


def moving_average(prev_average, new_value, beta):
    return beta * prev_average + (1 - beta) * new_value if prev_average is not None else new_value


def fit_slide_window(binary_warped, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y):
    sw = SlideWindow(hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y)

    # y coordinates
    sw.ploty = np.array([float(x) for x in range(binary_warped.shape[0])])

    # if len(sw.left.y) == 0:
    #     return False, sw

    # Fit a second order polynomial to approximate the points
    if sw.left.has_line is True:
        left_fit = np.polynomial.polynomial.polyfit(sw.left.y, sw.left.x, 2)
        left_fit_one_deg = np.polynomial.polynomial.polyfit(sw.left.y, sw.left.x, 1)
        left_fit_avg = None
        left_fit_avg = moving_average(left_fit_avg, left_fit, 0.92)
        #leftd = np.poly1d(np.flip(left_fit_avg))
        #left_derivative = leftd.deriv()
        sw.left.fitx = left_fit_avg[2] * sw.ploty ** 2 + left_fit_avg[1] * sw.ploty + left_fit_avg[0]
        left_one_deg_d = np.poly1d(np.flip(left_fit_one_deg))
        left_one_drivate = left_one_deg_d.deriv()
        # left_one_value = left_one_drivate(sw.ploty)
        sw.left.ondeg_fitx = left_one_deg_d(sw.ploty)
        sw.left.derivate = left_one_drivate.c[0]
        print("left drivate:", left_one_drivate.c[0])
        if left_one_drivate.c[0] > 1.0:
            sw.left.is_left = True
            sw.left.is_right = False
        if left_one_drivate.c[0] < -1.0:
            sw.left.is_left = False
            sw.left.is_right = True
    if sw.right.has_line is True:
        right_fit = np.polynomial.polynomial.polyfit(sw.right.y, sw.right.x, 2)
        right_fit_one_deg = np.polynomial.polynomial.polyfit(sw.right.y, sw.right.x, 1)
        right_fit_avg = None
        right_fit_avg = moving_average(right_fit_avg, right_fit, 0.92)
        # rightd = np.poly1d(np.flip(right_fit_avg))
        # right_drivate = rightd.deriv()
        sw.right.fitx = right_fit_avg[2] * sw.ploty ** 2 + right_fit_avg[1] * sw.ploty + right_fit_avg[0]
        right_one_deg_d = np.poly1d(np.flip(right_fit_one_deg))
        sw.right.ondeg_fitx = right_one_deg_d(sw.ploty)
        right_one_drivate = right_one_deg_d.deriv()
        sw.right.derivate = right_one_drivate.c[0]
        print("right drivate:", right_one_drivate.c[0])
        if right_one_drivate.c[0] > 1.0:
            sw.right.is_left = True
            sw.right.is_right = False
        if right_one_drivate.c[0] < -1.0:
            sw.right.is_left = False
            sw.right.is_right = True 
    #print('<<<<<<<<<<<<<<<<<<<')
    #print(right_fit)
    
   
    #global left_fit_avg, right_fit_avg
    
    
    
    

    #print(type(left_fit_avg))
    #print(left_fit_avg)
    #print(leftd)

    #print(right_fit_avg)
    #print(rightd)
    #print(right_drivate)
    #leftd_value = left_derivative(sw.ploty)
    #right_value = right_drivate(sw.ploty)
    #print(sw.ploty.shape)
    #print(leftd_value.shape)
    # Generate list of x and y values, using the terms of the polynomial
    # x = Ay^2 + By + C;
    # print("ploty shape:",sw.ploty.shape)
    
    
    # print("sw.left.fitx.shape:",sw.left.fitx.shape)
    # print("sw.right.fitx.shape:",sw.right.fitx.shape)



    
    # right_one_value = right_one_drivate(sw.ploty)

  
    # plt.plot(sw.ploty, left_one_value)
    # plt.show()
    #print("right drivate:",right_one_drivate.c[0])
    # plt.plot(sw.ploty, right_one_value)
    # plt.show()
    return True, sw


def show_lanes(sw, img_warped, img_orig, filename):
    global g_left_line, g_right_line
    img = img_warped.copy()
    if sw.left.has_line:
        print("show_lanes, sw.left.fitx.shape", sw.left.fitx.shape)
    else:
        print("has no left line")
    if sw.right.has_line:
        print("show_lanes, sw.right.fitx.shape", sw.right.fitx.shape)
    else:
        print("has no right line")
    if sw.left and sw.left.has_line is True and sw.left.is_left is False and sw.left.is_right is False:
        fitx_points_warped = np.float32([np.transpose(np.vstack([sw.left.fitx, sw.ploty]))])
        fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inv)
        left_line_warped = np.int_(fitx_points_warped[0])
        left_line = np.int_(fitx_points[0])
        n = len(left_line)

        for i in range(n - 1):
            cv2.line(img_orig, (left_line[i][0], left_line[i][1]), (left_line[i + 1][0], left_line[i + 1][1]),
                     (0, 255, 0), 5)
            cv2.line(img, (left_line_warped[i][0], left_line_warped[i][1]),
                     (left_line_warped[i + 1][0], left_line_warped[i + 1][1]), (0, 255, 0), 5)
        g_left_line = sw.left
    if sw.left.has_line is False and g_right_line is not None and g_left_line is not None:
        if sw.right and sw.right.has_line is True and sw.right.is_left is False and sw.right.is_right is False:
            fitx_points_warped = np.float32([np.transpose(np.vstack([g_left_line.fitx, sw.ploty]))])
            fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inv)
            left_line_warped = np.int_(fitx_points_warped[0])
            left_line = np.int_(fitx_points[0])
            n = len(left_line)

            for i in range(n - 1):
                cv2.line(img_orig, (left_line[i][0], left_line[i][1]), (left_line[i + 1][0], left_line[i + 1][1]),
                    (0, 255, 0), 5)
                cv2.line(img, (left_line_warped[i][0], left_line_warped[i][1]),
                     (left_line_warped[i + 1][0], left_line_warped[i + 1][1]), (0, 255, 0), 5)
    if sw.right and sw.right.has_line is True and sw.right.is_left is False and sw.right.is_right is False:
        fitx_points_warped = np.float32([np.transpose(np.vstack([sw.right.fitx, sw.ploty]))])
        fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inv)
        right_line_warped = np.int_(fitx_points_warped[0])
        right_line = np.int_(fitx_points[0])
        g_right_line = sw.right
        for i in range(len(right_line) - 1):
            cv2.line(img_orig, (right_line[i][0], right_line[i][1]), (right_line[i + 1][0], right_line[i + 1][1]),
                     (0, 0, 255), 5)
            cv2.line(img, (right_line_warped[i][0], right_line_warped[i][1]),
                     (right_line_warped[i + 1][0], right_line_warped[i + 1][1]), (0, 0, 255), 5)

    save_dir(img, "lanes_warped_", filename)
    know_image = cv2.warpPerspective(img, perspective_correction_inv, orig_size)
    print("try to write image")
    parital_img = know_image[199:303,]
    parital_img_2 = img_orig[199:303,]
    myimg = cv2.vconcat([parital_img,parital_img_2])
    save_dir(myimg, "know_image_",filename)
    return save_dir(img_orig, "lanes_orig_", filename)


def lanes_full_histogram(histogram):
    size = len(histogram)
    max_index_left = np.argmax(histogram[0:size // 2])
    max_index_right = np.argmax(histogram[size // 2:]) + size // 2

    return HistLanes(max_index_left, max_index_right, histogram[max_index_left], histogram[max_index_right])


def lanes_partial_histogram(histogram, left_lanes, right_lanes, tolerance):
    max_index_left = partial_lane(histogram, left_lanes, tolerance)
    max_index_right = partial_lane(histogram, right_lanes, tolerance)

    return HistLanes(max_index_left, max_index_right, histogram[max_index_left], histogram[max_index_right])


def partial_lane(histogram, lanes_previous_frames, tolerance):
    lane_avg = avg_x(lanes_previous_frames)
    lane_min = max(lane_avg - tolerance, 0)
    lane_max = min(lane_avg + tolerance, len(histogram))

    return np.argmax(histogram[lane_min:lane_max]) + lane_min


def detect_lanes(img_bgr, filename, left_lanes, right_lanes):
    img_warped = warp(img_bgr, filename)
    img_hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS).astype(float)
    img_edge = edge_detection(img_warped[:, :, 1], filename)
    (img_binary_combined, img_binary_solo) = threshold(img_hls[:, :, 1], img_edge, filename)
    hist = histogram(img_binary_combined, "hist_", filename)
    histogram(img_binary_solo, "hist_solo_", filename)

    if (len(left_lanes) > MIN_DETECTIONS):
        lanes = lanes_partial_histogram(hist, left_lanes, right_lanes, 30)
    else:
        lanes = lanes_full_histogram(hist)

    ret, sw = slide_window(img_warped, img_binary_combined, lanes, 15, filename, prefix="window_")

    if ret:
        left_lanes.append(deepcopy(sw.left))
        right_lanes.append(deepcopy(sw.right))
    else:
        # In case of problems, use the previous detection
        #print(">>>>>>>>>>>>>>>>>>")
        if(len(left_lanes) != 0):
            sw.left = left_lanes[len(left_lanes) - 1]
        if(len(right_lanes) != 0):
            sw.right = right_lanes[len(right_lanes) - 1]

        left_lanes.append(sw.left)
        right_lanes.append(sw.right)

    img_lane = show_lanes(sw, img_warped, img_bgr, filename)

    return img_lane


def reset():
    global left_fit_avg, right_fit_avg

    left_fit_avg = None
    right_fit_avg = None


# compute_perspective(640, 480, [118, 303], [188, 199], [544, 199], [625, 303])
# reset()
# for filename in os.listdir("test_images_sd"):
#     print(filename)
#     img = cv2.imread("test_images_sd/" + filename)
#     detect_lanes(img, filename, [], [])

set_save_files(False)
cap = cv2.VideoCapture("car.avi")
writer = cv2.VideoWriter('video-sd-out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 480))

n = 0
left_lanes = []
right_lanes = []

compute_perspective(640, 480, [118, 303], [188, 199], [544, 199], [625, 303])
reset()
set_save_files(False)
while True:
    ret, frame = cap.read()
    _frame = frame.copy()
    if not (ret):
        break

    # if n % 100 == 0:
    #     set_save_files(True)
    left_lanes = []
    right_lanes = []
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    if n == 1126:
        set_save_files(True)
    else:
        set_save_files(False)
    print("frame_" + str(n))
    frame = detect_lanes(frame, "frame_" + str(n) + ".jpg", [], [])
    frame = cv2.putText(frame, str(n), org, font, fontScale, color, thickness, cv2.LINE_AA)
    myframes = cv2.hconcat([_frame, frame])
    cv2.imshow('window',myframes)
    cv2.waitKey(100)
    # writer.write(myframes)

    if (len(left_lanes) > MAX_DETECTIONS):
        left_lanes = left_lanes[1:]
        right_lanes = right_lanes[1:]

    # if (n % 5 == 0):
    #     print("Saving frame", n)
    n = n + 1

    set_save_files(False)

writer.release()
cap.release()
