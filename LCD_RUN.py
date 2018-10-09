import cv2
import numpy as np

global prevgray
prevgray = cv2.cvtColor(cv2.VideoCapture(0).read()[1], cv2.COLOR_BGR2GRAY)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1) in lines:
        try:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        except :
            print('')
    return vis


def filtering(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Filtered Image', (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def edges(img):
    img = cv2.Canny(img, 2500, 1500, apertureSize=5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Edge Detection', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def features(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img1 = cv2.drawKeypoints(img, kp, img ,color=(0, 255, 0), flags=0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Features', (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return img1


def optflow(img):
    global prevgray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gray, 'Optical Flow', (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return draw_flow(gray, flow)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ctr = 0
    while True:
        _ , img = cap.read()
        if ctr == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Input Image', (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Output', img)
        elif ctr == 1:
            cv2.imshow('Output', filtering(img))
        elif ctr == 2:
            cv2.imshow('Output', edges(img))
        elif ctr == 3:
            cv2.imshow('Output', features(img))
        elif ctr == 4:
            cv2.imshow('Output', optflow(img))

        ch = cv2.waitKey(1)
        # print (ch)
        if ch == 27:
            break
        elif ch == 83:
            ctr = ctr + 1
            if ctr>4:
                ctr = 0
        elif ch == 81:
            ctr = ctr - 1
            if ctr<0:
                ctr = 4
