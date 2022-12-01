import cv2
import numpy as np
import utilis2 as utlis

webcam = False
path = 'mobile_on_page.jpg'
cap = cv2.VideoCapture('')
cap.set(10,160)
cap.set(3,960)
cap.set(4,540)
scale = 3
wP = 210 *scale
hP = 297 *scale

while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)

    #utlis.getContours(img, showCanny=True)
    imgContours, conts = utlis.getContours(img,showCanny=True, minArea=10000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        print(biggest)

    if len(conts) != 0:
        for obj in conts:
            cv2.polylines(imgContours, [obj[2]], True, (0, 255, 0), 2)
            nPoints = utlis.reorder(obj[2])
            nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
            nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
            cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv2.putText(imgContours, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.putText(imgContours, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
    img = cv2.resize(imgContours, (0, 0), None, 0.8, 0.8)
    cv2.imshow('A4', imgContours)

    # img = cv2.resize(img,(0,0),None,0.5,0.5)
    # cv2.imshow('Original',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()