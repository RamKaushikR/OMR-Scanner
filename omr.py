import numpy as np
import cv2
import argparse
import imutils
from imutils.perspective import four_point_transform
from imutils import contours


upper_limit = 500  # value above which an option is circled
options = 5  # for each question
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True,
                help='path to input image')

args = vars(ap.parse_args())

answer_key = {0: tuple([1]), 1: tuple([4]), 2: tuple([0]), 3: tuple([3]), 4: tuple([1])}

# discover edges in image
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# find the contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:  # paper has been found
            docCnt = approx
            break

if docCnt is None:
    print('No paper found in image')
    exit(0)

# top-down view
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

thresh = cv2.threshold(warped, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
quesCnt = []

for c in cnts:

    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        quesCnt.append(c)

quesCnt = contours.sort_contours(quesCnt,
                                 method='top-to-bottom')[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(quesCnt), options)):

    count = 0
    cnts = contours.sort_contours(quesCnt[i: i + options])[0]
    bubbled = {}

    for (j, c) in enumerate(cnts):

        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        bubbled[total] = []

        if total < upper_limit:  # option is not marked
            count += 1
        else:  # option is marked
            bubbled[total].append(j)

    if count == options:  # no option has been marked
        for answer in answer_key[q]: # just mark the correct answer from the key
            cv2.drawContours(paper, [cnts[answer]], -1, (0, 255, 0), 3)
        continue

    color = (0, 0, 255)
    k = answer_key[q]

    for total, key in bubbled.items():
        if k == tuple(key):  # correct answer
            color = (0, 255, 0)
            correct += 1
            for answer in k:
                cv2.drawContours(paper, [cnts[answer]], -1, color, 3)
        else:  # either partially correct or wrong            
            partial = 0

            for marked in key:
                if marked in k:
                    color = (0, 255, 0)
                    partial += 1
                else:
                    color = (0, 0, 255)

                cv2.drawContours(paper, [cnts[marked]], -1, color, 3)

            for answer in k: # mark the correct answer if it has not been marked
                if answer not in key:
                    cv2.drawContours(paper, [cnts[answer]], -1, (0, 255, 0), 3)

            correct += (partial / len(k))

    bubbled = {}

cv2.putText(paper, '{:.2f}%'.format(correct * 100 / (len(quesCnt)) * options), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow('Paper', image)
cv2.imshow('Score', paper)
cv2.waitKey(0)
