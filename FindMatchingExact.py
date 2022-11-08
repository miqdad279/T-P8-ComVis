import cv2
import numpy as np
MIN_MATCH_COUNT = 4

imgname1 = "template.png"
imgname2 = "find.png"

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)

matches = matcher.knnMatch(descs1, descs2, 2)

matches = sorted(matches, key=lambda x: x[0].distance)

good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
canvas = img2.copy()

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(canvas, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

matched = cv2.drawMatches(img1, kpts1, canvas, kpts2, good, None)

h, w = img1.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
found = cv2.warpPerspective(img2, perspectiveM, (w, h))

cv2.imwrite("matched.png", matched)
cv2.imwrite("found.png", found)
cv2.imshow("matched.png", matched)
cv2.imshow("found.png", found)
cv2.waitKey()
cv2.destroyAllWindows()
