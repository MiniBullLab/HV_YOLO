import cv2


video = cv2.VideoCapture(1)
while True:
    ret, frame = video.read()
    cv2.imshow('video', frame)
    cv2.waitKey(1)