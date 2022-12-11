import cv2
import numpy as np

cap = cv2.VideoCapture(1)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = 20.0
videoWriter = cv2.VideoWriter('press.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (int(width), int(height)))

while cap.isOpened():
    ret, frame = cap.read()

    try:
        cv2.imshow('Press', frame)
        videoWriter.write(frame)
    except Exception as e:
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
videoWriter.release()
cv2.destroyAllWindows()

