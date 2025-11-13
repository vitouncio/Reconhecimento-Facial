import os, cv2
p = os.path.join("vids","leon&nilce.mp4")
print("path exists:", os.path.exists(p))
cap = cv2.VideoCapture(p)
print("opened:", cap.isOpened())
print("CAP_PROP_FRAME_COUNT:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("CAP_PROP_FPS:", cap.get(cv2.CAP_PROP_FPS))
cap.release()
