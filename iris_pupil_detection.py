cap = cv2.VideoCapture('Eye-Video.mov')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
      frame = cv2.medianBlur(frame,5)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ret, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_TOZERO)
      frame = cv2.Canny(frame,100,200)
      cv2_imshow(frame)

      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

