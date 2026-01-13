import cv2

# Open webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # Save frame in the current directory
    cv2.imwrite("sample_frame.jpg", frame)
    cv2.imshow("Day 2 - Webcam Test", frame)
    cv2.waitKey(2000)

cap.release()
cv2.destroyAllWindows()

print("Sample frame captured and saved")
