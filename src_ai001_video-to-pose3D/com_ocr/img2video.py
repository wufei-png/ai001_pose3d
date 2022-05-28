import cv2

img=cv2.imread('test.jpg')
print(img.shape[0])
f=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('camera_test.avi', fourcc=f, fps=25,frameSize=(531,466),isColor=True)
for i in range(220):
  out.write(img)