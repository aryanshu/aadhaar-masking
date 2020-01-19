from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
args = vars(ap.parse_args())

#from google.colab.patches import cv2_imshow #uncomment this line for using colab
image=cv2.imread(args["image"])


def rotate_img(image,count=0):

  """
    rotate img if not aligned in right direction

  """

  
  try:
      text = pytesseract.image_to_osd(image)  
  except:
      text = None
  
  #print(text)
  if text is not None and count<4:
      text = text.split('\n')
      text = text[2].split(':')
      rotate = int(text[1].strip())
      
      if rotate==90:
          image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
      elif rotate==270:
          image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
      elif rotate==180:
          image = cv2.rotate(image, rotateCode=cv2.ROTATE_180)

  elif text is None and count<4:
      count=count+1
      image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
      image=rotate_img(image,count)

  return image

image=rotate_img(image)


def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			
			if scoresData[x] < 0.5:
				continue
 

			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
 
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
 
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
 

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
 

	return (rects, confidences)

orig = image.copy()
(origH, origW) = image.shape[:2]


(newW, newH) = (int(origW/32)*32, int(origH/32)*32)
rW = origW / float(newW)
rH = origH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

net = cv2.dnn.readNet('frozen_east_text_detection.pb')

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
  (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)


results = []

for (startX, startY, endX, endY) in boxes:

  startX = int(startX * rW)
  startY = int(startY * rH)
  endX = int(endX * rW)
  endY = int(endY * rH)


  dX = int((endX - startX) *.1)
  dY = int((endY - startY) *.1)


  startX = max(0, startX - dX)
  startY = max(0, startY - dY)
  endX = min(origW, endX + (dX * 2))
  endY = min(origH, endY + (dY * 2))


  roi = orig[startY:endY, startX:endX]

  config = ("-l eng --oem 1 --psm 7")
  text = pytesseract.image_to_string(roi, config=config)
  

  results.append(((startX, startY, endX, endY), text))
  




results = sorted(results, key=lambda r:r[0][1])
output = orig.copy() 


for ((startX, startY, endX, endY), text) in results:
  """
      uncomment for viewing the text detection
  """

  """
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))

  """
  text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

  numbers = sum(c.isdigit() for c in text)
  if (numbers==4 or numbers==5) and len(text)<8 and startY>int(newH/2) and newH<newW:
    cv2.rectangle(output, (startX, startY), (endX, endY),(255, 255, 255), -1)

  elif (numbers>2 and numbers<7) and len(text)<8 and startY>int(newH/2) and newH>newW:
    cv2.rectangle(output, (startX, startY), (endX, endY),(255, 255, 255), -1)



cv2.imshow(output) #comment this line for colab
#cv2_imshow(output) #uncomment this line for colab

