import cv2
import os
import json
​
print cv2.__version__
​
resizeScale = 4
minSizeDivision = 10
maxSizeDivision = 3
​
cascade = cv2.CascadeClassifier('./cascade4_2/cascade.xml')
filelist = os.listdir('./imgs/test/')
jsonData = []
for filename in filelist:
    img = cv2.imread('./imgs/test/' + filename)
    resizeW = img.shape[1] / resizeScale
    resizeH = img.shape[0] / resizeScale
    img = cv2.resize(img, (resizeW, resizeH))
    scaleFactor = 1.375
    minNeighbors = 0
    head = None
    while True:
        head0 = cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                         minSize=(resizeW / minSizeDivision, resizeH / minSizeDivision),
                                         maxSize=(resizeW / maxSizeDivision, resizeH / maxSizeDivision))
        if len(head0) == 0:
            if head is None:
                scaleFactor -= 0.01
                minNeighbors = 0
                continue
            break
        minNeighbors += 10
        head = head0
    while True:
        minNeighbors -= 1
        head0 = cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                         minSize=(resizeW / minSizeDivision, resizeH / minSizeDivision),
                                         maxSize=(resizeW / maxSizeDivision, resizeH / maxSizeDivision))
        if len(head0) != 0:
            head = head0
            break
    imageData = {'class': 'image'}
    annotations = []
    for i, (x, y, w, h) in enumerate(head):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        annotations.append({'x': x * resizeScale,
                            'y': y * resizeScale,
                            'width': w * resizeScale,
                            'height': h * resizeScale,
                            'type': 'rect',
                            'class': 'Head'})
    cv2.imwrite('E:/testAnnotation2015-11-19/' + filename, img)
    imageData.update({'annotations': annotations, 'filename': filename})
    jsonData.append(imageData)
with open('testAnnotations2015-11-19.json', 'w') as f:
    json.dump(jsonData, f, sort_keys=True, indent=4)