import cv2
import os
import json

resize_scale = 3
initial_scaleFactor = 1.34
minSize_division = 10
maxSize_division = 3
margin_rate = 0.15

cascade = cv2.CascadeClassifier('./cascade4_2/cascade.xml')
file_list = os.listdir('./imgs/test/')
with open('testHistZones2015-11-26_CLAHE.json') as f:
    hist_zones_data = json.load(f)

json_data = []
for i, filename in enumerate(file_list):
    rect = (
        hist_zones_data[i]['annotations'][0]['x'],
        hist_zones_data[i]['annotations'][0]['y'],
        hist_zones_data[i]['annotations'][0]['width'],
        hist_zones_data[i]['annotations'][0]['height']
    )
    img = cv2.imread('./imgs/test/' + filename)
    if rect[2] == 0 or rect[3] == 0:
        x_body, y_body, w_body, h_body = (0, 0, img.shape[1], img.shape[0])
        print 'nobody: ' + filename
    else:
        x_body, y_body, w_body, h_body = rect
        if w_body < img.shape[1] / minSize_division or h_body < img.shape[1] / minSize_division:
            x_body = max(0, x_body + w_body / 2 - img.shape[1] / maxSize_division / 2)
            y_body = max(0, y_body + h_body / 2 - img.shape[1] / maxSize_division / 2)
            w_body = img.shape[1] / maxSize_division
            h_body = img.shape[1] / maxSize_division
        margin_x = img.shape[1] * margin_rate
        margin_y = img.shape[1] * margin_rate
        x_body = int(x_body - margin_x)
        y_body = int(y_body - margin_y)
        w_body = int(w_body + 2 * margin_x)
        h_body = int(h_body + 2 * margin_y)
        if img.shape[1] < x_body + w_body:
            w_body -= x_body + w_body - img.shape[1]
        if img.shape[0] < y_body + h_body:
            h_body -= y_body + h_body - img.shape[0]
        x_body = max(0, x_body)
        y_body = max(0, y_body)
    r_w = int(img.shape[1] / resize_scale)
    r_h = int(img.shape[0] / resize_scale)
    r_x_body = int(x_body / resize_scale)
    r_y_body = int(y_body / resize_scale)
    r_w_body = int(w_body / resize_scale)
    r_h_body = int(h_body / resize_scale)
    img = cv2.resize(img, (r_w, r_h))
    img_body = img[r_y_body: r_y_body + r_h_body, r_x_body: r_x_body + r_w_body]
    minSize = (r_w / minSize_division, r_w / minSize_division)
    maxSize = (r_w / maxSize_division, r_w / maxSize_division)
    scaleFactor = initial_scaleFactor
    minNeighbors = 0
    head = None
    while True:
        head_0 = cascade.detectMultiScale(img_body, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                          minSize=minSize, maxSize=maxSize)
        if len(head_0) == 0:
            if head is None:
                scaleFactor -= 0.01
                minNeighbors = 0
                continue
            break
        minNeighbors += 10
        head = head_0
    while True:
        minNeighbors -= 1
        head_0 = cascade.detectMultiScale(img_body, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                          minSize=minSize, maxSize=maxSize)
        if len(head_0) != 0:
            head = head_0
            break
    image_data = {'class': 'image'}
    annotations = []
    for x, y, w, h in head:
        cv2.rectangle(img, (r_x_body + x, r_y_body + y), (r_x_body + x + w, r_y_body + y + h), (255, 255, 255), 2)
        cv2.rectangle(img, (r_x_body, r_y_body), (r_x_body + r_w_body, r_y_body + r_h_body), (255, 255, 0), 2)
        annotations.append({'x': (r_x_body + x) * resize_scale,
                            'y': (r_y_body + y) * resize_scale,
                            'width': w * resize_scale,
                            'height': h * resize_scale,
                            'type': 'rect',
                            'class': 'Head'})
    cv2.imwrite('E:/RWR/testAnnotation2015-12-10_with_hist_CLAHE/' + filename, img)
    image_data.update({'annotations': annotations, 'filename': filename})
    json_data.append(image_data)
with open('testAnnotations2015-12-10_with_hist_CLAHE.json', 'w') as f:
    json.dump(json_data, f, sort_keys=True, indent=4)