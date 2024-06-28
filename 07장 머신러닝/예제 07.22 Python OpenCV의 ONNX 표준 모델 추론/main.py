import cv2
import numpy as np

src = cv2.imread("crowd-of-people.jpg")

height, width = src.shape[:2]
inputW, inputH = 640, 640
strides = [8, 16, 32]

net = cv2.dnn.readNetFromONNX("onnx_model/yunet.onnx")
inputBlob = cv2.dnn.blobFromImage(src, 1, (inputW, inputH))

net.setInput(inputBlob)
outBlobNames = net.getUnconnectedOutLayersNames()
outBlobNames = sorted(
    list(outBlobNames), key=lambda x: (x.split("_")[0], int(x.split("_")[1]))
)
outputBlobs = net.forward(outBlobNames)
bbox, classes, kps, objectness = [
    outputBlobs[i : i + 3] for i in range(0, len(outputBlobs), 3)
]

faces = []
landmarks = []
scores = []
for i in range(len(strides)):
    rows = int(inputH / strides[i])
    cols = int(inputW / strides[i])

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c

            clsScore = classes[i][0][idx][0]
            objScore = objectness[i][0][idx][0]
            score = np.sqrt(clsScore * objScore)

            box = bbox[i][0][idx]
            kp = kps[i][0][idx]
            
            cx = (c + box[0]) * strides[i] / inputW * width
            cy = (r + box[1]) * strides[i] / inputH * height
            w = np.exp(box[2]) * strides[i] / inputW * width
            h = np.exp(box[3]) * strides[i] / inputH * height

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0

            rex = (c + kp[0]) * strides[i] / inputW * width
            rey = (r + kp[1]) * strides[i] / inputH * height
            lex = (c + kp[2]) * strides[i] / inputW * width
            ley = (r + kp[3]) * strides[i] / inputH * height

            ntx = (c + kp[4]) * strides[i] / inputW * width
            nty = (r + kp[5]) * strides[i] / inputH * height

            rcmx = (c + kp[6]) * strides[i] / inputW * width
            rcmy = (r + kp[7]) * strides[i] / inputH * height
            lcmx = (c + kp[8]) * strides[i] / inputW * width
            lcmy = (r + kp[9]) * strides[i] / inputH * height

            scores.append(score)
            faces.append([x1, y1, w, h])
            landmarks.append([rex, rey, lex, ley, ntx, nty, rcmx, rcmy, lcmx, lcmy])


indices = cv2.dnn.NMSBoxes(faces, scores, score_threshold=0.7, nms_threshold=0.4)
for i in indices:
    x, y, w, h = list(map(int, faces[int(i)]))
    landmark = list(map(int, landmarks[int(i)]))
    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255))
    cv2.circle(src, (landmark[0], landmark[1]), 3, (0, 255, 255), 2)
    cv2.circle(src, (landmark[2], landmark[3]), 3, (0, 255, 255), 2)
    cv2.circle(src, (landmark[4], landmark[5]), 3, (255, 0, 255), 2)


cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()