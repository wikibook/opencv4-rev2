import cv2
import numpy as np

config = "tensorflow_model/graph.pbtxt"
model = "tensorflow_model/frozen_inference_graph.pb"
with open("tensorflow_model/labelmap.txt") as file:
    classNames = file.read().splitlines()

image = cv2.imread("bus.jpg")
net = cv2.dnn.readNetFromTensorflow(model, config)
inputBlob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

net.setInput(inputBlob)
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

height, width = image.shape[:2]
threshold = 0.9
for (idx, box) in enumerate(boxes[0, 0, :, :]):
    classID = int(box[1])
    confidence = box[2]
    label = classNames[classID]

    if confidence > threshold:
        box = box[3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype(int)

        mask = masks[idx, classID]
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        mask = ((mask > threshold) * 255).astype(np.uint8)

        mh, mw = mask.shape[:2]
        color_mask = np.full((mh, mw, 3), (255, 0, 0), dtype=np.uint8)
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        image[y1:y2, x1:x2] = cv2.addWeighted(
            image[y1:y2, x1:x2], 1.0, color_mask, 1.0, 0.0
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()