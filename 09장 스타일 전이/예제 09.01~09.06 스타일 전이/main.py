import cv2
import numpy as np

def segmentation(model_path, image):
    height, width = image.shape[:2]
    model = cv2.dnn.readNetFromONNX(model_path)

    image = image.astype(np.float32, copy=False) / 255.0
    image -= np.full_like(image, [0.5, 0.5, 0.5], dtype=np.float32)
    image /= np.full_like(image, [0.5, 0.5, 0.5], dtype=np.float32)
    
    input_blob = cv2.dnn.blobFromImage(image, 1.0, (192, 192), swapRB=True)
    model.setInput(input_blob)
    
    blob_name = model.getUnconnectedOutLayersNames()
    output_blob = model.forward(blob_name[0])
    
    dst = output_blob[0].transpose(1,2,0)
    dst = cv2.resize(dst, (width, height), interpolation=cv2.INTER_LINEAR)
    
    dst = dst.transpose(2,0,1)[np.newaxis, ...]
    dst = np.argmax(dst, axis=1).transpose(1,2,0)

    dst = (dst * 255).astype(np.uint8)
    return dst

def style_transfer(model_path, image):
    height, width = image.shape[:2]
    model = cv2.dnn.readNetFromONNX(model_path)

    input_blob = cv2.dnn.blobFromImage(image, swapRB=True)
    model.setInput(input_blob)
    
    blob_name = model.getUnconnectedOutLayersNames()
    print(blob_name)
    output_blob = model.forward(blob_name[0])

    dst = output_blob[0].transpose((1, 2, 0))
    dst = dst.clip(0, 255).astype(np.uint8)
    
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    dst = cv2.resize(dst, (width, height), interpolation=cv2.INTER_LINEAR)
    return dst

def image_transfer(src, segment, style):
    result1 = np.where(segment == 255, src, style)
    result2 = np.where(segment == 0, src, style)
    return result1, result2


src = cv2.imread("skateboard.jpg")
seg = segmentation("onnx_model/human_segmentation.onnx", src)
stl = style_transfer("onnx_model/mosaic.onnx", src)

result1, result2 = image_transfer(src, seg, stl)
result = cv2.hconcat((result1, result2))

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()