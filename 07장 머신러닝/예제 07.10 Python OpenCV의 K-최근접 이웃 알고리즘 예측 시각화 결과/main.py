import cv2
import numpy as np

def loadTrainData(image_path, label_path):
    with open(image_path, "rb") as image_data:
        images = np.frombuffer(image_data.read(), dtype=np.uint8, offset=16).reshape(-1, 784)

    with open(label_path, "rb") as label_data:
        labels = np.frombuffer(label_data.read(), dtype=np.uint8, offset=8)

    return images, labels


label_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

train_x, train_y = loadTrainData(
    "./fashion-mnist/train-images-idx3-ubyte", "./fashion-mnist/train-labels-idx1-ubyte"
)
test_x, test_y = loadTrainData(
    "./fashion-mnist/t10k-images-idx3-ubyte", "./fashion-mnist/t10k-labels-idx1-ubyte"
)

knn = cv2.ml.KNearest_create()
knn.train(train_x.astype(np.float32), cv2.ml.ROW_SAMPLE, train_y.astype(np.int32))

count = 500
retval, results, neighborResponses, dist = knn.findNearest(
    test_x[:count].astype(np.float32), k=7
)

for idx, result in enumerate(results):
    print(f"Index : {idx}")
    print(f"예측값 : {label_dict[int(result.item())]}")
    print(f"실젯값 : {label_dict[test_y[idx]]}")
    cv2.imshow("images", test_x[idx].reshape(28, 28, 1))
    cv2.waitKey(0)

cv2.destroyAllWindows()