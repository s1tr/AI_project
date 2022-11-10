import numpy as np
import cv2

image_path = 'images/2.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image = cv2.imread(image_path)

input_shape = (300, 300)
mean = (127.5, 127.5, 127.5)
scale = 1/127.5

blob = cv2.dnn.blobFromImage(
    image, scalefactor=scale, size=input_shape, mean=mean, swapRB=True)

model.setInput(blob)

detected_objects = model.forward()

classes = ["background", "airoplane", "bicycle", "brid", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "mototbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]


colors = np.random.uniform(0, 255, size=(len(classes), 3))

height, width = image.shape[0], image.shape[1]

min_confidence = 20
for i in range(detected_objects.shape[2]):
    confidence = detected_objects[0][0][i][2]*100

    if confidence > min_confidence:
        class_index = int(detected_objects[0, 0, i, 1])

        x1 = int(detected_objects[0, 0, i, 3] * width)
        y1 = int(detected_objects[0, 0, i, 4] * height)
        x2 = int(detected_objects[0, 0, i, 5] * width)
        y2 = int(detected_objects[0, 0, i, 6] * height)

        prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_index], 3)
        cv2.putText(image, prediction_text, (x1, y1-15 if y1 > 30 else y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)


cv2.imshow("Detected Object", image)
cv2.waitKey()
