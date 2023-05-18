import cv2
import numpy as np
import os

# Загружаем обученную модель
svm = cv2.ml.SVM_load("helmet_detection_svm.xml")

# Путь к папке с изображениями
images_path = "C:\\Program Files (x86)\\neuro\\test"

# Функция для извлечения функций (features) из изображения
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = cv2.HOGDescriptor().compute(gray)
    return features.flatten()

# Проходим по всем изображениям в папке и применяем модель
for image_file in os.listdir(images_path):
    image_path = os.path.join(images_path, image_file)
    features = extract_features(image_path)
    features = np.array([features])
    result = svm.predict(features)[1]
    
    # Если изображение содержит каску, печатаем имя файла и выводим его на экран
    if result == 1:
        print("Image {} contains a helmet".format(image_file))
        image = cv2.imread(image_path)
        cv2.imshow("Helmet Detection", image)
        cv2.waitKey(0)
    else:
        print("NO")
        image = cv2.imread(image_path)
        cv2.imshow("Helmet Detection", image)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()
