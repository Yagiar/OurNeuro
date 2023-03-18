import cv2
import numpy as np
import os

# Пути к папкам с изображениями
with_helmet_path = "C:\\Program Files (x86)\\neuro\\positive"
without_helmet_path = "C:\\Program Files (x86)\\neuro\\negative"

# Функция для извлечения функций (features) из изображения
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = cv2.HOGDescriptor().compute(gray)
    return features.flatten()

# Создаем списки с функциями (features) и метками (labels) для каждой папки
with_helmet_features = np.array([extract_features(os.path.join(with_helmet_path, image)) for image in os.listdir(with_helmet_path)])
with_helmet_labels = [1] * len(with_helmet_features)
without_helmet_features = np.array([extract_features(os.path.join(without_helmet_path, image)) for image in os.listdir(without_helmet_path)])
without_helmet_labels = [0] * len(without_helmet_features)

# Объединяем списки функций и меток
features = np.vstack((with_helmet_features, without_helmet_features))
labels = np.array(with_helmet_labels + without_helmet_labels)

# Создаем и обучаем классификатор SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# Сохраняем обученную модель
svm.save("helmet_detection_svm.xml")
print("Cat")
