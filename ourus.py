import cv2
from roboflow import Roboflow
import numpy as np
import matplotlib.pyplot as plt
rf = Roboflow(api_key="9waiKoHTKYKH1bfhwqH1")
project = rf.workspace().project("j-a-t-t6yrz")
model = project.version(2).model

fig, ax = plt.subplots()

def plot_heatmap(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    normalized_diff = diff / np.max(diff)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(normalized_diff, cmap='hot')
    plt.colorbar(heatmap)

prev_frame = None

# функция обработки слайдеров
def update(val):
    global start_frame, end_frame
    start_frame = cv2.getTrackbarPos('Start Frame', 'Video Player')
    end_frame = cv2.getTrackbarPos('End Frame', 'Video Player')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
cap = cv2.VideoCapture('C:/Users/shabu/Downloads/1_06.53.52.avi')
cv2.namedWindow('Video Player') # создаем окно
cv2.createTrackbar('Frame', 'Video Player', 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), update) # создаем слайдер
# создаем два слайдера для выбора начального и конечного кадра
cv2.createTrackbar('Start Frame', 'Video Player', 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), update)
cv2.createTrackbar('End Frame', 'Video Player', 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), update)


while(cap.isOpened()):
    ret, frame = cap.read() # читаем кадр видео
    
    
    # обновляем значение слайдера в соответствии с текущим кадром
    cv2.setTrackbarPos('Frame', 'Video Player', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    predictions = model.predict(frame, confidence=25, overlap=100).json()['predictions']
    
    for pred in predictions:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, pred['class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Video Player', frame) # отображаем кадр
    
    # ждем нажатия клавиши
    key = cv2.waitKey(40)
    
    # если нажата клавиша "q", выходим из цикла
    if key == ord('q'):
        break
    
    # если нажата клавиша "p", ставим видео на паузу
    if key == ord('p'):
        cv2.waitKey(-1)
        
    # если нажата клавиша "h", строим Heat Map только для выбранного периода
    if key == ord('h'):
        if prev_frame is not None:
            if start_frame <= cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                plot_heatmap(prev_frame, frame)
                plt.show()
                plt.close()
        prev_frame = frame

    
cap.release() # закрываем видеофайл
cv2.destroyAllWindows() # закрываем окно с видео
