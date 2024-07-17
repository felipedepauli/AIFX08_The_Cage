import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Carregar a imagem
image_path = './000015.jpg'
image = cv2.imread(image_path)

# Carregar o modelo YOLOv9
model = YOLO("yolov9-seg.pt")

# Realizar a detecção dos veículos
results = model(image)

# Função para segmentar e desenhar contornos
def segment_and_draw_contours(image, crop, x_offset, y_offset):
    # Convertendo a imagem para escala de cinza e equalizando o histograma
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # Definindo a região de interesse (ROI)
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, crop.shape[1] - 10, crop.shape[0] - 10)  # Ajuste conforme necessário

    # Aplicando o algoritmo GrabCut
    cv2.grabCut(crop, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    crop = crop * mask2[:, :, np.newaxis]
    
    cv2.imshow('Segmented Vehicle', crop)
    cv2.waitKey(0)

    # Encontrando e desenhando contornos
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_crop, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            cnt[:, 0, 0] += x_offset
            cnt[:, 0, 1] += y_offset
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    return image

# Processar os resultados para obter as caixas delimitadoras
boxes = []
for result in results:
    for det in result.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = det.conf[0]
        cls = det.cls[0]
        if int(cls) == 2:
            boxes.append([x1, y1, x2, y2])

# Processar cada veículo detectado
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    image = segment_and_draw_contours(image, crop, x1, y1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Mostrar a imagem final
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
