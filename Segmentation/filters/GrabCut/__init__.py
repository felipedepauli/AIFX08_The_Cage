import cv2
import numpy as np
import os
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents
from scipy.interpolate import splprep, splev

class ContourDetection2Filter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        self.intermediate_images = {}
        processed_image = image.copy()
        
        # Convertendo a imagem para escala de cinza e equalizando o histograma
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        # Definindo a região de interesse (ROI)
        mask = np.zeros(gray.shape, np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, gray.shape[1] - 20, gray.shape[0] - 20)

        # Aplicando o algoritmo GrabCut
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        cropped = image * mask2[:, :, np.newaxis]

        # Encontrar componentes conectados na imagem segmentada
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_crop, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        output_image = self.draw_smooth_contours(processed_image, cc)

        self.intermediate_images["Gray Image"] = gray
        self.intermediate_images["Equalized Image"] = equalized
        self.intermediate_images["Cropped"] = cropped
        self.intermediate_images["Result"] = output_image

        return output_image

    def draw_smooth_contours(self, image, cc):
        smoothed_image = image.copy()
        for i in range(cc.size()):
            region = cc[i]
            cnt = self.extract_contour(region)
            if len(cnt) < 4:  # Verificação do número de pontos
                cv2.polylines(smoothed_image, [cnt], isClosed=True, color=(0, 255, 0), thickness=2)
                continue
            x = cnt[:, 0]
            y = cnt[:, 1]
            tck, u = splprep([x, y], s=0)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)
            x_new = x_new.astype(np.int32)
            y_new = y_new.astype(np.int32)
            smooth_contour = np.stack((x_new, y_new), axis=-1)
            cv2.polylines(smoothed_image, [smooth_contour], isClosed=True, color=(0, 255, 0), thickness=2)
        return smoothed_image


    def extract_contour(self, region):
        points = []
        current = region.start_seg
        while current is not None:
            for col in range(current.start_col, current.end_col + 1):
                points.append([col, current.row])
            current = current.next
        return np.array(points)

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
