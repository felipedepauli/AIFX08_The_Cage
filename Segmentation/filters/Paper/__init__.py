import cv2
import os
import numpy as np
from filters.Filter import Filter
from scipy.interpolate import splprep, splev
from utils.ConnectedComponents import ConnectedComponents

class BoundingBox3DFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        cc = ConnectedComponents(binary)
        contour_image = self.draw_bounding_boxes_3d(image, cc)

        self.intermediate_images = {
            "Original Image": image,
            "Gray Image": gray,
            "Binary Image": binary,
            "Contour Image": contour_image
        }

        return contour_image

    def draw_bounding_boxes_3d(self, image, cc):
        output_image = image.copy()
        for i in range(cc.size()):
            region = cc[i]
            contour = self.extract_contour(region)
            if len(contour) < 3:
                continue

            x = contour[:, 0]
            y = contour[:, 1]

            tck, u = splprep([x, y], s=0)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            x_new = x_new.astype(np.int32)
            y_new = y_new.astype(np.int32)
            smooth_contour = np.stack((x_new, y_new), axis=-1)

            cv2.polylines(output_image, [smooth_contour], isClosed=True, color=(0, 255, 0), thickness=2)

            # Adiciona profundidade (simplificação)
            for (x, y) in zip(x_new, y_new):
                cv2.line(output_image, (x, y), (x, y - 50), (0, 0, 255), 1)

        return output_image

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

# Teste da classe BoundingBox3DFilter
if __name__ == "__main__":
    input_image_path = "data/vehicle.jpg"
    output_dir = "output/out0"
    image = cv2.imread(input_image_path)

    bounding_box_3d_filter = BoundingBox3DFilter(config={})
    result = bounding_box_3d_filter.filter(image)

    cv2.imshow("3D Bounding Box Segmentation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
