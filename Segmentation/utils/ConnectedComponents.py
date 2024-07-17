import cv2
import numpy as np

class LineSegments:
    def __init__(self, start_col, end_col, row):
        self.start_col = start_col
        self.end_col = end_col
        self.row = row
        self.next = None

class RegionDesc:
    def __init__(self):
        self.start_seg = None
        self.end_seg = None
        self.top = float('inf')
        self.bottom = -1
        self.left = float('inf')
        self.right = -1

class ConnectedComponents:
    def __init__(self, image):
        self.image = image
        self.m_img_height, self.m_img_width = image.shape[:2]
        self.m_contours = []
        self.calculate(image)

    def draw(self, obj, output_img, px_val):
        current_seg = obj.start_seg
        while current_seg is not None:
            output_img[current_seg.row, current_seg.start_col:current_seg.end_col + 1] = px_val
            current_seg = current_seg.next

    def __getitem__(self, pos):
        return self.m_contours[pos]

    def returnRect(self, obj):
        return (obj.left, obj.top, obj.right - obj.left + 1, obj.bottom - obj.top + 1)

    def size(self):
        return len(self.m_contours)

    def calculate(self, img):
        visited = np.zeros_like(img, dtype=bool)
        for row in range(self.m_img_height):
            for col in range(self.m_img_width):
                if img[row, col] > 0 and not visited[row, col]:
                    region = self.flood_fill(img, row, col, visited)
                    self.m_contours.append(region)

    def flood_fill(self, img, row, col, visited):
        region = RegionDesc()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if not visited[r, c] and img[r, c] > 0:
                visited[r, c] = True
                seg = LineSegments(c, c, r)
                if region.start_seg is None:
                    region.start_seg = seg
                    region.end_seg = seg
                else:
                    region.end_seg.next = seg
                    region.end_seg = seg
                region.top = min(region.top, r)
                region.bottom = max(region.bottom, r)
                region.left = min(region.left, c)
                region.right = max(region.right, c)
                if c > 0:
                    stack.append((r, c - 1))
                if c < self.m_img_width - 1:
                    stack.append((r, c + 1))
                if r > 0:
                    stack.append((r - 1, c))
                if r < self.m_img_height - 1:
                    stack.append((r + 1, c))
        return region

# Teste da classe ConnectedComponents
if __name__ == "__main__":
    image = cv2.imread("vehicle.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    cc = ConnectedComponents(binary)

    output_image = np.zeros_like(image)
    for i in range(cc.size()):
        cc.draw(cc[i], output_image, 255)

    cv2.imshow("Connected Components", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
