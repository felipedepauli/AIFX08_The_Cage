import cv2
import numpy as np
from ultralytics import YOLO


CAR         = 2
MOTORCYCLE  = 3
BUS         = 5
TRUCK       = 7

class BBox3D:
    def __init__(self):
        self.vp_tangents = {
            'tUl': None,
            'tUr': None,
            'tVl': None,
            'tVr': None,
            'tWl': None,
            'tWr': None
        }
        self.key_points = {
            'A': None,
            'B': None,
            'C': None,
            'D': None,
            'E': None,
            'F': None,
            'G': None,
            'H': None
        }
        self.box_sections = {
            'front': None,
            'bottom': None,
            'left': None,
            'right': None,
            'back': None,
            'top': None
        }

    def calculate_intersection_points(self, image_width, image_height, vanishing_points):
        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                return None  # Lines do not intersect

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y
        
        def distance(point1, point2):
            return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

        # Calculate A, B and C
        if self.vp_tangents['tUl'] and self.vp_tangents['tVl']:
            self.key_points['A'] = tuple([point for point in map(int, line_intersection(self.vp_tangents['tUl'], self.vp_tangents['tVl']))])
        if self.vp_tangents['tVl'] and self.vp_tangents['tWr']:
            self.key_points['B'] = tuple([point for point in map(int, line_intersection(self.vp_tangents['tVl'], self.vp_tangents['tWr']))])
        if self.vp_tangents['tUl'] and self.vp_tangents['tWl']:
            self.key_points['C'] = tuple([point for point in map(int, line_intersection(self.vp_tangents['tUl'], self.vp_tangents['tWl']))])

        # Calculate D, E and F
        if self.key_points['C'] and self.vp_tangents['tVr']:
            line_vertical_C = (self.key_points['C'], (self.key_points['C'][0], image_height))
            self.key_points['D'] = tuple([point for point in map(int, line_intersection(line_vertical_C, self.vp_tangents['tVr']))])
        if self.key_points['B'] and self.vp_tangents['tUr']:
            line_vertical_B = (self.key_points['B'], (self.key_points['B'][0], image_height))
            self.key_points['F'] = tuple([point for point in map(int, line_intersection(line_vertical_B, self.vp_tangents['tUr']))])
            
        # Calculate E using Ed and Ef
        Ed, Ef = None, None
        if self.vp_tangents['tUl'] and self.key_points['A']:
            Ed = line_intersection((vanishing_points["depth"], self.key_points['D']), ((self.key_points['A'][0], 0), (self.key_points['A'][0], image_height)))
        if self.vp_tangents['tVl'] and self.key_points['A']:
            Ef = line_intersection((vanishing_points["length"], self.key_points['F']), ((self.key_points['A'][0], 0), (self.key_points['A'][0], image_height)))
        
        if Ed and Ef:
            distance_A_Ed = distance(self.key_points['A'], Ed)
            distance_A_Ef = distance(self.key_points['A'], Ef)
            self.key_points['E'] = Ed if distance_A_Ed <= distance_A_Ef else Ef
            
        # Calculate G
        if self.vp_tangents['tUr'] and self.vp_tangents['tVr']:
            self.key_points['G'] = tuple([point for point in map(int, line_intersection(self.vp_tangents['tUr'], self.vp_tangents['tVr']))])
            
        # Calculate H
        if self.key_points['B'] and self.key_points['E']:
            line_B_vp_depth = (self.key_points['B'], vanishing_points['depth'])
            line_C_vp_length = (self.key_points['C'], vanishing_points['length'])
            self.key_points['H'] = tuple([point for point in map(int, line_intersection(line_B_vp_depth, line_C_vp_length))])

        # Ensure points are within the image bounds
        for key, point in self.key_points.items():
            if point:
                self.key_points[key] = (max(0, min(point[0], image_width)), max(0, min(point[1], image_height)))

        print("Intersection points:", self.key_points)

    def draw_3d_bounding_box(self, image):
        def draw_polygon(image, points, color, transparency):
            overlay = image.copy()
            cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], color)
            cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

        if all(point is not None for point in [self.key_points['A'], self.key_points['B'], self.key_points['F'], self.key_points['E']]):
            draw_polygon(image, [self.key_points['A'], self.key_points['B'], self.key_points['F'], self.key_points['E']], (0, 255, 0), 0.5)
        if all(point is not None for point in [self.key_points['A'], self.key_points['C'], self.key_points['D'], self.key_points['E']]):
            draw_polygon(image, [self.key_points['A'], self.key_points['C'], self.key_points['D'], self.key_points['E']], (255, 0, 0), 0.3)
        if all(point is not None for point in [self.key_points['D'], self.key_points['G'], self.key_points['F'], self.key_points['E']]):
            draw_polygon(image, [self.key_points['D'], self.key_points['G'], self.key_points['F'], self.key_points['E']], (0, 0, 255), 0.3)
        
        self.box_sections['front'] = [
            (self.key_points['E'], self.key_points['F']),
            (self.key_points['F'], self.key_points['B']),
            (self.key_points['B'], self.key_points['A']),
            (self.key_points['A'], self.key_points['E'])
        ]
        self.box_sections['bottom'] = [
            (self.key_points['A'], self.key_points['B']),
            (self.key_points['B'], self.key_points['H']),
            (self.key_points['H'], self.key_points['C']),
            (self.key_points['C'], self.key_points['A'])
        ]
        self.box_sections['left'] = [
            (self.key_points['C'], self.key_points['A']),
            (self.key_points['A'], self.key_points['E']),
            (self.key_points['E'], self.key_points['D']),
            (self.key_points['D'], self.key_points['C'])
        ]
        self.box_sections['right'] = [
            (self.key_points['B'], self.key_points['F']),
            (self.key_points['F'], self.key_points['G']),
            (self.key_points['G'], self.key_points['H']),
            (self.key_points['H'], self.key_points['B'])
        ]
        self.box_sections['back'] = [
            (self.key_points['D'], self.key_points['G']),
            (self.key_points['G'], self.key_points['H']),
            (self.key_points['H'], self.key_points['C']),
            (self.key_points['C'], self.key_points['D'])   
        ]
        self.box_sections['top'] = [
            (self.key_points['E'], self.key_points['F']),
            (self.key_points['F'], self.key_points['G']),
            (self.key_points['G'], self.key_points['D']),
            (self.key_points['D'], self.key_points['E'])
        ]

        
        def draw_dashed_line(image, start_point, end_point, color, thickness=2, dash_length=2):
            """
            Draws a dashed line on an image.

            Args:
                image (numpy.ndarray): The input image.
                start_point (tuple): The starting point of the line (x, y).
                end_point (tuple): The ending point of the line (x, y).
                color (tuple): The color of the line (B, G, R).
                thickness (int): The thickness of the line.
                dash_length (int): The length of each dash.

            Returns:
                None
            """
            x1, y1 = start_point
            x2, y2 = end_point
            line_length = int(np.hypot(x2 - x1, y2 - y1))
            num_dashes = line_length // dash_length

            for i in range(num_dashes + 1):
                start_x = int(x1 + (x2 - x1) * i / num_dashes)
                start_y = int(y1 + (y2 - y1) * i / num_dashes)
                end_x = int(x1 + (x2 - x1) * (i + 0.5) / num_dashes)
                end_y = int(y1 + (y2 - y1) * (i + 0.5) / num_dashes)
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
                
        first = [self.box_sections['front'], self.box_sections['bottom'], self.box_sections['left'], self.box_sections['right'], self.box_sections['back']]
        for section in first:
            for line in section:
                if line[0] and line[1]:
                    cv2.line(image, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (0, 0, 0), 2)

        second = [self.box_sections['top']]
        for lines in second:
            for line in lines:
                if line[0] and line[1]:
                    draw_dashed_line(image, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (0, 0, 0), 2)

        return image


class BoxDetection:
    """
    Class for detecting and segmenting vehicles in an image using YOLOv9.

    Attributes:
        model (YOLO)                    : The YOLOv9 model for vehicle detection and segmentation.
        vanishing_points (dict)         : Dictionary to store the vanishing points for depth, length, and height.
        bboxes (list)                   : List to store the bounding boxes of detected vehicles.
        points_3d (dict)                : Dictionary to store the 3D points of interest.

    Methods:
        setConf(config)                 : Sets the vanishing points based on the given configuration.
        calculate_vanishing_point(lines): Calculates the vanishing point given two lines.
        detect_vehicles(image)          : Detects vehicles in the given image and returns the annotated image and detections.
        segment_vehicle(image, bbox)    : Segments the vehicle in the given image based on the provided bounding box.
        draw_contours(image, contours)  : Draws the contours of segmented objects on the given image.
        draw_lines(image)               : Draws lines based on the vanishing points on the given image.
        get_tangent_lines(vp, contours) : Calculates and returns the tangent lines from the vanishing points to the vehicle blob.
        draw_tangents(image, tangents)  : Draws the tangent lines on the given image.
        process_image(image_path)       : Processes the image by detecting and segmenting vehicles, and returns the annotated images.

    """

    def __init__(self):
        self.model = YOLO("yolov9-seg.pt")  # Use the YOLOv9 segmentation model
        self.vanishing_points = {
            'depth': None,
            'length': None,
            'height': None
        }
        self.bboxes = []
        self.points_3d = {
            'A': None,
            'B': None,
            'C': None
        }

    def setConf(self, config):
        """
        Sets the vanishing points based on the given configuration.

        Args:
            config (dict): The configuration dictionary containing the vanishing points for depth, length, and height.

        """
        # For each vanishing point, calculate the coordinates based on the given lines
        if 'depth' in config:
            self.vanishing_points['depth']  = self.calculate_vanishing_point(config['depth'])
        if 'length' in config:
            self.vanishing_points['length'] = self.calculate_vanishing_point(config['length'])
        if 'height' in config:
            self.vanishing_points['height'] = config['height']

    def calculate_vanishing_point(self, lines):
        """
        Calculates the vanishing point given two lines. A vanishing point is a point in the image plane that corresponds to the intersection of parallel lines in the 3D world. It is used to calculate the depth, length, and height of objects in the image, correcting the perspective distortion. It's required to have two lines to calculate the vanishing point.

        Args:
            lines (list): List of two lines represented by two points each.

        Returns:
            tuple: The coordinates of the vanishing point.

        """
        # First we get the coefficients of the lines (each line is determined by Ax + By = C), being A = y2 - y1, B = x1 - x2, and C = A * x1 + B * y1
        (x1, y1), (x2, y2) = lines[0]
        (x3, y3), (x4, y4) = lines[1]

        # First line: A1x + B1y = C1
        A1, B1 = y2 - y1, x1 - x2
        C1 = A1 * x1 + B1 * y1

        # Second line: A2x + B2y = C2
        A2, B2 = y4 - y3, x3 - x4
        C2 = A2 * x3 + B2 * y3

        # Calculate the determinant of the system of equations
        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            # Lines are parallel, so there is no vanishing point
            return None
        else:
            # Calculate the coordinates of the vanishing point
            x = (C1 * B2 - B1 * C2) / determinant
            y = (A1 * C2 - C1 * A2) / determinant
            return int(x), int(y)

    def detect_vehicles(self, image):
        """
        Detects vehicles in the given image and returns the annotated image and detections.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            tuple: The annotated image and a list of detections.

        """
        # Perform vehicle detection using YOLOv9 set to detect cars, buses, trucks, and motorcycles
        results = self.model(image, classes=[2, 3, 5, 7])
        
        # Get the bounding boxes and draw them on the image
        self.bboxes = results[0].boxes
        
        detections = []
        # For each bounding box, draw it on the image and add the detection to the detections list
        for box in self.bboxes:
            # Extract the coordinates, confidence of the bounding box and the class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Add the detection to the list
            detection = {
                'rect': (x1, y1, x2, y2),
                'class': cls,
                'confidence': conf.item(),
                'segmentation': None  # Placeholder for segmentation
            }
            detections.append(detection)
            
        return detections

    def segment_vehicle(self, image, detection, threshold_area=20000):
        """
        Segments the vehicle in the given image based on the provided bounding box. It's important to notice that the segmentation is performed only on the region of interest (ROI) defined by the bounding box. This is done to improve the performance of the segmentation model, since it's not necessary to process the entire image, gaining speed and reducing memory usage, and also to avoid segmenting other objects that are not vehicles.

        Args:
            image (numpy.ndarray): The input image.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2).
            threshold_area (int): The minimum area threshold for contours to be considered valid.

        Returns:
            list: List of contours representing the segmented objects.

        """
        if detection['class'] == MOTORCYCLE:
            print("Motorcycle detected")
            threshold_area = 10
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = detection['rect']
        
        # This is the region of interest (ROI) where the segmentation will be performed. It's exactly the bounding box coordinates
        roi = image[y1:y2, x1:x2]
        
        # Perform segmentation only on the region of interest
        results = self.model(roi)
        segmented_objects = []
        
        for result in results:
            if result.masks is not None:
                # Move masks to CPU and convert to numpy arrays
                masks = result.masks.data.cpu().numpy()
                
                # Iterate over the masks and extract the contours
                for mask in masks:
                    mask = (mask > 0.5).astype(np.uint8)  # Binary, converting to uint8

                    # Resize the mask to the size of the ROI to ensure no scaling issues
                    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))

                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Adjust the contours to the original position in the complete image
                    valid_contours = []
                    for contour in contours:
                        if cv2.contourArea(contour) >= threshold_area:
                            contour[:, :, 0] += x1
                            contour[:, :, 1] += y1
                            valid_contours.append(contour)
                    
                    segmented_objects.extend(valid_contours)
                    
        detection['segmentation'] = segmented_objects
        return detection

    def draw_contours(self, image, contours):
        """
        Draws the contours of segmented objects on the given image.

        Args:
            image (numpy.ndarray): The input image.
            contours (list): List of contours to be drawn.

        Returns:
            numpy.ndarray: The image with drawn contours.

        """
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
        return image

    def draw_lines(self, image):
        """
        Draws lines based on the vanishing points on the given image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image with drawn lines.

        """
        height, width = image.shape[:2]
        if self.vanishing_points['depth']:
            vp = self.vanishing_points['depth']
            if vp[0] <= 0:
                for i in range(-2 * width, 5 * width, 100):
                    cv2.line(image, (vp[0], vp[1]), (i, height), (0, 255, 0), 2)
            else:
                for i in range(-2 * width, 5 * width, 100):
                    cv2.line(image, (vp[0], vp[1]), (i, 0), (0, 255, 0), 2)
        if self.vanishing_points['length']:
            vp = self.vanishing_points['length']
            if vp[0] < 0:
                x, y = width, 0 if vp[1] <= 0 else -height if 0 < vp[1] < height else -4 * height
                max_y = 3 * height if vp[1] <= 0 or 0 < vp[1] < height else height
                for i in range(0, max_y, 50):
                    cv2.line(image, (vp[0], vp[1]), (x, y + i), (255, 0, 0), 2)
            elif vp[0] > width:
                x, y = 0, 0 if vp[1] <= 0 else -height if 0 < vp[1] < height else -4 * height
                max_y = 3 * height if vp[1] <= 0 or 0 < vp[1] < height else height
                for i in range(0, max_y, 50):
                    cv2.line(image, (vp[0], vp[1]), (x, y + i), (255, 0, 0), 2)
        if self.vanishing_points['height'] == 'vertical':
            for i in range(0, width, int(width / 20)):
                cv2.line(image, (i, 0), (i, height), (0, 0, 255), 2)
        return image

    def get_tangent_lines(self, vp, contours, image_width, image_height):
        """
        Calculates and returns the tangent lines from the vanishing point to the vehicle blob.

        Args:
            vp (tuple): The vanishing point coordinates.
            contours (list): List of contours representing the vehicle.

        Returns:
            tuple: A tuple containing the left tangent line and the right tangent line.
        """
        def angle_between(p1, p2):
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        def extend_line(vp, point, image_width, image_height):
            # Calculate the slope
            dx = point[0] - vp[0]
            dy = point[1] - vp[1]
            
            # Determine the intersection points with the image borders
            if dx != 0:
                slope = dy / dx
                if dx > 0:
                    x2 = image_width
                    y2 = vp[1] + slope * (image_width - vp[0])
                else:
                    x2 = 0
                    y2 = vp[1] + slope * (0 - vp[0])
                if y2 > image_height:
                    y2 = image_height
                    x2 = vp[0] + (image_height - vp[1]) / slope
                elif y2 < 0:
                    y2 = 0
                    x2 = vp[0] - (vp[1] / slope)
            else:
                x2 = point[0]
                y2 = image_height if dy > 0 else 0

            return (int(x2), int(y2))

        min_angle = float('inf')
        max_angle = float('-inf')
        lT, rT = None, None

        if vp == 'vertical':
            leftmost_point = None
            rightmost_point = None
            for contour in contours:
                for point in contour:
                    point = tuple(point[0])
                    if leftmost_point is None or point[0] < leftmost_point[0]:
                        leftmost_point = point
                    if rightmost_point is None or point[0] > rightmost_point[0]:
                        rightmost_point = point
            lT = ((leftmost_point[0], 0), (leftmost_point[0], image_height))
            rT = ((rightmost_point[0], 0), (rightmost_point[0], image_height))
        else:
            for contour in contours:
                for point in contour:
                    point = tuple(point[0])
                    angle = angle_between(vp, point)
                    line = (vp, point)

                    if angle < min_angle:
                        min_angle = angle
                        lT = line
                    if angle > max_angle:
                        max_angle = angle
                        rT = line
                        
                # Extend the tangent lines
        if lT:
            lT = (lT[0], extend_line(lT[0], lT[1], image_width, image_height))
        if rT:
            rT = (rT[0], extend_line(rT[0], rT[1], image_width, image_height))

        return lT, rT

    def draw_tangents(self, image, tangents):
        """
        Draws the tangent lines on the given image.

        Args:
            image (numpy.ndarray): The input image.
            tangents (tuple): A tuple containing the left tangent line and the right tangent line.

        Returns:
            numpy.ndarray: The image with drawn tangent lines.
        """
        lT, rT = tangents
        cv2.line(image, lT[0], lT[1], (145, 65, 255), 1)
        cv2.line(image, rT[0], rT[1], (145, 65, 255), 1)
        
        
        cv2.imshow('Image with Bounding Boxes and Contours', image)
        cv2.waitKey(0)
        return image

    def process_image(self, image_path):
        """
        Processes the image by detecting and segmenting vehicles, and returns the annotated images.

        Args:
            image_path (str): The path to the input image.

        Returns:
            tuple: The annotated image with bounding boxes and the image with segmented objects.

        """
        image = cv2.imread(image_path)
        if image is None:
            print("Error loading image")
            return None
        drawn_image = image.copy()

        detections = self.detect_vehicles(image)
        drawn_image = self.draw_lines(drawn_image)

        return image, drawn_image, detections
