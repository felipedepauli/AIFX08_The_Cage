import cv2
from BoxDetection import BoxDetection, BBox3D

def main():
    # Load the image
    image_path = 'image.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error loading image")
        return

    image_height, image_width = image.shape[:2]

    # Instanciate the BoxDetection class
    box_detector = BoxDetection()

    # Set the configuration for the box detection
    config = {
        'depth': [((150, 15), (550, 700)), ((980, 600), (315, 15))],
        'length': [((1270, 450), (100, 810)), ((25, 100), (544, 6))],
        'height': 'vertical'
    }
    box_detector.setConf(config)
    
    # Process the image
    original_image, image_with_lines, detections = box_detector.process_image(image_path)

    # Draw the lines formed by the vanishing points
    image_with_lines = original_image.copy()
    image_with_lines = box_detector.draw_lines(image_with_lines)
    
    cv2.imshow('Image with Lines', image_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Images with bounding boxes
    image_with_bboxes = original_image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['rect']
        cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{box_detector.model.names[detection["class"]]}: {detection["confidence"]:.2f}'
        cv2.putText(image_with_bboxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Imagem com bounding boxes e contornos segmentados
    image_with_bboxes_and_contours = image_with_bboxes.copy()
    
    for detection in detections:
        detection = box_detector.segment_vehicle(image, detection)
    
    for detection in detections:
        if detection['segmentation']:
            for contour in detection['segmentation']:
                image_with_bboxes_and_contours = box_detector.draw_contours(image_with_bboxes_and_contours, [contour])

    cv2.imshow('Image with Bounding Boxes and Contours', image_with_bboxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Image with Bounding Boxes and Contours', image_with_bboxes_and_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    bbox3d = BBox3D()
    for detection in detections:
        if detection['segmentation']:
            for contour in detection['segmentation']:
                complete_image = image_with_bboxes_and_contours.copy()
                lT, rT = box_detector.get_tangent_lines(box_detector.vanishing_points['depth'], [contour], image_width, image_height)
                bbox3d.vp_tangents['tUl'] = rT
                bbox3d.vp_tangents['tUr'] = lT
                box_detector.draw_tangents(complete_image, (lT, rT))

                lT, rT = box_detector.get_tangent_lines(box_detector.vanishing_points['length'], [contour], image_width, image_height)
                bbox3d.vp_tangents['tVl'] = lT
                bbox3d.vp_tangents['tVr'] = rT
                box_detector.draw_tangents(complete_image, (lT, rT))

                lT, rT = box_detector.get_tangent_lines(box_detector.vanishing_points['height'], [contour], image_width, image_height)
                bbox3d.vp_tangents['tWl'] = lT
                bbox3d.vp_tangents['tWr'] = rT
                box_detector.draw_tangents(complete_image, (lT, rT))

                bbox3d.calculate_intersection_points(image_width, image_height, box_detector.vanishing_points)
                print(bbox3d.key_points)
                
                points = [bbox3d.key_points['A'], bbox3d.key_points['B'], bbox3d.key_points['C'], bbox3d.key_points['D'], bbox3d.key_points['E'], bbox3d.key_points['F'], bbox3d.key_points['G'], bbox3d.key_points['H']]
                labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                for point, label in zip(points, labels):
                    if point:
                        cv2.circle(complete_image, (int(point[0]), int(point[1])), 10, (0, 0, 0), -1)
                        cv2.putText(complete_image, label, (int(point[0]), int(point[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                complete_image = bbox3d.draw_3d_bounding_box(complete_image)

                cv2.imshow('Image with Bounding Boxes and Contours', complete_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
