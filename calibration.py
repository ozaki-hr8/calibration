from tf_model_object_detection import Model
import numpy as np
import imutils
import cv2


def compute_perspective_transform(corner_points, width, height, image):
    corner_points_array = np.float32(corner_points)
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def compute_point_perspective_transformation(matrix, list_originals):
    list_points_to_detect = np.float32(list_originals).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(
        list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append(
            [transformed_points[i][0][0], transformed_points[i][0][1]])
    return transformed_points_list


def human_detection(boxes, scores, classes, height, width):
    array_boxes = list()
    for i in range(boxes.shape[1]):
        if int(classes[i]) == 1 and scores[i] > 0.95:
            box = [boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2],
                   boxes[0, i, 3]] * np.array([height, width, height, width])
            array_boxes.append(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return array_boxes


def getBoundingboxCentral(array_boxes_detected):
    array_central, array_groundpoints = [], []
    for index, box in enumerate(array_boxes_detected):
        central, ground_point = get_points_from_box(box)
        array_central.append(central)
        array_groundpoints.append(central)
    return array_central, array_groundpoints


def get_points_from_box(box):
    center_x = int(((box[1]+box[3])/2))
    center_y = int(((box[0]+box[2])/2))
    center_y_ground = center_y + ((box[2] - box[0])/2)
    return (center_x, center_y), (center_x, int(center_y_ground))


def draw_rectangle(corner_points):
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]),
             (corner_points[1][0], corner_points[1][1]), (0, 0, 0), thickness=2)
    cv2.line(frame, (corner_points[1][0], corner_points[1][1]),
             (corner_points[3][0], corner_points[3][1]), (0, 0, 0), thickness=2)
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]),
             (corner_points[2][0], corner_points[2][1]), (0, 0, 0), thickness=2)
    cv2.line(frame, (corner_points[3][0], corner_points[3][1]),
             (corner_points[2][0], corner_points[2][1]), (0, 0, 0), thickness=2)

# transformation setting
corner_points = [(272,78),(433,85),(247,338),(491,338)]
viewWidth=393
viewHeight=500
frameSize=500
img_path='birdseye_view/frontier.png'

# load model
model = Model(
    "faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb")

matrix, imgOutput = compute_perspective_transform(
    corner_points, viewWidth, viewHeight, cv2.imread(img_path))
height, width, _ = imgOutput.shape
inputVideo =  cv2.VideoCapture("input/frontier.mp4")

# Inference
while True:
    img = cv2.imread("birdseye_view/frontier.png")
    birdseye_view = cv2.resize(
        img, (width, height), interpolation=cv2.INTER_AREA)
    (isLastFrame, frame) = inputVideo.read()
    if not isLastFrame:
        break
    else:
        frame = imutils.resize(frame, width=int(frameSize))
        (boxes, scores, classes) = model.predict(frame)
        array_boxes_detected = human_detection(
            boxes, scores[0].tolist(), classes[0].tolist(), frame.shape[0], frame.shape[1])
        array_central, array_groundpoints = getBoundingboxCentral(
            array_boxes_detected)
        if(array_groundpoints):
            transformed_originals = compute_point_perspective_transformation(
                matrix, array_groundpoints)

        for point in transformed_originals:
            x, y = point
            cv2.circle(birdseye_view, (int(x), int(y)), 10, (0, 255, 0), 2)

        for index, original in enumerate(transformed_originals):
            if not (original[0] > width or original[0] < 0 or original[1] > height+200 or original[1] < 0):
                if(array_boxes_detected):
                    cv2.rectangle(frame, (array_boxes_detected[index][1], array_boxes_detected[index][0]), (
                        array_boxes_detected[index][3], array_boxes_detected[index][2]), (0, 255, 0), 2)
            else:
                pass

    draw_rectangle(corner_points)
    cv2.imshow("Bird's-eye View", birdseye_view)
    cv2.imshow("Original", frame)

    original = cv2.VideoWriter_fourcc(*"MJPG")
    outputOriginal = cv2.VideoWriter("output/video.avi", original, 25,(frame.shape[1], frame.shape[0]), True)
    birdseye = cv2.VideoWriter_fourcc(*"MJPG")
    outputBirdseye = cv2.VideoWriter("output/birdseye_view.avi", birdseye, 25,(birdseye_view.shape[1], birdseye_view.shape[0]), True)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
