from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.affinity import scale
import json
import numpy as np
from PIL import Image, ImageDraw
from tkinter import filedialog
import pandas as pd
import cv2

def getCircleRectFromLine(line):
    """Computes parameters to draw with cv2.circle"""
    if len(line) != 2:
        return None
    (c, point) = line
    r = np.subtract(c, point)
    d = np.linalg.norm(r)
    center = tuple(map(int, c))
    radius = int(d)
    return (center, radius)


def create_mask(name, input_json, imagesize=(640, 480), pixel_ext=25):
    # Load the JSON data
    with open(input_json, 'r') as json_file:
        data = json.load(json_file)

    # Extract points for each object
    object_points = {}
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        object_points[label] = points

    # Get the image size from the JSON data
    image_width, image_height = imagesize

    # Create an empty mask
    shapes = np.empty(shape=(len(data['shapes']), image_height, image_width))

    # Create masks for each object and their 15-pixel radius
    i = 0
    for label, points in object_points.items():
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        points = np.array(points, dtype=np.int32)

        # Check the shape type
        if shape['shape_type'] == 'circle':
            # Calculate circle center and radius
            center, radius = getCircleRectFromLine(np.array(points))

            # center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
            # radius = abs(points[0][0] - points[1][0]) / 2
            
            # Create a circle using the shapely Point object and buffer it
            circle = Point(center).buffer(radius+pixel_ext)

            # Convert the circle boundary points back to list
            border_points = list(circle.exterior.coords)
        else:
            # Convert points to shapely polygon and buffer it
            poly = Polygon(points)
            poly = poly.buffer(pixel_ext)

            # Convert polygon points back to list
            border_points = list(poly.exterior.coords)

        # Create a mask for each object
        object_mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(object_mask)
        draw.polygon(tuple(map(tuple, border_points)), fill=1)

        # Add the object mask to the main mask
        mask = np.logical_or(mask, np.array(object_mask))
        mask = np.array(mask, dtype=np.uint8)

        shapes[i] = mask
        i += 1


    # for shape in shapes:
    #     # Display the mask
    #     Image.fromarray(shape * 255).show()


    return shapes


def create_mask_from_points(name, points, imagesize=(640, 480), pixel_ext=100):
    # Load the JSON data
    # Get the image size from the JSON data
    image_width, image_height = imagesize

    # Create an empty mask
    shapes = np.empty(shape=(len(points), image_height, image_width))

    # Create masks for each object and their 15-pixel radius
    i = 0
    for point in points:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        points = np.array(points, dtype=np.int32)


        # center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        # radius = abs(points[0][0] - points[1][0]) / 2
        
        # Create a circle using the shapely Point object and buffer it
        circle = Point(point).buffer(pixel_ext)

        # Convert the circle boundary points back to list
        border_points = list(circle.exterior.coords)


        # Create a mask for each object
        object_mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(object_mask)
        draw.polygon(tuple(map(tuple, border_points)), fill=1)

        # Add the object mask to the main mask
        mask = np.logical_or(mask, np.array(object_mask))
        mask = np.array(mask, dtype=np.uint8)

        shapes[i] = mask
        i += 1


    # for shape in shapes:
    #     # Display the mask
    #     Image.fromarray(shape * 255).show()


    return shapes


import numpy as np
import cv2
from tkinter import filedialog
from tkinter import Tk

def calculate_angle(A, B, C):
    # Vector AB
    AB = B - A

    # Vector BC
    BC = C - B

    # Dot product
    dot_product = np.dot(AB, BC)

    # Magnitudes
    AB_magnitude = np.linalg.norm(AB)
    BC_magnitude = np.linalg.norm(BC)

    if (AB_magnitude * BC_magnitude) == 0:
        return 0
    
    # Cosine of angle
    cosine_angle = dot_product / (AB_magnitude * BC_magnitude)

    # Angle in radians
    angle = np.arccos(cosine_angle)

    # Angle in degrees
    angle = np.degrees(angle)

    return angle


def create_viz(videofile, snout_x, snout_y, neck_x, neck_y, objs, shapes, fps, angle=45):


    # Initialize Tkinter file dialog
    root = Tk()
    root.withdraw()

    file = videofile

    # Load the video
    cap = cv2.VideoCapture(file)

    # Set the font for text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    # Window size
    window_width = 800
    window_height = 600

    # Get total frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a window
    cv2.namedWindow('Video with Overlay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video with Overlay', window_width, window_height)

    # Add a trackbar to the window
    cv2.createTrackbar('Frame no.', 'Video with Overlay', 0, total_frames - 1, lambda x: None)

    # Initial frame
    frame_no = 0
    num_objects = len(objs)
    obj_nearest_points = [None] * num_objects
    exploring_objects = [False] * num_objects
    obj_counter = [0] * num_objects

    while True:
        # Set the current frame number on the trackbar
        cv2.setTrackbarPos('Frame no.', 'Video with Overlay', frame_no)

        # Set the video position to the frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        # Read the current frame
        ret, frame = cap.read()

        # If the frame was not successfully read, break the loop
        if not ret:
            break

        # Get the mouse positions for the current frame
        sx, sy = snout_x[frame_no+2], snout_y[frame_no+2]
        nx, ny = neck_x[frame_no+2], neck_y[frame_no+2]

        sx, sy, nx, ny = int(sx), int(sy), int(nx), int(ny)

        for i in range(num_objects):
            object_points = np.array(np.where(objs[i] == 1))
            object_distances = np.sqrt((object_points[0] - sy)**2 + (object_points[1] - sx)**2)
            obj_nearest_points[i] = object_points.T[np.argmin(object_distances)]
            angle_to_object = calculate_angle(np.array([nx, ny]), np.array([sx, sy]), obj_nearest_points[i][::-1])
            exploring_objects[i] = angle_to_object % 90 <= angle/2 and shapes[i, sy, sx] == 1 #and objs[i, sy, sx] == 0
            if exploring_objects[i]:
                obj_counter[i] += 1

        for obj in objs:
            # Get the coordinates of the boundary points of the shape
            contours, _ = cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the frame
            cv2.drawContours(frame, contours, -1, (255, 255, 0), 2)

        for shape in shapes:
            # Get the coordinates of the boundary points of the shape
            contours, _ = cv2.findContours(shape.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the frame
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

        # Draw the snout position and orientation on the frame
        frame = cv2.circle(frame, (sx, sy), 5, (0, 255, 0), -1)  # Draw the snout position as a green circle
        frame = cv2.line(frame, (sx, sy), (nx, ny), (255, 0, 0), 2)  # Draw the orientation as a blue line

        for i in range(num_objects):
            frame = cv2.circle(frame, (obj_nearest_points[i][1], obj_nearest_points[i][0]), 5, (255, 0, 0), -1)
        

        # Overlay whether the mouse is exploring an object
        text = ', '.join([f'Exploring Object {i + 1}: {exploring_objects[i]}' for i in range(num_objects)])
        # frame = cv2.putText(frame, text, (10, 30), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(num_objects):
            if exploring_objects[i]:
                print(f'EO {i + 1}\n')

        # Display the frame
        cv2.imshow('Video with Overlay', frame)

        # Wait for a key press and get the key pressed
        key = cv2.waitKey(0)

        # If the 'q' key is pressed, stop the loop
        if key == ord('q'):
            break
        elif key == ord('d'):  # 'd' key for next frame
            frame_no = min(frame_no + 1, total_frames - 1)
        elif key == ord('a'):  # 'a' key for previous frame
            frame_no = max(frame_no - 1, 0)
        elif key == ord('f'):  # 'f' key for next ten frames
            frame_no = min(frame_no + 10, total_frames - 10)
        else:
            frame_no = min(frame_no + 1, total_frames - 1)



    # After the loop release the cap object
    cap.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


    for i in range(num_objects):
        print(f"Object {i+1} Count: {obj_counter[i]}")

    for i in range(1, num_objects):
        print(f"Discrimination Index for Object {i+1}: {(obj_counter[0]-obj_counter[i])/(obj_counter[0]+obj_counter[i])}")


def create_viz_from_points(videofile, snout_x, snout_y, neck_x, neck_y, points, fps, angle=45, ext=75):

    shapes = create_mask_from_points(name='test', points=points, pixel_ext=ext)

    # Initialize Tkinter file dialog
    root = Tk()
    root.withdraw()

    file = videofile

    # Load the video
    cap = cv2.VideoCapture(file)

    # Set the font for text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    # Window size
    window_width = 800
    window_height = 600

    # Get total frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a window
    cv2.namedWindow('Video with Overlay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video with Overlay', window_width, window_height)

    # Add a trackbar to the window
    cv2.createTrackbar('Frame no.', 'Video with Overlay', 0, total_frames - 1, lambda x: None)

    # Initial frame
    frame_no = 0
    num_objects = len(shapes)
    exploring_objects = [False] * num_objects
    obj_counter = [0] * num_objects

    while True:
        # Set the current frame number on the trackbar
        cv2.setTrackbarPos('Frame no.', 'Video with Overlay', frame_no)

        # Set the video position to the frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        # Read the current frame
        ret, frame = cap.read()

        # If the frame was not successfully read, break the loop
        if not ret:
            break

        # Get the mouse positions for the current frame
        sx, sy = snout_x[frame_no+2], snout_y[frame_no+2]
        nx, ny = neck_x[frame_no+2], neck_y[frame_no+2]

        sx, sy, nx, ny = int(sx), int(sy), int(nx), int(ny)

        for i in range(num_objects):
            exploring_objects[i] = shapes[i, sy, sx] == 1 #and objs[i, sy, sx] == 0
            if exploring_objects[i]:
                obj_counter[i] += 1


        for shape in shapes:
            # Get the coordinates of the boundary points of the shape
            contours, _ = cv2.findContours(shape.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the frame
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

        # Draw the snout position and orientation on the frame
        frame = cv2.circle(frame, (sx, sy), 5, (0, 255, 0), -1)  # Draw the snout position as a green circle
        frame = cv2.line(frame, (sx, sy), (nx, ny), (255, 0, 0), 2)  # Draw the orientation as a blue line


        # Overlay whether the mouse is exploring an object
        text = ', '.join([f'Exploring Object {i + 1}: {exploring_objects[i]}' for i in range(num_objects)])
        # frame = cv2.putText(frame, text, (10, 30), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(num_objects):
            if exploring_objects[i]:
                print(f'EO {i + 1}\n')

        # Display the frame
        cv2.imshow('Video with Overlay', frame)

        # Wait for a key press and get the key pressed
        key = cv2.waitKey(0)

        # If the 'q' key is pressed, stop the loop
        if key == ord('q'):
            break
        elif key == ord('d'):  # 'd' key for next frame
            frame_no = min(frame_no + 1, total_frames - 1)
        elif key == ord('a'):  # 'a' key for previous frame
            frame_no = max(frame_no - 1, 0)
        elif key == ord('f'):  # 'f' key for next ten frames
            frame_no = min(frame_no + 10, total_frames - 10)
        else:
            frame_no = min(frame_no + 1, total_frames - 1)



    # After the loop release the cap object
    cap.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


    for i in range(num_objects):
        print(f"Object {i+1} Count: {obj_counter[i]}")

    for i in range(1, num_objects):
        print(f"Discrimination Index for Object {i+1}: {(obj_counter[0]-obj_counter[i])/(obj_counter[0]+obj_counter[i])}")


