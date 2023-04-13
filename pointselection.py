# importing the module
import cv2
import csv
import numpy as np
import zipfile
import pandas as pd
from io import BytesIO
from scipy.spatial.distance import cdist
import os
import time

# read the images from the folder and store them in a dictionary
folder_path = 'C:/Users/Hampi/Desktop/portal_co/images_2_rs'
image_dict = {}
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    if os.path.isfile(filepath):
        image_name = os.path.splitext(filename)[0]
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is not None:
            image_dict[image_name] = image

# read the image coordinates from the CSV file
df = pd.read_csv('image_coordinates_180.csv')

# global variables
points = []
image_path = ""
point_count = 0

# function to detect edges and display the coordinates of the clicked points on the edge
def detect_edge(event, x, y, flags, params):
    global points, img, img_name, point_count
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if point_count < 5:
            # check if the clicked point is on the edge
            if edges[y, x] == 255:
                point_count += 1
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('image', img)
                # append the selected point to the points list
                if len(points) == 0:
                    points.extend([img_name, x, y])
                else:
                    points.extend([x, y])
                # print the current point coordinates to the console
                print(f"Selected point ({x},{y}) on {img_name}")
            else:
                # find the nearest edge point to the clicked point
                edge_points = np.where(edges == 255)
                dist = cdist([(y,x)], np.transpose(edge_points))
                idx = np.argmin(dist)
                nearest_edge_point = edge_points[1][idx], edge_points[0][idx]
                point_count += 1
                cv2.circle(img, nearest_edge_point, 5, (0, 0, 255), -1)
                cv2.imshow('image', img)
                # append the selected point to the points list
                if len(points) == 0:
                    points.extend([img_name,nearest_edge_point[0], nearest_edge_point[1]])
                else:
                    points.extend([nearest_edge_point[0], nearest_edge_point[1]])
                # print the current point coordinates to the console
                print(f"Selected point ({nearest_edge_point[0]},{nearest_edge_point[1]}) on {img_name}")
        else:
            print("Cannot select more than 5 points.")

# function to write points to a CSV file
def write_points_to_csv():
    if not os.path.exists('image_points.csv'):
        with open('image_points.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['img_path', 'x2_1', 'y2_1', 'x2_2', 'y2_2', 'x2_3', 'y2_3', 'x2_4', 'y2_4', 'x2_5', 'y2_5'])
            writer.writerow(points)
    else:
        with open('image_points.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(points)
def is_image_data_present_in_csv():
    if not os.path.exists('image_points.csv'):
        return False
    else:
        with open('image_points.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == img_name:
                    return True
            return False
if __name__ == "__main__":
# loop through the image names in the CSV file
    img_names = []
    for i in range(2476,2501):
        img_names.append('img_'+str(i)+'.png')
    for img_name in img_names:
        img_name_without_extension = img_name.split('.')[0]
        # retrieve the corresponding image from the dictionary
        img = image_dict[img_name_without_extension]

        # set image path
        image_path = os.path.join(folder_path, img_name)

        # reset points and point count
        points = [image_path]
        point_count = 0

        # converting the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # performing Canny edge detection
        edges = cv2.Canny(gray_img, 100, 200)

        # read the coordinates for the current image
        for i in range(1, 6):
            x, y = df.loc[df['img_names'] == img_name, [f'x1_{i}', f'y1_{i}']].values[0]
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, f'point_{i}', (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # displaying the edge-detected image with annotations and label
        cv2.putText(img, img_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('image', img)

        # set mouse handler for the image
        # and call the detect_edge() function
        cv2.setMouseCallback('image', detect_edge)

        # wait for a key to be pressed
        while True:
            key = cv2.waitKey(1000)
            if key == ord('q'):
                # write the selected points to a CSV file and break the loop
                write_points_to_csv()
                break
            elif point_count >= 5:
                # write the selected points to a CSV file if the data for the current image does not exist
                if not is_image_data_present_in_csv():
                    write_points_to_csv()
                break

        # close the window
        cv2.destroyAllWindows()
        