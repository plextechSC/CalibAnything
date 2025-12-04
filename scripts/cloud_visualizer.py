import open3d as opd
import numpy as np
import math
import cv2
import json


np.set_printoptions(precision=6, suppress=True)

# gets the rotation matrix
def eulerangles_to_rotmat(roll, pitch, yaw):
  rotmat_roll = np.array(
    [
      [1, 0, 0],
      [0, math.cos(roll), -math.sin(roll)],
      [0, math.sin(roll), math.cos(roll)]
    ]
  )
  rotmat_pitch = np.array(
    [
      [math.cos(pitch), 0, math.sin(pitch)],
      [0, 1, 0],
      [-math.sin(pitch), 0, math.cos(pitch)]
    ]
  )
  rotmat_yaw = np.array(
    [
      [math.cos(yaw), -math.sin(yaw), 0],
      [math.sin(yaw),  math.cos(yaw), 0],
      [0, 0, 1]
    ]
  )
  rotmat = np.matmul(np.matmul(rotmat_yaw , rotmat_pitch), rotmat_roll)
  return rotmat

# optical extrinsic from the tool.
calibrationpath = './data/lucid/fwc_c/fwc_c.json'
# calibrationpath = './data/lucid/fnc/fnc.json'
with open(calibrationpath, "r") as f:
    data = json.load(f)

# import pdb;pdb.set_trace()
intr = data['intrinsic_params']
extr = data['extrinsic_params']

roll = math.radians(extr['roll'])
pitch = math.radians(extr['pitch'])
yaw = math.radians(extr['yaw'])
px = extr['px']
py = extr['py']
pz = extr['pz']

rotation_matrix = eulerangles_to_rotmat(roll, pitch, yaw)

translation_vector = np.array([px, py, pz])
# import pdb;pdb.set_trace()
# intrinsic from the tool
fx= intr['fx'] * 2
fy= intr['fy'] * 2
cx= intr['cx'] * 2
cy= intr['cy'] * 2

k1= intr['k1']
k2= intr['k2']
k3= intr['k3']
k4= intr['k4']

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=float)

distortion = np.array([[k1], [k2], [k3], [k4]], dtype=float)
rotation_vector = np.array([[0.],
       [0.],
       [0.]])
translation_vector2 = np.array([[0.],
       [0.],
       [0.]])

#below code block is for modifying intrinsics for fwc_c, fwc_l, fwc_r
#only use this code block if dealing with distorted images/raw data
scale = 1.0
intrinsics = np.array([[intr['fx']*1, 0, intr['cx']*scale],
                        [0, intr['fy']*1, intr['cy']*scale],
                        [0, 0, 1]], dtype=float)


scaled_K = intrinsics.copy()
offset_h_1 = 500 # x
offset_h_2 = 964 # y
if scale != 1.0:
    offset_h_1 *= scale
    offset_h_2 *= scale

scaled_K[0][2] =  2 * intrinsics[0, 2]
scaled_K[1][2] =  2 * intrinsics[1, 2] - offset_h_1

intrinsics = scaled_K
camera_matrix = scaled_K
# # import pdb;pdb.set_trace()
distortion = np.zeros((4, 1), dtype=float)
#code block ends

#printing distortion and extrinsics
print('distortion', distortion)
# print('instrinsics', intrinsics)
print('extrinsics', camera_matrix)

# import pdb;pdb.set_trace()
import colorsys

def create_color_ramp(num_colors):
  """
  Generates a color ramp cycling through the color spectrum.

  Args:
    num_colors: The number of distinct colors to generate in the ramp.

  Returns:
    A list of RGB color tuples, where each tuple represents a color
    in the ramp and the values are normalized between 0 and 1.
  """
  color_ramp = []
  for i in range(num_colors):
    # Adjust hue (0 to 1), keeping saturation and value constant for a vibrant ramp
    # Hue values cycle through the color wheel, while saturation and value control the intensity
    hue = i / num_colors
    saturation = 1.0  # Full saturation
    value = 1.0       # Full brightness
    rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
    color_ramp.append((int(rgb_color[0]*255),
                       int(rgb_color[1]*255),
                       int(rgb_color[2]*255)))
  return color_ramp

# Example usage:
num_colors_in_ramp = 100 # Adjust as needed
my_color_ramp = create_color_ramp(num_colors_in_ramp)

# import pdb;pdb.set_trace()
def project_to_file(point_cloud_file, image_file, output_file):
  # path the pcd file to load the pcd file
  point_cloud = opd.io.read_point_cloud(point_cloud_file)

  # convert lidar 3d points to image 3d points
#   import pdb;pdb.set_trace()
  points = np.array(point_cloud.points)
  # mask = (points[:, 0] > 0) & (points[:, 0] < 50) & (points[:, 1] >= -15) & (points[:, 1] <= 15) #front
  # mask += (points[:, 1] > 0) & (points[:, 1] < 50) & (points[:, 0] >= -25) & (points[:, 0] <= 25) #left
  # mask += (points[:, 1] < 0) & (points[:, 1] > -25) & (points[:, 0] >= -15) & (points[:, 0] <= 15) #right
  # mask += (points[:, 0] < 0) & (points[:, 1] >= -20) & (points[:, 1] <= 20) #rear
  # import pdb;pdb.set_trace()
  filtered_points = points
  # filtered_points = points[mask]
  all_image_3d_points = []
  for lidar_3d_point in filtered_points:
    image_3d_point = np.matmul(np.linalg.inv(rotation_matrix), lidar_3d_point - translation_vector) #main code 
    all_image_3d_points.append(image_3d_point)
  # import pdb;pdb.set_trace()    
  # read the image file
  image = cv2.imread(image_file)
  # for point in lidar_3d_point:
    
  for image_3d_point in all_image_3d_points:
    if image_3d_point[2] >=0:
      # import pdb;pdb.set_trace() 
      #try to interchangeably use cv2.fisheye.projectPoints and cv2.projectPoints and see what works for different scenarios
      # projected_points, _ = cv2.fisheye.projectPoints(np.array([np.array([image_3d_point], dtype='float32')], dtype='float32'), rotation_vector,
                                                      # translation_vector2, camera_matrix, distortion)
      projected_points, _ = cv2.projectPoints(np.array([np.array([image_3d_point], dtype='float32')], dtype='float32'), rotation_vector, translation_vector2, camera_matrix, distortion)#for undistorted image only the image_3d_point and camera_matrix have values, rest all are zeros 
      projected_point_in_2d = projected_points[0][0]
      if projected_point_in_2d[0] >= 0 and projected_point_in_2d[0] < image.shape[1] and projected_point_in_2d[1] >= 0 and projected_point_in_2d[1] < image.shape[0]:
        image_point_distance = np.linalg.norm(image_3d_point)
        if image_point_distance > 20:
          color_index = 99
        else:
          color_index = int(image_point_distance * 5)
        image = cv2.circle(image,(int(projected_point_in_2d[0]), int(projected_point_in_2d[1])), 3, my_color_ramp[color_index])

  cv2.imwrite(output_file, image)
  print("write to " + output_file)

# import glob
# import os

# pcd_files = glob.glob("*.pcd")

# for pcd_file in pcd_files:
#   base_name, _ = os.path.splitext(pcd_file)
#   project_to_file(pcd_file, base_name + ".jpg", base_name + "_projection.jpg")
image_file = "./data/lucid/fwc_c/0.png" # "../data/cam03/000000.png"
# image_file = "./data/lucid/fnc/0.png"
lidar_file = "./data/lucid/000000.pcd"  # PCD file
output_file = "./cloudvisualization_output.png"
project_to_file(lidar_file, image_file, output_file)