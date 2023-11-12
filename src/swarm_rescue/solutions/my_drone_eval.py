import math
import random
import time
from typing import Optional, Union
import arcade
import numpy as np
# import matplotlib
# matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.constants import DRONE_INITIAL_HEALTH, MAX_RANGE_LIDAR_SENSOR

def clamp(x, a, b):
    return max(min(x, b), a)

def coeff(demi_angle):
    # return 1 - abs(demi_angle)/45
    # return np.exp((demi_angle/45)**2)
    return 1

class MyDroneEval(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.last_time = time.time()
        self.follow_left = 1
        self.people_ranges = []
        self.estimated_gps_position:np.ndarray = np.array([0, 0])
        self.estimated_velocity:np.ndarray = np.array([0, 0])
        self.estimated_angle:float = 0
        self.walls_distances = []
        self.semantics = []
        self.points = []
        self.walls = []
        self.blobs = []
        self.grabbed_person = False

        if True:
            plt.figure("walls")
            plt.axis((-300, 300, 0, 300))
            plt.ion()
            plt.show()


        self.state = "follow_wall"

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def is_person(self, angle_deg: float) -> bool:
        angle_rad = angle_deg/180*np.pi
        for person_range in self.people_ranges:
            if person_range[0] < angle_rad < person_range[1]:
                return True
        return False
    
    def grab_person(self):
        speed = 1
        angle = 0
        stride = 0
        for value in self.semantics:
            if value.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and self.grabbed_person==0 :
                angle = value.angle
                if value.distance < 50 :
                    speed = 0.5
                if value.distance < 25 :
                    self.grabbed_person = 1
        return (speed, angle, stride)

    
    def measured_velocity(self) -> Union[np.ndarray, None]:
        """
        Give the measured velocity of the drone in the two dimensions, in pixels per second
        You must use this value for your calculation in the control() function.

        /!\ Replacemement of the original function
        """
        odom = self.odometer_values()
        if odom is not None:
            speed = odom[0]
            vx = speed * math.cos(self.estimated_angle)
            vy = speed * math.sin(self.estimated_angle)
            self.estimated_angle += odom[2]
            return np.array([vx, vy])
        else:
            return None
    
    def update_position(self):
        gps = self.measured_gps_position()
        if gps is not None:
            self.estimated_gps_position = gps
            self.estimated_angle = self.measured_compass_angle()
        else:
            self.estimated_gps_position += self.estimated_velocity
            velocity = self.measured_velocity()
            if velocity is not None:
                self.estimated_velocity = velocity
    
    def update_distances(self):
        lidar_values = self.lidar_values()
        self.walls_distances = np.array([(2*demi_angle, lidar_values[demi_angle+90]*coeff(demi_angle)) for demi_angle in range(-90, 90) if not self.is_person(2*demi_angle)])
    
    def update_semantic(self):
        self.semantics = self.semantic_values()
        self.people_ranges = []
        for value in self.semantics:
            if value.entity_type==DroneSemanticSensor.TypeEntity.WOUNDED_PERSON or value.entity_type==DroneSemanticSensor.TypeEntity.GRASPED_WOUNDED_PERSON:
                r = 12  # Radius of a person
                alpha = np.arctan(r/value.distance)
                self.people_ranges.append((value.angle-alpha, value.angle+alpha))
            if value.entity_type==DroneSemanticSensor.TypeEntity.DRONE:
                r = 10  # Radius of a drone
                alpha = np.arctan(r/value.distance)
                self.people_ranges.append((value.angle-alpha, value.angle+alpha))


    def follow_wall(self):
        dist_min = np.min(self.walls_distances[:, 1])
        # dist_max = np.max(self.walls_distances[:, 1])

        front_rays = [x[1] for x in self.walls_distances if -10<x[0]<10]
        distance_devant = min(front_rays) if len(front_rays)>0 else math.inf

        angle_min = self.walls_distances[np.argmin(self.walls_distances[:,1])][0]
        dist_min = np.min(self.walls_distances[:,1])

        if time.time() - self.last_time > 0.2 or dist_min > 80:
            self.follow_left = np.sign(angle_min)
            self.last_time = time.time()
        
        delta_dir = 90*self.follow_left
        stride = 0
        speed = 1
        side_vector = np.array([np.cos(self.estimated_angle+self.follow_left*np.pi/2), np.sin(self.estimated_angle+self.follow_left*np.pi/2)])
        k = 10
        alpha = 0*2*np.sqrt(k*self.base._mass)
        stride += self.follow_left*k*np.sign(dist_min-20)*clamp((dist_min-20), -1, 1)**2-np.dot(side_vector, self.estimated_velocity)*alpha
        if distance_devant < 60:
            delta_dir += self.follow_left*90
            stride-=self.follow_left*1
            speed = 0.1

        angle = angle_min-delta_dir
        if abs(angle)>15:
            speed *= 0.2
        if abs(angle)>30:
            stride += angle*np.linalg.norm(self.true_velocity())*100
        angle = clamp(angle/180*np.pi, -1, 1)
        stride = clamp(stride, -1, 1)

        return (speed, angle, stride)

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        self.update_position()
        self.update_distances()
        self.update_semantic()
        self.add_walls()

        speed, angle, stride = 1.0, 0.0, 0.0

        match self.state:
            case "follow_wall":
                speed, angle, stride = self.follow_wall()
                for semantic in self.semantics:
                    if semantic.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not self.grabbed_person:
                        self.state = "grab_person"
                        break
            case "grab_person":
                speed, angle, stride = self.grab_person()
                if self.grabbed_person:
                    self.state = "follow_wall"
            case _:
                self.state = "follow_wall"
        command = {"forward": clamp(speed, -1, 1),
                   "lateral": clamp(stride, -1, 1),
                   "rotation": clamp(angle, -1, 1),
                   "grasper": 1.0}

        return command
    
    def add_walls(self):
        drone_position = self.true_position() + self._half_size_array
        drone_angle = self.true_angle()
        points = []
        for angle, distance in self.walls_distances:
            if distance < MAX_RANGE_LIDAR_SENSOR-10:
                x = drone_position[0]+np.cos(np.deg2rad(angle)+drone_angle)*distance
                y = drone_position[1]+np.sin(np.deg2rad(angle)+drone_angle)*distance
                point = np.array([x, y])
                further_enough = True
                for point_other in self.points:
                    if np.dot(point-point_other, point-point_other) < (20)**2:
                        further_enough = False
                        break
                if further_enough:
                    points.append(point)
        for point in points:
            self.points.append(point)
            self.extend_blobs(point)
    
    def distance_to_blob(self, point, blob):
        # # Manhatan distance
        # if blob[0][0]<point[0]<blob[1][0] and blob[0][1]<point[1]<blob[1][1]:
        #     return 0
        # return min(abs(point[0]-blob[0][0])+abs(point[1]-blob[0][1]), abs(point[0]-blob[1][0])+abs(point[1]-blob[1][1]))
        if blob[0][0]<point[0]<blob[1][0] and blob[0][1]<point[1]<blob[1][1]:
            return 0
        if abs(blob[0][0]-blob[1][0])>abs(blob[0][1]-blob[1][1]):
            # Horizontal
            if min(abs(point[0]-blob[0][0]), abs(point[0]-blob[1][0]))<min(abs(point[1]-blob[0][1]), abs(point[1]-blob[1][1])):
                # Not aligned
                return math.inf
            else:
                return min(abs(point[0]-blob[0][0]), abs(point[0]-blob[1][0]))
        else:
            # Vertical
            if min(abs(point[1]-blob[0][1]), abs(point[1]-blob[1][1]))<min(abs(point[0]-blob[0][0]), abs(point[0]-blob[1][0])):
                # Not aligned
                return math.inf
            else:
                return min(abs(point[1]-blob[0][1]), abs(point[1]-blob[1][1]))
    
    def extend_blobs(self, point):
        if len(self.blobs) == 0:
            self.blobs.append([point, point.copy()])
            return
        min_distance = math.inf
        closest_blob = None
        for blob in self.blobs:
            distance = self.distance_to_blob(point, blob)
            if distance < 50 and distance < min_distance:
                min_distance = distance
                closest_blob = blob
        if closest_blob is None:
            self.blobs.append([point, point.copy()])
            return
        self.extend_blob(closest_blob, point)
    
    def distance_between_blobs(self, blob1, blob2):
        # Distance between rects
        if blob1[0][0]<blob2[1][0] and blob1[1][0]>blob2[0][0] and blob1[0][1]<blob2[1][1] and blob1[1][1]>blob2[0][1]:
            return 0
        return min(distance_to_segment(blob1[0], blob2[0], blob2[1]), distance_to_segment(blob1[1], blob2[0], blob2[1]), distance_to_segment(blob2[0], blob1[0], blob1[1]), distance_to_segment(blob2[1], blob1[0], blob1[1]))
    
    def extend_blob(self, blob, point):
        blob[0] = np.minimum(blob[0], point)
        blob[1] = np.maximum(blob[1], point)
        to_remove = None
        for (i, blob_other) in enumerate(self.blobs):
            if blob_other is not blob and self.distance_between_blobs(blob, blob_other) < 50:
                # Check alignment
                blob[0] = np.minimum(blob[0], blob_other[0])
                blob[1] = np.maximum(blob[1], blob_other[1])
                to_remove = i
                break
        if to_remove is not None:
            self.blobs.pop(to_remove)
    
    def extend_wall(self, wall, point):
        # We extend the wall and keep it horizontal or vertical
        if np.dot(point-wall[0], wall[1]-wall[0]) or np.dot(point-wall[1], wall[0]-wall[1]):
            return
        if np.dot(point-wall[0], point-wall[0]) > np.dot(point-wall[1], point-wall[1]):
            if abs(wall[0][0]-point[0]) > abs(wall[0][1]-point[1]):
                point[1] = wall[0][1]
            else:
                point[0] = wall[0][0]
        else:
            if abs(wall[1][0]-point[0]) > abs(wall[1][1]-point[1]):
                point[1] = wall[1][1]
            else:
                point[0] = wall[1][0]
        wall[0] = point
        # if abs(wall[0][0]-wall[1][0]) > abs(wall[0][1]-wall[1][1]):
        #     wall[0][1] = wall[1][1]
        # else:
        #     wall[0][0] = wall[1][0]
    
    def draw_health(self):
        position = self.true_position() + self._half_size_array
        width = 50
        alpha = self.drone_health/DRONE_INITIAL_HEALTH
        color = arcade.color.GREEN if self.drone_health==DRONE_INITIAL_HEALTH else arcade.color.YELLOW if self.drone_health>DRONE_INITIAL_HEALTH/2 else arcade.color.ORANGE if self.drone_health>DRONE_INITIAL_HEALTH/4 else arcade.color.RED
        y_offset = 30
        arcade.draw_line(position[0]-width/2, position[1]+y_offset, position[0]+width/2, position[1]+y_offset, arcade.color.GRAY, 3)
        arcade.draw_line(position[0]-width/2, position[1]+y_offset, position[0]-width/2+alpha*width, position[1]+y_offset, color, 3)
    
    def draw_estimated_position(self):
        position_estimated = self.estimated_gps_position + self._half_size_array
        arcade.draw_circle_filled(position_estimated[0], position_estimated[1], 5, arcade.color.RED)
    
    def draw_wall_side(self):
        position = self.true_position() + self._half_size_array
        angle = self.true_angle()+self.follow_left*np.pi/2
        arcade.draw_line(position[0], position[1], np.cos(angle)*50+position[0], np.sin(angle)*50+position[1], arcade.color.RED, 3)
    
    def draw_top_layer(self):
        super().draw_top_layer()

        self.draw_estimated_position()
        self.draw_wall_side()
        self.draw_health()
    
    def display(self):
        DroneAbstract.display(self)
        self.display_walls()

    def display_walls(self):
        if self.lidar_values() is not None:
            plt.figure("walls")
            plt.cla()
            plt.axis((-30, 1000, -30, 1000))
            plt.plot([x[0] for x in self.points], [x[1] for x in self.points], "o", markersize=0.5, color="blue")
            for wall in self.walls:
                plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], "-", color="black", linewidth=2)
            for blob in self.blobs:
                plt.plot([blob[0][0], blob[0][0], blob[1][0], blob[1][0], blob[0][0]], [blob[0][1], blob[1][1], blob[1][1], blob[0][1], blob[0][1]], "-", color="red", linewidth=2)
            # plt.plot([100, 100], [500, 100], "-", linewidth=2)
            # plt.plot([100, 500], [100, 100], "-", linewidth=2)
            plt.grid(False)
            # plt.draw()
            # plt.show()
            plt.pause(0.0001)

def distance_to_segment(p, v, w):
    vw = w-v
    l2 = np.dot(vw, vw)
    if (l2 == 0.0): return np.linalg.norm(p-v)
    t = max(0, min(1, np.dot(p - v, w - v) / l2))
    projection = v + t * (w - v)
    return np.linalg.norm(p-projection)
