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


WALL_POINTS_DISTANCE = 50

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
        self.points_walls = []
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
        if distance_devant < (120 if self.grabbed_person else 60):
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
        # Bon ça c'est la partie embêtante...on peut pas utiliser ça en fait
        # drone_position = self.measured_gps_position() + self._half_size_array
        # drone_angle = self.estimated_angle
        start_index = len(self.points)
        for angle, distance in self.walls_distances:
            if distance < MAX_RANGE_LIDAR_SENSOR-10:
                x = drone_position[0]+np.cos(np.deg2rad(angle)+drone_angle)*distance
                y = drone_position[1]+np.sin(np.deg2rad(angle)+drone_angle)*distance
                point = np.array([x, y])
                further_enough = True
                for point_other in self.points:
                    if np.linalg.norm(point-point_other) < WALL_POINTS_DISTANCE:
                        further_enough = False
                        break
                if further_enough:
                    self.points.append(point)
                    self.points_walls.append(None)
        
        for i in range(start_index, len(self.points)):
            point = self.points[i]
            for j in range(0, i):
                point_other = self.points[j]
                if WALL_POINTS_DISTANCE/10 <= np.linalg.norm(point-point_other) < 2*WALL_POINTS_DISTANCE:
                    print("Adding wall", i, j)
                    self.insert_wall(i, j)
    
    def merge_walls(self, i, j, i_point, index_2_point):
        wall_1 = self.walls[i]
        wall_2 = self.walls[j]
        if wall_1[0]==i_point:
            if wall_2[0]==index_2_point:
                self.walls[i] = (wall_2[1], wall_1[1])
            else:
                self.walls[i] = (wall_1[1], wall_2[0])
        else:
            if wall_2[0]==index_2_point:
                self.walls[i] = (wall_2[1], wall_1[0])
            else:
                self.walls[i] = (wall_1[0], wall_2[0])
        self.walls[j] = None
        # Every point pointing to wall_2 should now point to wall_1
        for k in range(len(self.points_walls)):
            if self.points_walls[k] is None:
                continue
            for l in range(len(self.points_walls[k])):
                if self.points_walls[k][l] == j:
                    self.points_walls[k][l] = i
    
    def insert_wall(self, i, j):
        colinear_criteria = np.pi/8
        potential_wall_direction = (self.points[j]-self.points[i])/np.linalg.norm(self.points[j]-self.points[i])
        angle = np.angle(potential_wall_direction[0]+potential_wall_direction[1]*1j)
        angle = np.round(angle/(np.pi/2))*np.pi/2
        potential_wall_direction = np.array([np.cos(angle), np.sin(angle)])
        if self.points_walls[i] is not None and self.points_walls[j] is not None:
            # We aldready know the two ends
            print("Two ends already known")
            for k in range(len(self.points_walls[i])):
                for l in range(len(self.points_walls[j])):
                    if self.points_walls[i][k] == self.points_walls[j][l]:
                        # The two ends are already connected
                        return
            # We merge the two walls if they are colinear enough
            for k in range(len(self.points_walls[i])):
                for l in range(len(self.points_walls[j])):
                        wall_1 = self.walls[self.points_walls[i][k]]
                        wall_2 = self.walls[self.points_walls[j][l]]
                        wall_1_dir = self.points[wall_1[1]]-self.points[wall_1[0]]
                        wall_1_dir /= np.linalg.norm(wall_1_dir)
                        wall_2_dir = self.points[wall_2[1]]-self.points[wall_2[0]]
                        wall_2_dir /= np.linalg.norm(wall_2_dir)
                        perpendicular = np.arccos(abs(np.dot(wall_1_dir, wall_2_dir))) > colinear_criteria
                        if not perpendicular:
                            print("Merging two walls")
                            self.merge_walls(self.points_walls[i][k], self.points_walls[j][l], i, j)
                            return
            # We fallback to extending on either side
        elif self.points_walls[i] is not None:
            wall = self.walls[self.points_walls[i][0]]
            wall_dir = self.points[wall[1]]-self.points[wall[0]]
            wall_dir = wall_dir / np.linalg.norm(wall_dir)
            # print(np.dot(potential_wall_direction, wall_dir))
            perpendicular = np.arccos(abs(np.dot(potential_wall_direction, wall_dir))) > colinear_criteria
            if wall[0]==i:
                if perpendicular:
                    self.walls.append((i, j))
                    new_wall_id = len(self.walls)-1
                    self.points_walls[j] = [new_wall_id]
                    self.points_walls[i].append(new_wall_id)
                else:
                    self.walls[self.points_walls[i][0]] = (j, wall[1])
                    self.points_walls[j] = self.points_walls[i]
            else:
                if perpendicular:
                    self.walls.append((i, j))
                    new_wall_id = len(self.walls)-1
                    self.points_walls[j] = [new_wall_id]
                    self.points_walls[i].append(new_wall_id)
                else:
                    self.walls[self.points_walls[i][0]] = (wall[0], j)
                    self.points_walls[j] = self.points_walls[i]
        elif self.points_walls[j] is not None:
            wall = self.walls[self.points_walls[j][0]]
            wall_dir = self.points[wall[1]]-self.points[wall[0]]
            wall_dir /= np.linalg.norm(wall_dir)
            # print(np.dot(potential_wall_direction, wall_dir))
            perpendicular = np.arccos(abs(np.dot(potential_wall_direction, wall_dir))) > colinear_criteria
            if wall[0]==j:
                if perpendicular:
                    self.walls.append((i, j))
                    new_wall_id = len(self.walls)-1
                    self.points_walls[i] = [new_wall_id]
                    self.points_walls[j].append(new_wall_id)
                else:
                    self.walls[self.points_walls[j][0]] = (i, wall[1])
                    self.points_walls[i] = self.points_walls[j]
            else:
                if perpendicular:
                    self.walls.append((i, j))
                    new_wall_id = len(self.walls)-1
                    self.points_walls[i] = [new_wall_id]
                    self.points_walls[j].append(new_wall_id)
                else:
                    self.walls[self.points_walls[j][0]] = (wall[0], i)
                    self.points_walls[i] = self.points_walls[j]
        else:
            new_wall_id = len(self.walls)
            self.walls.append((i, j))
            self.points_walls[i] = [new_wall_id]
            self.points_walls[j] = [new_wall_id]

    
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
            plt.axis((-30, 1700, -30, 1200))
            # plt.axis((-30, 900, -30, 600))
            plt.plot([x[0] for x in self.points], [x[1] for x in self.points], "o", markersize=0.5, color="blue")
            # for (i, point) in enumerate(self.points_walls):
            #     if point is not None:
            #         for (j, wall) in enumerate(point):
            #             plt.text(self.points[i][0], self.points[i][1]+j*40, "("+str(i)+") "+str(wall))
            # print(len([x for x in self.walls if x is not None]))
            for wall in self.walls:
                if wall is not None:
                    plt.plot([self.points[wall[0]][0], self.points[wall[1]][0]], [self.points[wall[0]][1], self.points[wall[1]][1]], "-", color="black", linewidth=0.5)
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
