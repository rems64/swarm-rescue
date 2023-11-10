import math
import random
import time
from typing import Optional, Union
import arcade
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.constants import DRONE_INITIAL_HEALTH

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
        self.persones_ranges = []
        self.estimated_gps_position:np.ndarray = np.array([0, 0])
        self.estimated_velocity:np.ndarray = np.array([0, 0])
        self.estimated_angle:float = 0

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def is_person(self, angle_deg: float) -> bool:
        angle_rad = angle_deg/180*np.pi
        for person_range in self.persones_ranges:
            if person_range[0] < angle_rad < person_range[1]:
                return True
        return False
    
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


    def process_lidar_sensor(self):
        """
        Returns the direction of the largest free zone
        """
        lidar_values = self.lidar_values()
        semantic_values = self.semantic_values()

        self.persones_ranges = []
        for value in semantic_values:
            if value.entity_type==DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                r = 12  # Radius of a person
                alpha = np.arctan(r/value.distance)
                self.persones_ranges.append((value.angle-alpha, value.angle+alpha))

        if lidar_values is None:
            return 0
        
        dists = np.array([(2*demi_angle, lidar_values[demi_angle+90]*coeff(demi_angle)) for demi_angle in range(-90, 90) if not self.is_person(2*demi_angle)])

        dist_min = np.min(dists[:, 1])
        dist_max = np.max(dists[:, 1])

        distance_devant = min([x[1] for x in dists if -10<x[0]<10])
        # toto = sum(lidar_values)
        # dists = np.array([2*angle*(lidar_values[angle+90]/toto) for angle in range(-90, 90)])
        # angle = dists.sum()
        angle_min = dists[np.argmin(dists[:,1])][0]
        dist_min = np.min(dists[:,1])

        if time.time() - self.last_time > 0.2 or dist_min > 80:
            self.follow_left = np.sign(angle_min)
            self.last_time = time.time()
        
        delta_dir = 90*self.follow_left
        stride = 0
        speed = 1
        side_vector = np.array([np.cos(self.estimated_angle+self.follow_left*np.pi/2), np.sin(self.estimated_angle+self.follow_left*np.pi/2)])
        k = 10
        alpha = 0*2*np.sqrt(k*self.base._mass)
        stride += self.follow_left*k*np.sign(dist_min-30)*clamp((dist_min-30), -1, 1)**2-np.dot(side_vector, self.estimated_velocity)*alpha
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
        command = {"forward": 0.8,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        speed, angle, stride = self.process_lidar_sensor()
        self.measured_gps_position()
        command["forward"] = speed
        command["rotation"] = angle
        command["lateral"] = stride

        # measured_angle = 0
        # if self.measured_compass_angle() is not None:
        #     measured_angle = self.measured_compass_angle()

        # diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
        # if self.isTurning and abs(diff_angle) < 0.2:
        #     self.isTurning = False
        #     self.counterStraight = 0
        #     self.distStopStraight = random.uniform(10, 50)

        # if self.isTurning:
        #     return command_turn
        # else:
        #     return command_straight
        return command
    
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