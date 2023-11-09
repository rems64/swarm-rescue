import math
import random
import time
from typing import Optional
import arcade
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

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
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False
        self.last_time = time.time()
        self.follow_left = 1
        self.persones_ranges = []

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def is_person(self, angle: float) -> bool:
        for person_range in self.persones_ranges:
            if person_range[0] < angle < person_range[1]:
                print("person detected")
                return True
        return False

    def process_lidar_sensor(self):
        """
        Returns the direction of the largest free zone
        """
        lidar_values = self.lidar_values()
        semantic_values = self.semantic_values()

        self.persones_ranges = []
        for value in semantic_values:
            if value.entity_type==DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                r = 10  # Radius of a person
                alpha = np.arctan(r/value.distance)
                self.persones_ranges.append((value.angle-alpha, value.angle+alpha))

        if lidar_values is None:
            return 0
        
        dists = np.array([lidar_values[demi_angle+90]*coeff(demi_angle) for demi_angle in range(-90, 90) if not self.is_person(2*demi_angle)])

        dist_min = min(dists)
        dist_max = max(dists)

        distance_devant = max(dists[85:95])
        # toto = sum(lidar_values)
        # dists = np.array([2*angle*(lidar_values[angle+90]/toto) for angle in range(-90, 90)])
        # angle = dists.sum()
        
        angle_min = 2*(np.argmin(dists)-90)
        dist_min = min(dists)

        if time.time() - self.last_time > 0.2 or dist_min > 80:
            self.follow_left = np.sign(angle_min)
            self.last_time = time.time()
        
        delta_dir = 90*self.follow_left
        stride = 0
        speed = 1
        if self.follow_left > 0:
            # On suit le mur en le gardant à droite
            stride += np.sign(dist_min-30)*clamp((dist_min-30)/5, -1, 1)**2
            if distance_devant < 100:
                delta_dir += 90
                stride-=1
                speed = 0.1
            # if dist_min < 80:
            #     delta_dir += 10
            # else:
            #     delta_dir -= 10
        else:
            stride += -np.sign(dist_min-30)*clamp((dist_min-30)/5, -1, 1)**2
            if distance_devant < 100:
                delta_dir += -90
                stride+=1
                speed = 0.1
            # if dist_min < 80:
            #     delta_dir -= 10
            # else:
            #     delta_dir += 10
            # On suit le mur en le gardant à gauche

        angle = angle_min-delta_dir
        angle = clamp(angle/180*np.pi, -1, 1)
        stride = clamp(stride, -1, 1)

        return (speed, angle, stride)

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        command = {"forward": 0.8,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        speed, angle, stride = self.process_lidar_sensor()
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
    
    def draw_top_layer(self):
        super().draw_top_layer()
        position = self.true_position() + self._half_size_array
        angle = self.true_angle()+self.follow_left*np.pi/2
        arcade.draw_line(position[0], position[1], np.cos(angle)*50+position[0], np.sin(angle)*50+position[1], arcade.color.RED, 3)
        # arcade.draw_circle_outline(position[0], position[1], 50, arcade.color.RED, 1)