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


from .paths import Path, Point


def clamp(x, a, b):
    return max(min(x, b), a)


def coeff(demi_angle):
    # return 1 - abs(demi_angle)/45
    # return np.exp((demi_angle/45)**2)
    return 1


def anglegps(xd, yd, x, y):
    x, y = xd - x, yd - y
    angle = np.arctan(y / x)
    if x < 0:
        if y > 0:
            angle = np.pi - angle
        else:
            angle = angle - np.pi
    return angle


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def clamp2(angle):
    if abs(angle) > np.pi:
        angle = 2 * np.pi - angle
    return angle


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
        self.path = Path(self)
        self.estimated_gps_position: np.ndarray = np.array([0, 0])
        self.estimated_velocity: np.ndarray = np.array([0, 0])
        self.estimated_angle: float = 0
        self.walls_distances = []
        self.semantics = []
        self.grabbed_person = False
        self.safe_zone = False
        self.state = "follow_wall"
        self.spawn = True
        self.pos_safe_zone = [0, 0]
        self.pos_far_safe_zone = None

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def is_person(self, angle_deg: float) -> bool:
        angle_rad = angle_deg / 180 * np.pi
        for person_range in self.people_ranges:
            if person_range[0] < angle_rad < person_range[1]:
                return True
        return False

    def mur_directif(self, angle):
        return np.array([(self.lidar_values()[(demi_angle + 90)%181] * coeff(demi_angle)) for demi_angle in
                         range(angle - 15, angle + 15) if
                         not self.is_person(2 * demi_angle)])

    def grab_person(self):
        speed = 1
        angle = 0
        stride = 0
        L = []
        for value in self.semantics:
            if value.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not value.grasped and self.grabbed_person == 0:
                L.append((value.distance, value.angle))
        if L:
            speed, angle = min(L)
            if speed < 25:
                self.grabbed_person = True
                self.state = 'back_safe_zone'
                self.pos_far_safe_zone = self.estimated_gps_position
            stride = angle
            speed = np.pi / 2 / 1 + np.exp(value.distance / 50)
        else:
            self.state = 'follow_wall'
        return (speed, angle, stride)

    def back_zone(self, speed, angle, stride):
        L = []
        for value in self.semantics:
            if value.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and self.grabbed_person == 1:
                L.append((value.distance, value.angle))
        if L:
            speed, angle = min(L)
            if speed < 30:
                self.grabbed_person = False
                self.safe_zone = False
                self.state = "back_by_gps"
            stride = angle
            speed = np.pi / 2 / 1 + np.exp(value.distance / 50)
        return (speed, angle, stride)

    def back_zone_gps(self, speed, angle, stride):
        x1, y1 = self.pos_safe_zone
        x2, y2 = self.estimated_gps_position
        angle = anglegps(x1, y1, x2, y2) - self.estimated_angle
        angle = clamp2(angle)
        Murs=self.mur_directif(int((180*angle/np.pi/2)))
        if len(Murs) >= 1 and min(Murs) < 37:
            self.state = 'follow_wall'
        speed = 1
        stride = angle

        return speed, angle, stride

    def search_by_gps(self):
        xd, yd = self.pos_far_safe_zone
        x, y = self.estimated_gps_position
        if distance(xd, yd, x, y) < 30:
            self.state = "follow_wall"
        angle = anglegps(xd, yd, x, y) - self.estimated_angle
        angle = clamp2(angle)
        speed = 1
        stride = angle
        return speed, angle, stride

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
        self.walls_distances = np.array(
            [(2 * demi_angle, lidar_values[demi_angle + 90] * coeff(demi_angle)) for demi_angle in range(-90, 90) if
             not self.is_person(2 * demi_angle)])

    def update_semantic(self):
        self.safe_zone = False
        self.semantics = self.semantic_values()
        self.people_ranges = []
        for value in self.semantics:
            if value.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON or value.entity_type == DroneSemanticSensor.TypeEntity.GRASPED_WOUNDED_PERSON:
                r = 12  # Radius of a person
                alpha = np.arctan(r / value.distance)
                self.people_ranges.append((value.angle - alpha, value.angle + alpha))
            if value.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                r = 10  # Radius of a drone
                alpha = np.arctan(r / value.distance)
                self.people_ranges.append((value.angle - alpha, value.angle + alpha))
            elif value.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                self.safe_zone = True

    def follow_wall(self):
        dist_min = np.min(self.walls_distances[:, 1])
        # dist_max = np.max(self.walls_distances[:, 1])

        front_rays = [x[1] for x in self.walls_distances if -10 < x[0] < 10]
        distance_devant = min(front_rays) if len(front_rays) > 0 else math.inf

        angle_min = self.walls_distances[np.argmin(self.walls_distances[:, 1])][0]
        dist_min = np.min(self.walls_distances[:, 1])

        if time.time() - self.last_time > 0.2 or dist_min > 80:
            self.follow_left = np.sign(angle_min)
            self.last_time = time.time()

        delta_dir = 90 * self.follow_left
        stride = 0
        speed = 1
        side_vector = np.array([np.cos(self.estimated_angle + self.follow_left * np.pi / 2),
                                np.sin(self.estimated_angle + self.follow_left * np.pi / 2)])
        k = 10
        alpha = 0 * 2 * np.sqrt(k * self.base._mass)
        stride += self.follow_left * k * np.sign(dist_min - 20) * clamp((dist_min - 20), -1, 1) ** 2 - np.dot(
            side_vector, self.estimated_velocity) * alpha
        if distance_devant < 60:
            delta_dir += self.follow_left * 90
            stride -= self.follow_left * 1
            speed = 0.1

        angle = angle_min - delta_dir
        if abs(angle) > 15:
            speed *= 0.2
        if abs(angle) > 30:
            stride += angle * np.linalg.norm(self.true_velocity()) * 100
        angle = clamp(angle / 180 * np.pi, -1, 1)
        stride = clamp(stride, -1, 1)

        return (speed, angle, stride)

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        self.update_position()
        self.update_semantic()
        self.update_distances()

        if self.spawn:
            self.spawn = False
            self.pos_safe_zone = self.estimated_gps_position

        speed, angle, stride = 1.0, 0.0, 0.0
        match self.state:
            case "follow_wall":
                speed, angle, stride = self.follow_wall()
                for semantic in self.semantics:
                    if semantic.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not self.grabbed_person:
                        self.state = "grab_person"
                    if semantic.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and self.grabbed_person == 1:
                        self.state = 'back_safe_zone'
                        break
            case "grab_person":
                if self.grabbed_person == 0:
                    speed, angle, stride = self.grab_person()
                else:
                    self.state = "follow_wall"
            case "back_safe_zone":
                speed, angle, stride = self.back_zone_gps(speed, angle, stride)
                speed, angle, stride = self.back_zone(speed, angle, stride)
            case "back_by_gps":
                for semantic in self.semantics:
                    if semantic.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not self.grabbed_person:
                        self.state = "grab_person"
                speed, angle, stride = self.search_by_gps()
            case _:
                self.state = "follow_wall"
        command = {"forward": clamp(speed, -1, 1),
                   "lateral": clamp(stride, -1, 1),
                   "rotation": clamp(angle, -1, 1),
                   "grasper": self.grabbed_person}

        return command

    def draw_health(self):
        position = self.true_position() + self._half_size_array
        width = 50
        alpha = self.drone_health / DRONE_INITIAL_HEALTH
        color = arcade.color.GREEN if self.drone_health == DRONE_INITIAL_HEALTH else arcade.color.YELLOW if self.drone_health > DRONE_INITIAL_HEALTH / 2 else arcade.color.ORANGE if self.drone_health > DRONE_INITIAL_HEALTH / 4 else arcade.color.RED
        y_offset = 30
        arcade.draw_line(position[0] - width / 2, position[1] + y_offset, position[0] + width / 2,
                         position[1] + y_offset, arcade.color.GRAY, 3)
        arcade.draw_line(position[0] - width / 2, position[1] + y_offset, position[0] - width / 2 + alpha * width,
                         position[1] + y_offset, color, 3)

    def draw_estimated_position(self):
        position_estimated = self.estimated_gps_position + self._half_size_array
        arcade.draw_circle_filled(position_estimated[0], position_estimated[1], 5, arcade.color.RED)

    def draw_wall_side(self):
        position = self.true_position() + self._half_size_array
        angle = self.true_angle() + self.follow_left * np.pi / 2
        arcade.draw_line(position[0], position[1], np.cos(angle) * 50 + position[0], np.sin(angle) * 50 + position[1],
                         arcade.color.RED, 3)

    def draw_top_layer(self):
        super().draw_top_layer()

        self.draw_estimated_position()
        self.draw_wall_side()
        self.draw_health()
