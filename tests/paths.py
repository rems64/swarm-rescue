from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor

from math import sqrt

class Point:
    
    STEP_FOR_EQUALITY = 50
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        
    @property
    def step_for_equality(self):
        self.step_for_equality = self.__class__.STEP_FOR_EQUALITY
        
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'
    
    __repr__ = __str__
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __rmul__(self, λ):
        return Point(λ * self.x, λ * self.y)
    
    def __eq__(self, other):
        return abs(self.x - other.x) < self.step_for_equality and abs(self.y - other.y) < self.step_for_equality
    
    def __ne__(self, other):
        return not (self == other)
        
    def __le__(self, other):
        return other.x - self.x >= 2 * self.step_for_equality and other.y - self.y >= 2 * self.step_for_equality
    
    def __ge__(self, other):
        return self.x - other.x >= 2 * self.step_for_equality and self.y - other.y >= 2 * self.step_for_equality
    
    def __lt__(self, other):
        return other.x - self.x > 2 * self.step_for_equality and other.y - self.y > 2 * self.step_for_equality
    
    def __gt__(self, other):
        return self.x - other.x > 2 * self.step_for_equality and self.y - other.y > 2 * self.step_for_equality
    
    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def getSlope(self, other):
        return (self.y - other.y) / (self.x - other.x)
    
    @classmethod
    def getStepForEquality(cls):
        return cls.STEP_FOR_EQUALITY
    


class Line:
    
    def __init__(self, start: (float), end: (float)):
        self.start = Point(*start)
        self.end = Point(*end)
    
    def __str__(self):
        return '[' + str(self.start) + ' -> ' + str(self.end) + ']'
    
    __repr__ = __str__
    
    def __contains__(self, point: Point):
        xBool = (self.start.x <= point.x <= self.end.x or self.end.x <= point.x <= self.start.x)
        yBool = (self.start.y <= point.y <= self.end.y or self.end.y <= point.y <= self.start.y)
        return xBool and yBool
    
    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end) or (self.start == other.end and self.end == other.start)
    
    def __ne__(self, other):
        return not self == other
    
    def __le__(self, other):
        """Checks if self is contained in other"""
        return self.start >= other.start and self.end <= other.end
    
    def __ge__(self, other):
        """Checks if other is contained in self"""
        return self.start <= other.start and self.end >= other.end
    
    def __len__(self):
        return self.start.distance(self.end)
    
    def extend(self, other):
        if (self.start in other or self.end in other or other.start in self or other.end in self):
            self = Line(min(self.start, other.start), max(self.end, other.end))
        else: raise ValueError('Lines are not connected')
    
    def append(self, point: Point):
        if point <= self.start:
            self = Line(point, self.end)
        elif point >= self.end:
            self = Line(self.start, point)
        elif point in self:
            pass
        else: raise ValueError('Cannot append point to line')
    
    def distanceToPoint(self, point: Point):
        A = self.start
        B = self.end 
        C = point
        d = (A.distance(C) ** 2 - B.distance(C) ** 2 + A.distance(B) ** 2) / (2 * A.distance(B))
        return sqrt(A.distance(C) ** 2 - d ** 2), A + (B - A) * (d / A.distance(B))

    @staticmethod
    def getBeforeDict(lines):
        before = {}
        for line in lines:
            before[line.end] = line.start
        return before
    
    @staticmethod
    def getAfterDict(lines):
        after = {}
        for line in lines:
            after[line.start] = line.end
        return after
    
    @staticmethod
    def reOrganize(lines):
        lines.sort(key=lambda line: line.start)
        for i in range(len(lines) - 1):
            if lines[i].end != lines[i + 1].start:
                raise ValueError('Lines are not connected')
        return lines
    
    @staticmethod
    def getLineFromPoint(lines, point):
        lineFromPoint = {}
        for line in lines:
            if point in line:
                return line 
        return None
    
    
    
class Path:
    
    def __init__(self, drone, *lines):
        lines = Line.reOrganize(list(lines))
        self.lines: [Line] = lines
        self.drone = drone
        self.points = set(line.start for line in lines).union(set(line.end for line in lines))
        self.before = Line.getBeforeDict(lines)
        self.after = Line.getAfterDict(lines)
        self.start = self.lines[0].start
        self.end = self.lines[-1].end
        
    def __str__(self):
        res = '['
        for line in self:
            res += str(line.start) + ' -> '
        res += str(self[-1].end) + ']'
        return res
    
    __repr__ = __str__

    def __len__(self):
        return sum(len(line) for line in self)
    
    def __getitem__(self, index):
        return self.lines[index]
    
    def __setitem__(self, index, value):
        self.lines[index] = value
        
    def __delitem__(self, index):
        del self.lines[index]
    
    def __contains__(self, line):
        return line in self.lines
    
    def append(self, point: Point):
        closestDistance = float('inf')
        closestPoint = None
        closestIndex = None
        for i, line in enumerate(self.lines):
            d, p = line.distanceToPoint(point)
            if d < closestDistance and self.drone.canSee(p):
                closestDistance = d
                closestPoint = p
                closestIndex = i
        
        self.lines[closestIndex].end = closestPoint
        self.lines.append(Line(closestPoint, point))
    
    
    def goHomeAngle(self):
        position = Point(*self.drone.estimeted_gps_position)
        line = Line.getLineFromPoint(self.lines, position)
        self.drone.estimatedAngle = line.start.getSlope(position)


    def goHome(self):
        angle = self.goHomeAngle()
        speed = 1.0
        stride = 0.0
        
        for value in self.drone.semantics:
            if value.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                self.drone.state = "back_safe_zone"
        
        return angle, speed, stride

