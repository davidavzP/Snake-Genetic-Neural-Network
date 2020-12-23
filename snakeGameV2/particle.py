import math
import game

class Particle():
    def __init__(self, pos, direction):
        self.pos = pos
        self.rays = []
        for i in range(0, 360, 45):
            ray = Ray(pos[0], pos[1], direction, math.radians(i))
            self.rays.append(ray)

        
class Ray():
    def __init__(self, x,y, direction, angle=0.0):
        self.pos = (x,y)
        t1 = 0
        t2 = 1
        if direction.value == game.Direction.UP.value:
            t1 = 0
            t2 = -1
        if direction.value == game.Direction.DOWN.value:
            t1 = 0
            t2 = 1
        if direction.value == game.Direction.LEFT.value:
            t1 = -1
            t2 = 0
        if direction.value == game.Direction.RIGHT.value:
            t1 = 1
            t2 = 0
        self.dir = self.rotate(t1,t2,angle)
    
    def rotate(self,x,y,a):
        x1 = x*math.cos(a) - y*math.sin(a)
        y1 = x*math.sin(a) + y*math.cos(a)
        return (x1,y1)

    def distanceTo(self,ox, oy):
        return math.sqrt(((ox - self.pos[0])**2) + ((oy - self.pos[1])**2))

    def lookAt(self, x,y):
        self.dir[0] = x - self.pos[0]
        self.dir[1] = y - self.pos[1]
        norm = math.sqrt(self.dir[0]**2, self.dir[1]**2)
        self.dir[0] = self.dir[0] / norm
        self.dir[1] = self.dir[1] / norm
    
    def cast(self,bound):
        x1 = bound[0]
        y1 = bound[1]
        x2 = bound[2]
        y2 = bound[3]

        x3 = self.pos[0]
        y3 = self.pos[1]
        x4 = self.pos[0] + self.dir[0]
        y4 = self.pos[1] + self.dir[1]

        den = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if den == 0:
            return None
        t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / den
        u = - ((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / den
        if t >= 0 and t <= 1 and u >= 0:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x,y)
        else:
            return None
