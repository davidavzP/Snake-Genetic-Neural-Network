import random
import particle as prt
import objects as obj
import tensorflow as tf
import shape 
import game 
import math

class Apple(obj.Object):
    def __init__(self, x,y,length=1):
        self.pos = (x,y)
        self.length = length
        self.rect = shape.Rectangle(x,y, self.length, self.length)

    def updatePos(self, snake, w,h):        
        x = random.randint(1,w - 2)
        y = random.randint(1,h - 2)

        self.pos = (x,y)

        for p in snake:
            if p == (x,y):
                self.updatePos(snake, w, h)

        self.rect = shape.Rectangle(self.pos[0],self.pos[1], self.length, self.length)
    
    def name(self):
        return 'apple'

    def getPoints(self):
        return self.rect.lines

class Snek(obj.Object):
    def __init__(self, start, direction, size):
        self.size = size
        self.body = start
        self.dirs = [direction for i in range(len(self.body))]
        assert(len(self.body) >= 2)  
        self.hx = self.body[0][0]
        self.hy = self.body[0][1]
        self.headir = self.dirs[0]
        self.growing = False

        self.vision = prt.Particle(self.centerHeadPt())
        self.rects = self.buildRects()
    
    def generate_randSnekPos(s1, s2, w,h):
        length = 3
        start_points = []
        direction = random.randint(1, 4)
        x = random.randint(s1 + 3, w - 4)
        y = random.randint(s2 + 3, h - 4)
        if direction == 1:
            direction = game.Direction.UP
            for i in range(length):
                start_points.append((x, (y+i)))
        if direction == 2:
            direction = game.Direction.DOWN
            for i in range(length):
                start_points.append((x, (y-i)))
        if direction == 3:
            direction = game.Direction.LEFT
            for i in range(length):
                start_points.append(((x+i), y))
        if direction == 4:
            direction = game.Direction.RIGHT
            for i in range(length):
                start_points.append(((x-i), y))
        assert(len(start_points) == length)
        return start_points,direction
    
    def buildRects(self):
        rects = []
        for i in range(2,len(self.body)):
            point = self.body[i]
            rect = shape.Rectangle(point[0], point[1], self.size, self.size)
            rects += rect.lines
        return rects
    
    def push(self, s):
        self.body.insert(0,s)
    
    def dirpush(self, d):
        self.dirs.insert(0, d)
        self.headir = d
    
    def pop(self):
        self.body.pop()
    
    def dirpop(self):
        self.dirs.pop()
        
    def updateVision(self):
        self.vision = prt.Particle(self.centerHeadPt())
    
    def updateHeadDirection(self, d):
        self.headir = d
    
    def updateBody(self):
        self.rects = self.buildRects()

    def centerHeadPt(self):
        x = self.hx + self.size/2
        y = self.hy + self.size/2
        return (x,y)
    
    def getHeadDir(self):
        return self.dirs[0]
    
    def getTailDir(self):
        return self.dirs[-1]
    
    def name(self):
        return 'snake'

    def getPoints(self):
        return self.rects

class Player():
    def __init__(self, bounds, snake, apple, brain):
        self.objects = [bounds]
        self.snake = snake
        self.apple = apple
        self.num_apples = 0
        self.objects.append(snake)
        self.objects.append(apple)
        self.brain = brain

    def removeObject(self, object1):
        self.objects.remove(object1)

    def addObject(self, object1):
        self.objects.append(object1)
    
    def look(self, maximum, minimum):
        inputs = []
        vision = self.snake.vision
        assert(len(self.objects) == 3)
        for ray in vision.rays:
            vals = []
            for obj in self.objects:
                name = obj.name()
                record = maximum
                for line in obj.getPoints():
                   point = ray.cast(line) 
                   if point != None:
                       dist = ray.distanceTo(point[0], point[1])
                       if dist < record:
                            record = dist
                distance = (record - minimum)/ (maximum - minimum)
                vals.append(distance)
            assert(len(vals) == len(self.objects))
            inputs.extend(vals)
        return inputs
    
    def move(self, vision):
        inputs = []
        inputs.extend(vision)
        inputs.extend(self.snake.getHeadDir().value)
        inputs.extend(self.snake.getTailDir().value)
        assert(len(inputs) == 32)
        output = self.brain(inputs)
        
        output = tf.reshape(output, [-1])
        index = tf.math.argmax(output).numpy()
        direction = None
        if index == 0:
            direction = game.Direction.UP
        elif index == 1:
            direction = game.Direction.DOWN
        elif index == 2:
            direction = game.Direction.LEFT
        else:
            direction = game.Direction.RIGHT
        return direction
    
    def setFitness(self, c):
        self.brain.fitness = c

    def calculateFitness(self, frames, apples):
        frames = float(frames)
        apples = float(apples)
        return (20*apples) + (5*frames)
        
        # a = frames + (math.pow(2, apples) + (math.pow(apples, 2.1)*50)) 
        # b = (0.25 * frames)
        # b = math.pow(apples, 1.2) * math.pow(b, 1.3)
        # c = a - b
        # return max(c, 0.1)

    

