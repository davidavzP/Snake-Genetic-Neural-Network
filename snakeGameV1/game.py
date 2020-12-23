from PyQt5.QtWidgets import * 
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import numpy as np
import objects as obj
import particle as prt
import snek as sk
import gnn
import shape
import sys
import random
import enum
import time
import math


class Game():
    def start(self):
        # create pyqt5 app 
        App = QApplication([]) 
    
        # create the instance of our Window 
        window = Window() 
    
        # start the app 
        sys.exit(App.exec()) 


class Window(QMainWindow): 
    def __init__(self): 
        super(Window, self).__init__() 
        
        self.gameSize = (0, 0, 600, 600)
        self.textSpace = 200
        self.windowSize = [self.gameSize[2], self.gameSize[3] + self.textSpace]
        self.BLOCKS = self.gameSize[2] / 60
        self.board = Board(self)

        self.setCentralWidget(self.board) 
        #WINDOW: (x,y), SIZE: (w,h) 
        self.setGeometry(400, 100, self.windowSize[0], self.windowSize[1])

        self.board.start()

        self.show()

    
class Board(QFrame):
        SPEED = 80

        def __init__(self, parent):
            super(Board, self).__init__(parent)
            self.BLOCKS = parent.BLOCKS
            self.gameSize = parent.gameSize
            self.gameLength = self.gameSize[2]
            self.frameLimit = 100
            self.initText()
            self.initBoard2()
            self.setFocusPolicy(Qt.StrongFocus)
            

           
        def initBoardTest(self):
            self.players = []
            self.dead = []
            self.bounds = GameBounds(shape.Rectangle(0,0,self.BLOCKS,self.BLOCKS))
            snake = sk.Snek([(1,6), (1,7), (1,8)], Direction.UP, 1)
            apple = sk.Apple(6, 6, 1)
            brain = gnn.get_rand_Agent()
            player = sk.Player(self.bounds, snake, apple, brain)
            self.players.append(player)
            
        
        def initBoard2(self):
            self.players = []
            self.best = []
            self.dead = []
            self.bounds = GameBounds(shape.Rectangle(0,0,self.BLOCKS,self.BLOCKS))
            for i in range(250):
                points, direction = sk.Snek.generate_randSnekPos(0,0,self.BLOCKS,self.BLOCKS)
                snake = sk.Snek(points, direction, 1)
                apple = sk.Apple(6, 6, 1)
                apple.updatePos(snake.body, self.BLOCKS, self.BLOCKS)
                brain = gnn.get_rand_Agent()
                player = sk.Player(self.bounds, snake, apple, brain)
                self.players.append(player)
            self.best = self.players[0]
        
        def initText(self):
            self.frames = 0
            self.generation = 1
            self.lbcount = QLabel("Frames: ", self)
            self.count = QLabel(str(self.frames), self) 
            self.lbgens = QLabel("Generations: ", self)
            self.gens = QLabel(str(self.generation), self)
  
            # setting geometry 
            self.count.setGeometry(70, 610, 50, 50) 
            self.lbcount.setGeometry(10, 610, 100, 50)
            self.lbgens.setGeometry(10, 630, 150, 50)
            self.gens.setGeometry(120, 630, 200, 50)
    
            # setting alignment 
            self.count.setAlignment(Qt.AlignCenter) 
    
            # setting font 
            font = QFont('Times', 14)
            self.count.setFont(font) 
            self.lbcount.setFont(font)
            self.lbgens.setFont(font)
            self.gens.setFont(font)

    
        def square_length(self):
            return int(self.gameLength / self.BLOCKS)
        
        def start(self):
            self.timer = QBasicTimer()
            self.timer.start(Board.SPEED, self)
        
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            ##Game Boarder
            self.draw_boarder(painter)
            ##############

            ##Apple
            ###applepos = self.apple.position() 
            
            player = self.best
            applepos = player.apple.pos
            self.draw_square(painter, applepos[0]*self.square_length(), applepos[1]*self.square_length(), QColor(0xFF0000))
            #######
            ###snake = self.snake
            snake = player.snake
            for pos in snake.body:
                x =  pos[0] * self.square_length()
                y = pos[1] * self.square_length()
                self.draw_square(painter, x, y, QColor(0x228B22))

            #self.drawSnekVision(painter, self.players[0])

        # def drawSnekVision(self,painter, player):
        #     painter.setPen(QPen(Qt.black, 1))
        #     vision = player.snake.vision
        #     for ray in vision.rays:
        #         closest = None
        #         record = math.inf
        #         for o in player.objects:
        #             for line in o.getPoints():
        #                 point = ray.cast(line)
        #                 if point != None:
        #                     dist = ray.distanceTo(point[0], point[1])
        #                     if dist < record:
        #                         record = dist
        #                         closest = point
        #         if closest != None:
        #             pos = ray.pos
        #             dr = ray.dir
        #             x1 = pos[0]*self.square_length()
        #             y1 = pos[1]*self.square_length()
        #             x2 = x1 + dr[0]*100
        #             y2 = y1 + dr[1]*100
        #             painter.drawLine(x1,y1,closest[0]*self.square_length(),closest[1]*self.square_length())
                
        def draw_boarder(self, painter):
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(1,1, self.gameLength - 1, self.gameLength)

        def draw_square(self, painter, x, y, color):
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(x+ 1, y+1, self.square_length(), self.square_length())
            painter.setOpacity(0.5)
            painter.fillRect(x+ 1, y+1, self.square_length(), self.square_length() , color)

        def timerEvent(self, event):
            if event.timerId() == self.timer.timerId():
                
                if len(self.players) > 0:
                    if self.frames < self.frameLimit:
                        for player in self.players:
                            inputs = player.look(self.BLOCKS, 0.0)
                            output = player.move(inputs)
                            player.snake.updateHeadDirection(output)
                            self.move(player)
                            self.apple_ate(player)
                            self.is_dead(player)
                        self.frames += 1
                    else:
                        self.dead.extend(self.players)
                        self.players.clear()
                else:
                    self.frames = 0
                    self.generation += 1
                    population = sorted(self.dead, key=lambda player: player.brain.fitness, reverse=True)
                    population = population[:250]
                    
                    print("MAX FIT: ", population[0].brain.fitness, "MIN FIT: ", population[-1].brain.fitness)
                    #assert(population[0].brain.fitness > population[-1].brain.fitness)
                    mutation_rate = 1/(population[0].brain.fitness)*9
                    
                    print("MUTATION: ", mutation_rate)
                    pool = []
                    r = 0.0  
                    for i in range(len(population)):
                        r += population[i].brain.fitness

                    for i in range(len(population)):
                        pick = random.uniform(0, r)
                        current = 0.0
                        for i in range(len(population)):
                            current += population[i].brain.fitness     
                            if current > pick:
                                pool.append(population[i])
                                break
                                        
                    print(len(pool))
                    children = []
                    for i in range(int(len(population))):
                        children.append(self.makeNewPlayer(population[i].brain))
                
                    for i in range(len(population)):
                        i1 = gnn.chooseAgent(pool)
                        i2 = gnn.chooseAgent(pool)
                        child = gnn.crossOver2(pool, i1, i2)
                        child = gnn.mutate(child, mutation_rate)
                        children.append(self.makeNewPlayer(child))
                    
                    
                    self.best = children[0]
                    self.dead.clear()
                    self.players = children
                    

                self.count.setText(str(self.frames))
                self.gens.setText(str(self.generation))
                self.update()

        def makeNewPlayer(self, brain):
            points, direction = sk.Snek.generate_randSnekPos(0,0,self.BLOCKS,self.BLOCKS)
            snake = sk.Snek(points, direction, 1)
            apple = sk.Apple(6, 6, 1)
            apple.updatePos(snake.body, self.BLOCKS, self.BLOCKS)
            player = sk.Player(self.bounds, snake, apple, brain)
            return player
        

        def keyPressEvent(self, event):
            pass
            # key = event.key()
            # ##FOR HUMAN INPUT ONLY##
            # time.sleep(.1)
            # #####################
            # for player in self.players:
            #     snake = player.snake
            #     direction = snake.headir
            #     if key == Qt.Key_Left:
            #         if direction != Direction.RIGHT:
            #                 snake.updateHeadDirection(Direction.LEFT)
            #     elif key == Qt.Key_Right:
            #         if direction != Direction.LEFT:
            #             snake.updateHeadDirection(Direction.RIGHT)
            #     elif key == Qt.Key_Up:
            #         if direction != Direction.DOWN:
            #             snake.updateHeadDirection(Direction.UP)
            #     elif key == Qt.Key_Down:
            #         if direction != Direction.UP:
            #             snake.updateHeadDirection(Direction.DOWN)

        def move(self, player):
            snake = player.snake
            direction = snake.headir
            if direction.value == Direction.UP.value:
                snake.hx, snake.hy = snake.hx, snake.hy - 1
                if snake.hy < 0:
                    self.end_game(player)
            if direction.value == Direction.DOWN.value:
                snake.hx, snake.hy = snake.hx, snake.hy + 1
                if snake.hy == self.BLOCKS:
                    self.end_game(player)
            if direction.value == Direction.LEFT.value:
                snake.hx, snake.hy = snake.hx - 1, snake.hy
                if snake.hx < 0:
                    self.end_game(player)
            if direction.value == Direction.RIGHT.value:
                snake.hx, snake.hy = snake.hx + 1, snake.hy
                if snake.hx == self.BLOCKS:
                    self.end_game(player)
            
            head = (snake.hx, snake.hy)
            snake.push(head)
            snake.dirpush(direction)

            if not snake.growing:
                snake.pop()
                snake.dirpop()
            else:
                snake.growing = False
            
            snake.updateVision()
            snake.updateBody()


        def apple_ate(self, player):
            head = player.snake.body[0]
            apple = player.apple
            applepos = apple.pos
            if applepos == head:
                player.removeObject(apple)
                apple.updatePos(player.snake.body, self.BLOCKS, self.BLOCKS)
                player.addObject(apple)
                player.snake.growing = True
                player.num_apples += 1
        
        def is_dead(self, player):
            body = player.snake.body 
            for i in range(1, len(body)):
                if body[0] == body[i]:
                    self.end_game(player)
        
        def end_game(self, player):
            self.players.remove(player)
            fitness = player.calculateFitness(self.frames, player.num_apples)
            player.setFitness(fitness)
            self.dead.append(player)
            
        
class GameBounds(obj.Object):
    def __init__(self, rect):
        self.rect = rect

    def name(self):
        return "walls"
    
    def getPoints(self):
        return self.rect.lines

class Direction(enum.Enum):
    UP =    [1.0, 0.0, 0.0, 0.0]
    DOWN =  [0.0, 1.0, 0.0, 0.0]
    LEFT =  [0.0, 0.0, 1.0, 0.0]
    RIGHT = [0.0, 0.0, 0.0, 1.0]           

    


if __name__ == '__main__':
    Game().start()
