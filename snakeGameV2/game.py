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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

        self.initPlot()
    
    def initPlot(self):
            plt.style.use("fivethirtyeight")
            self.x_vals = []
            self.y_vals = []
            self.x_vals.append(self.board.generation)
            self.y_vals.append(self.board.players[0].brain.fitness)
            plt.plot(self.x_vals,self.y_vals)

            ani = FuncAnimation(plt.gcf(), self.animatePlot, interval='100')

            plt.tight_layout()
            plt.show()
    
    def animatePlot(self,i):
        gen = self.board.generation
        if gen > self.x_vals[-1]:
            self.x_vals.append(self.board.generation)
            self.y_vals.append(self.board.players[0].brain.fitness)
            plt.cla()
            plt.plot(self.x_vals, self.y_vals, label='Max Fitness')
            plt.legend()
            plt.tight_layout()

    
class Board(QFrame):
        SPEED = 50

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
            apple = sk.Apple(1, 5, 1)
            brain = gnn.get_rand_Agent()
            player = sk.Player(self.bounds, snake, apple, brain)
            self.players.append(player)
            
        
        def initBoard2(self):
            self.popsize = 1000
            self.players = []
            self.best = []
            self.dead = []
            self.bounds = GameBounds(shape.Rectangle(0,0,self.BLOCKS,self.BLOCKS))
            for i in range(self.popsize):
                snake = sk.Snek.createRandomSnake(0,0,self.BLOCKS,self.BLOCKS,1)
                apple = sk.Apple.createRandomApple(snake.body, self.BLOCKS, self.BLOCKS)
                brain = gnn.get_rand_Agent()
                player = sk.Player(self.bounds, snake, apple, brain)
                self.players.append(player)
            self.best = self.players[0]
        
        def initText(self):
            self.frames = 1
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

            for player in self.players:
                applepos = player.apple.pos
                self.draw_square(painter, applepos[0]*self.square_length(), applepos[1]*self.square_length(), QColor(0xFF0000))

                snake = player.snake
                for i in range(len(snake.body)):
                    pos = snake.body[i]
                    x =  pos[0] * self.square_length()
                    y = pos[1] * self.square_length()
                    color = QColor(0x228B22)
                    if i == 0:
                        color = QColor(0x003200)
                    self.draw_square(painter, x, y,color)
                
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
                        self.updateAllPlayers()
                    else:
                        self.killAllPlayers()
                    self.frames += 1 
                else:
                    self.frames = 1
                    newpop = self.createNewGen()
                    self.dead.clear()
                    self.players = newpop

                self.count.setText(str(self.frames))
                self.gens.setText(str(self.generation))
                self.update()

        def createNewGen(self):
            pop = sorted(self.dead, key=lambda player: player.brain.fitness, reverse=True)
            print("MAX FIT: ", pop[0].brain.fitness, "MIN FIT: ", pop[-1].brain.fitness)
            print("MAX APPLE: ", pop[0].num_apples)
            fourth = int(math.floor(self.popsize/4))
            pop = pop[:self.popsize]
            pool = pop[:fourth]
            self.generation += 1
            mutrate = 0.05
            print("MUTATION RATE: ", mutrate)
            assert(len(pop) > 1)

            newpop = []
            for i in range(10):
                newpop.append(self.makeNewPlayer(pop[i].brain))
            assert(newpop[0].brain.fitness == pop[0].brain.fitness)
            for i in range(self.popsize):
                p1,p2 = self.getParents(pool)     
                childbrain = gnn.crossOver2(pool, p1, p2)
                childbrain = gnn.mutate(childbrain, mutrate)  
                newpop.append(self.makeNewPlayer(childbrain))

            return newpop

        def getParents(self,pop):
            p1 = gnn.getParent(pop)
            p2 = gnn.getParent(pop)
            while p1 == p2:
                p2 = gnn.getParent(pop)
            return p1,p2

        
        def updateAllPlayers(self):
            for player in self.players:
                inputs = player.look(self.BLOCKS, 0.0)
                output = player.move(inputs)
                player.updateDirection(output)
                self.move(player)
                self.apple_ate(player)
                self.is_dead(player)
        
        def makeNewPlayer(self, brain):
            snake = sk.Snek.createRandomSnake(0,0,self.BLOCKS,self.BLOCKS,1)
            apple = sk.Apple.createRandomApple(snake.body, self.BLOCKS, self.BLOCKS)
            player = sk.Player(self.bounds, snake, apple, brain)
            return player
        
        def killAllPlayers(self):
            for player in self.players:
                self.end_game(player)
        
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
            snake.updateBody()
            snake.updateVision()
            


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