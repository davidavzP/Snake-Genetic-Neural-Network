class Rectangle():
    def __init__(self, x, y,width=1,height=1):
        self.startx = x
        self.starty = y
        self.w = width
        self.h = height
        self.lines = self.generatePoints()

    def generatePoints(self):
        top = (self.startx, self.starty, self.startx + self.w, self.starty) 
        left = (self.startx, self.starty, self.startx, self.starty + self.h) 
        right = (self.startx + self.w, self.starty, self.startx + self.w, self.starty + self.h)
        bottom = (self.startx, self.starty + self.h, self.startx + self.w, self.starty + self.h)
        return [top,left,right,bottom]
