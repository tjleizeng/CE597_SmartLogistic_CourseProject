class Point(object):
    def __init__(self, x, y , size = 3):
        self.x = x
        self.y = y
        self.size = size

    def __add__(self, other):
        self.x = self.x + other.x
        self.y = self.y + other.y
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x >= self.size:
            self.x = 2
        if self.y >= self.size:
            self.y = 0
        return self

    def __sub__(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))
