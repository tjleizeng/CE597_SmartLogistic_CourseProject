from point import Point


class Vehicle(object):
    def __init__(self, x, y, capacity=4):
        self.loc = Point(x, y)
        self.capacity = capacity
        self.load = 0
        self.plan = []  # plan is a tuple, location, load_change
        self.income = 0
        self.wage = 0
        self.willing_to_pickup = True

    def move(self, action):
        self.loc += action
        self.wage += abs(action.x) + abs(action.y)

    def update_plan(self, locations, load_changes):
        self.plan = []
        for i in range(len(locations)):
            self.plan.append((locations[i], load_changes[i]))

    def get_plan_cost(self):
        tmp_loc = self.loc
        cost = 0
        for location, _ in self.plan:
            cost += location - tmp_loc
            tmp_loc = location
        return cost

    def get_action(self):
        if self.plan:
            if self.plan[0][0] - self.loc > 0:
                if self.plan[0][0].x < self.loc.x:
                    return Point(-1, 0)
                elif self.plan[0][0].x > self.loc.x:
                    return Point(1, 0)
                elif self.plan[0][0].y < self.loc.y:
                    return Point(0, -1)
                else:
                    return Point(0, 1)
        return Point(0, 0)

    def get_action2(self, action):
        if action == 0:
            return Point(0,0)
        elif action == 1:
            return Point(-1,0)
        elif action == 2:
            return Point(1,0)
        elif action == 3:
            return Point(0,-1)
        elif action == 4:
            return Point(0,1)

