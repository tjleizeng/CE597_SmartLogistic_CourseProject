import numpy as np
import torch

from optimization import route_optimization
from point import Point
from qlearner import QLearner
from vehicle import Vehicle
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")


class Environment(object):
    def __init__(self, size=3, request_type=[[(0, 0), (2, 2)], [(1, 0), (1, 2)], [(2, 0), (0, 2)], [(0, 1), (2, 1)]],
                 generate_rate=[0.1, 0.05, 0.1, 0.05], initial_location=[(0, 0), (1, 1)], vehicle_capacity=[4, 4],
                 max_waiting_time=2, precision=0, horizon=2, T=1000, mode='optimization', seed=80):
        self.size = size
        self.precision = precision
        self.T = T

        self.request_type = [(Point(request[0][0], request[0][1]), Point(request[1][0], request[1][1])) for request in
                             request_type]

        self.generate_rate = np.array(generate_rate)
        self.max_waiting_time = max_waiting_time
        self.max_travel_time = 8

        self.initial_location = initial_location

        self.vehicles = [Vehicle(loc[0], loc[1]) for loc in initial_location]

        self.current_request = {}  # Change this to dict and queue
        for i in range(len(self.request_type)):
            self.current_request[i] = PriorityQueue()

        self.current_on_board = {}
        for i in range(len(self.request_type)):
            self.current_on_board[i] = PriorityQueue()

        self.pickup_ind = {}
        self.drop_off_ind = {}
        for i in range(len(self.request_type)):
            self.pickup_ind[(self.request_type[i][0].x, self.request_type[i][0].y)] = i
            self.drop_off_ind[(self.request_type[i][1].x, self.request_type[i][1].y)] = i

        # Use queue to generate the data
        # self.current_request_latest_pickup = []
        # self.current_request_latest_delivery = []

        self.current_prediction = []
        self.current_prediction_earliest_pickup = []
        self.current_prediction_latest_pickup = []
        self.current_prediction_latest_delivery = []

        self.future_request = []
        self.future_prediction = []  # For solving optimization problem
        self.train_request = []  # For training neural network

        self.actions = [0, 0, 0]
        self.mode = mode
        self.incomes = []
        self.wages = []
        self.t = 0
        self.steps_done = 0
        self.horizon = horizon

        self.seed = seed

        self.qlearner = QLearner(18 + 3 * 4 + 8, 5 * 5 * 4)

        # generate data for running the experiment
        self.generate_request()
        self.generate_predicted_requests()
        self.generate_training_requests()

    def generate_request(self):
        np.random.seed(self.seed)
        for t in range(self.T):
            generate = np.random.random(len(self.generate_rate)) < self.generate_rate
            generated_request = []
            for i in range(len(self.generate_rate)):
                if generate[i]:
                    generated_request.append(i)
            self.future_request.append(generated_request)

    # Use this to feed the train data or the future prediction
    def generate_predicted_requests(self):
        for t in range(self.T):
            generated_request = []
            for i in range(len(self.generate_rate)):
                if i in self.future_request[t]:
                    if np.random.random() >= self.precision:
                        generated_request.append(i)
                elif np.random.random() < self.precision:
                    generated_request.append(i)
            self.future_prediction.append(generated_request)

    def generate_training_requests(self):
        self.train_request = []
        for t in range(self.T):
            generate_rate = self.generate_rate * (1 + np.array(
                [1 if i > 0.5 else -1 for i in np.random.random(len(self.generate_rate))]) * self.precision)
            generate = np.random.random(len(self.generate_rate)) < generate_rate
            generated_request = []
            for i in range(len(generate_rate)):
                if generate[i]:
                    generated_request.append(i)
            self.train_request.append(generated_request)

    def step_optimization(self):
        # post decision stage of t
        # moving vehicles, pickup and dropoff requests
        # translate the vehicle command into up, down, left, and right
        # optimization, find the solution that intersect with the current vehicle location
        # RL, directly output it

        # old_reward = np.sum([vehicle.income for vehicle in self.vehicles]) - np.sum([vehicle.get_plan_cost() for vehicle in self.vehicles])
        # update this to consider current request
        k = 0
        for vehicle in self.vehicles:
            vehicle.move(vehicle.get_action())
            for plan in vehicle.plan:
                if vehicle.loc == plan[0]:  # execute the plan
                    if plan[1] > 0:
                        if self.current_request[self.pickup_ind[(plan[0].x, plan[0].y)]].qsize() > 0:
                            vehicle.load += plan[1]
                            request = self.current_request[self.pickup_ind[(plan[0].x, plan[0].y)]].get()
                            # add_on_board vehicle
                            self.current_on_board[self.pickup_ind[(plan[0].x, plan[0].y)]].put((request[1], k))
                            vehicle.income += 10 / 2
                            # remove this plan
                            vehicle.plan.remove(plan)
                        break
                    else:
                        if self.current_on_board[self.drop_off_ind[(plan[0].x, plan[0].y)]].qsize() > 0:
                            if k == self.current_on_board[self.drop_off_ind[(plan[0].x, plan[0].y)]].queue[0][1]:
                                vehicle.load += plan[1]
                                self.current_on_board[self.drop_off_ind[(plan[0].x, plan[0].y)]].get()
                                # remove the plan
                                vehicle.income += 10 / 2
                                vehicle.plan.remove(plan)
                                break
            k += 1
        # collect rewards
        # new_reward = sum([vehicle.income for vehicle in self.vehicles]) - sum([vehicle.get_plan_cost() for vehicle in self.vehicles])

        for i in range(4):
            while (self.current_request[i].qsize() > 0):
                if (self.t > self.current_request[i].queue[0][0]):
                    self.current_request[i].get()
                    continue
                break

        # pre decision stage of t+1
        # generate new request
        if self.t + 1 < self.T:
            for new_request in self.future_request[self.t + 1]:
                self.current_request[new_request].put(
                    (self.t + self.max_waiting_time, self.t + self.max_waiting_time + self.max_travel_time))
        # print(self.future_request[self.t + 1])
        # generate new prediction
        self.current_prediction = []
        self.current_prediction_earliest_pickup = []
        self.current_prediction_latest_pickup = []
        self.current_prediction_latest_delivery = []
        for h in range(1, self.horizon):
            if self.t + h < self.T:
                self.current_prediction += self.future_prediction[self.t + h]
                self.current_prediction_earliest_pickup += [self.t + h] * len(self.future_prediction[self.t + h])
                self.current_prediction_latest_pickup += [self.t + h + self.max_waiting_time] * len(
                    self.future_prediction[self.t + h])
                self.current_prediction_latest_delivery += [
                                                               self.t + h + self.max_waiting_time + self.max_travel_time] * len(
                    self.future_prediction[self.t + h])
        # optimization_input
        # num_request, num_to_serve, num_veh, incidence_mat,
        #                        earliest_pickup_time, latest_delivery_time,
        #                        latest_to_serve_time,
        #                        initial_loads,
        #                        load_change,
        #                        travel_cost_mat
        point_list = []
        num_request = np.sum([self.current_request[i].qsize() for i in range(4)]) + len(self.current_prediction)
        num_to_serve = np.sum([self.current_on_board[i].qsize() for i in range(4)])
        num_veh = 2
        incidence_mat = []
        for i in range(4):
            for request in self.current_on_board[i].queue:
                if request[1] == 0:
                    incidence_mat.append([1, 0])
                else:
                    incidence_mat.append([0, 1])
        earliest_pickup_time = [self.t for i in range(4) for request in
                                self.current_request[i].queue] + self.current_prediction_earliest_pickup + \
                               [self.t + [4, 2, 4, 2][i] for i in range(4) for request in
                                self.current_request[i].queue] + [
                                   self.current_prediction_earliest_pickup[i] + [4, 2, 4, 2][
                                       self.current_prediction[i]] for i in range(len(self.current_prediction))]
        latest_delivery_time = [request[0] for i in range(4) for request in
                                self.current_request[i].queue] + self.current_prediction_latest_pickup + [request[1]
                                                                                                          for i in
                                                                                                          range(4)
                                                                                                          for
                                                                                                          request in
                                                                                                          self.current_request[
                                                                                                              i].queue] + self.current_prediction_latest_delivery
        latest_to_serve_time = [request[0] for i in range(4) for request in self.current_on_board[i].queue]
        point_list += [self.request_type[i][0] for i in range(4) for request in self.current_request[i].queue] + [
            self.request_type[i][0] for i in self.current_prediction]
        # print([(point.x, point.y) for point in point_list])
        point_list += [self.request_type[i][1] for i in range(4) for request in self.current_request[i].queue] + [
            self.request_type[i][1] for i in self.current_prediction]
        # print([(point.x, point.y) for point in point_list])
        point_list += [self.request_type[i][1] for i in range(4) for request in self.current_on_board[i].queue]
        point_list += [vehicle.loc for vehicle in self.vehicles]
        initial_loads = [veh.load for veh in self.vehicles]
        load_change = [1] * num_request + [-1] * num_request + [-1] * num_to_serve
        travel_cost_mat = [[point_1 - point_2 for point_1 in point_list] for point_2 in point_list]

        # process_plan(simulation_output)
        print("RUN OPT")
        res = route_optimization(num_request, num_to_serve, num_veh, incidence_mat,
                                 earliest_pickup_time, latest_delivery_time,
                                 latest_to_serve_time,
                                 initial_loads,
                                 load_change,
                                 travel_cost_mat)

        print(num_request)
        print(res)
        print([vehicle.load for vehicle in self.vehicles])
        # print([(point_list[i].x, point_list[i].y) for i in res[1]])
        # print(load_change)
        # print([vehicle.loc.x for vehicle in self.vehicles])
        # print([vehicle.loc.y for vehicle in self.vehicles])

        for i in range(len(self.vehicles)):
            veh_res = res[i]
            plan = []
            for j in veh_res:
                destination = point_list[j]
                load = load_change[j]
                plan.append((destination, load))
            self.vehicles[i].plan = plan

        self.incomes.append(np.sum([vehicle.income for vehicle in self.vehicles]))
        self.wages.append(np.sum([vehicle.wage for vehicle in self.vehicles]))
        self.t += 1

    def train(self):
        # post decision stage of t
        # moving vehicles, pickup and dropoff requests
        # translate the vehicle command into up, down, left, and right
        # optimization, find the solution that intersect with the current vehicle location
        # RL, directly output it
        # update this to consider current request
        x = [0]
        y = [0]
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.draw()
        for episode in range(300):
            # self.generate_training_requests()
            self.reset()
            self.generate_training_requests()
            actions = [0, 0, 0]
            state = [0] * 18

            state[self.vehicles[0].loc.x * self.size + self.vehicles[0].loc.y] = 1
            state[self.vehicles[1].loc.x * self.size + self.vehicles[1].loc.y + 9] = 1

            # add a to serve state here
            to_serve_state = [0] * 8
            for i in range(4):
                for request in self.current_on_board[i].queue:
                    if request[1] == 0:
                        to_serve_state[i] += 1
                    else:
                        to_serve_state[4 + i] += 1

            state += to_serve_state
            # to_serve_state = [0] * 8 * 8
            # for i in range(4):
            #     for request in self.current_on_board[i].queue:
            #         if request[1] == 0:
            #             to_serve_state[i + max(0, 8 * (request[0] - t))] += 1
            #         else:
            #             to_serve_state[4 + i + max(0, 8 * (request[0] - t))] += 1
            # state += to_serve_state

            request_state = [0] * (len(self.generate_rate) * self.max_waiting_time)
            for key, value in self.current_request.items():
                for (et, lt) in value.queue:
                    request_state[key + len(self.request_type) * (lt - t)] += 1
            state += request_state
            old_state = state
            old_action = None
            self.steps_done += 1
            old_reward = new_reward = 0
            for t in range(self.T):
                state = [0] * 18

                state[self.vehicles[0].loc.x * self.size + self.vehicles[0].loc.y] = 1
                state[self.vehicles[1].loc.x * self.size + self.vehicles[1].loc.y + 9] = 1

                to_serve_state = [0] * 8
                for i in range(4):
                    for request in self.current_on_board[i].queue:
                        if request[1] == 0:
                            to_serve_state[i] += 1
                        else:
                            to_serve_state[4 + i] += 1

                state += to_serve_state

                request_state = [0] * (len(self.generate_rate) * (1 + self.max_waiting_time))
                for key, value in self.current_request.items():
                    for (et, lt) in value.queue:
                        request_state[key + len(self.request_type) * (et - t)] += 1
                state += request_state

                # generate mask
                mask = self.generate_mask()
                action = self.qlearner.select_action(torch.FloatTensor(state), self.steps_done, mask)
                reward = new_reward - old_reward
                if (old_action is not None):
                    self.qlearner.memory.push(old_state, old_action, state, reward, mask)
                old_state = state
                old_action = action
                actions[0], actions[1], actions[2] = (action % 25) // 5, (action % 25) % 5, (action // 25)
                self.qlearner.optimize_mode()

                # distance to requests
                dist_to_requests = 0
                for i in range(4):
                    for request in self.current_on_board[i].queue:
                        dist_to_requests += self.vehicles[request[1]].loc- self.request_type[i][1]
                old_reward = np.sum([vehicle.income for vehicle in self.vehicles]) - np.sum(
                    [vehicle.wage for vehicle in self.vehicles]) - dist_to_requests
                k = 0
                for vehicle in self.vehicles:
                    vehicle.move(vehicle.get_action2(actions[k]))
                    if actions[2] in [1,3] and k == 0:
                        vehicle.willing_to_pickup = True
                    else:
                        vehicle.willing_to_pickup = False
                    if actions[2] in [2,3] and k == 1:
                        vehicle.willing_to_pickup = True
                    else:
                        vehicle.willing_to_pickup = False
                    if (vehicle.loc.x, vehicle.loc.y) in self.pickup_ind.keys() and vehicle.willing_to_pickup:
                        while self.current_request[self.pickup_ind[(vehicle.loc.x, vehicle.loc.y)]].qsize() > 0:
                            if (vehicle.load < vehicle.capacity):
                                vehicle.load += 1
                                request = self.current_request[self.pickup_ind[(vehicle.loc.x, vehicle.loc.y)]].get()
                                # add_on_board vehicle
                                self.current_on_board[self.pickup_ind[(vehicle.loc.x, vehicle.loc.y)]].put(
                                    (request[1], k))
                                vehicle.income += 10
                                continue
                            break

                    elif ((vehicle.loc.x, vehicle.loc.y) in self.drop_off_ind.keys()):
                        while self.current_on_board[self.drop_off_ind[(vehicle.loc.x, vehicle.loc.y)]].qsize() > 0:
                            if k == self.current_on_board[self.drop_off_ind[(vehicle.loc.x, vehicle.loc.y)]].queue[0][1]:
                                if (vehicle.load > 0):
                                    vehicle.load -= 1
                                    self.current_on_board[self.drop_off_ind[(vehicle.loc.x, vehicle.loc.y)]].get()
                                    continue
                            break

                    k += 1
                # collect rewards
                dist_to_requests = 0
                for i in range(4):
                    for request in self.current_on_board[i].queue:
                        dist_to_requests += self.vehicles[request[1]].loc - self.request_type[i][1]

                new_reward = sum([vehicle.income for vehicle in self.vehicles]) - sum(
                    [vehicle.wage for vehicle in self.vehicles]) - dist_to_requests

                for i in range(4):
                    while (self.current_request[i].qsize() > 0):
                        if (t > self.current_request[i].queue[0][0]):
                            self.current_request[i].get()
                            continue
                        break

                # for i in range(4):
                #     while (self.current_on_board[i].qsize() > 0):
                #         if (t + (self.vehicles[self.current_on_board[i].queue[0][1]].loc-self.request_type[i][1]) > self.current_on_board[i].queue[0][0]):
                #             request = self.current_on_board[i].get()
                #             self.vehicles[request[1]].load -= 1
                #             self.vehicles[request[1]].income -= 10 # Fail to serve the passenger
                #             continue
                #         break

                # pre decision stage of t+1
                # generate new request
                if t + 1 < self.T:
                    for new_request in self.train_request[t + 1]:
                        self.current_request[new_request].put(
                            (t + self.max_waiting_time, t + self.max_waiting_time + self.max_travel_time))

                # state: vehicle location, to_serve_request, request num * maximum waiting time *4, candidate schedules
                # self.incomes.append(np.sum([vehicle.income for vehicle in self.vehicles]))
                # self.wages.append(np.sum([vehicle.wage for vehicle in self.vehicles]))
            print(np.sum([vehicle.income for vehicle in self.vehicles]) - \
                  np.sum([vehicle.wage for vehicle in self.vehicles]))
            if episode % 5 == 0:
                self.qlearner.update_targets()
                x.append(episode + 1)
                y.append(np.sum([vehicle.income for vehicle in self.vehicles]) - \
                         np.sum([vehicle.wage for vehicle in self.vehicles]))
                plt.plot(x, y)
                # recompute the ax.dataLim
                plt.xlim(0, episode + 1)
                plt.ylim(min(y), max(y))
                plt.draw()
                plt.pause(0.02)
        plt.ioff()
        plt.show()
        self.qlearner.save_models("model" + str(int(self.precision * 10)))

    def reset(self):
        self.initial_location = self.initial_location
        self.vehicles = [Vehicle(loc[0], loc[1]) for loc in self.initial_location]
        self.current_request = {}  # Change this to dict and queue
        for i in range(len(self.request_type)):
            self.current_request[i] = PriorityQueue()
        self.current_on_board = {}
        for i in range(len(self.request_type)):
            self.current_on_board[i] = PriorityQueue()
        self.incomes = []
        self.wages = []
        self.actions = [0, 0, 0]
        self.t = 0

    def generate_mask(self):
        valid_actions = [[1]*5, [1]*5, [0]*5, [0]*5]
        for i in range(2):
            if self.vehicles[i].loc.x - 1 < 0:
                valid_actions[i][1] = 0
            if self.vehicles[i].loc.x + 1 == self.size:
                valid_actions[i][2] = 0
            if self.vehicles[i].loc.y - 1 < 0:
                valid_actions[i][3] = 0
            if self.vehicles[i].loc.y + 1 == self.size:
                valid_actions[i][4] = 0
            if (self.vehicles[i].loc.x, self.vehicles[i].loc.y) in self.pickup_ind.keys() and (self.vehicles[i].load < self.vehicles[i].capacity):
                valid_actions[2+i][0] = 1
            if ((self.vehicles[i].loc.x - 1), self.vehicles[i].loc.y) in self.pickup_ind.keys() and (self.vehicles[i].load < self.vehicles[i].capacity):
                valid_actions[2 + i][1] = 1
            if ((self.vehicles[i].loc.x + 1), self.vehicles[i].loc.y) in self.pickup_ind.keys() and (self.vehicles[i].load < self.vehicles[i].capacity):
                valid_actions[2 + i][2] = 1
            if (self.vehicles[i].loc.x, (self.vehicles[i].loc.y-1)) in self.pickup_ind.keys() and (self.vehicles[i].load < self.vehicles[i].capacity):
                valid_actions[2 + i][3] = 1
            if (self.vehicles[i].loc.x, (self.vehicles[i].loc.y+1)) in self.pickup_ind.keys() and (self.vehicles[i].load < self.vehicles[i].capacity):
                valid_actions[2 + i][4] = 1
        mask = [i*j for i in valid_actions[0] for j in valid_actions[1]] + \
               [valid_actions[0][i]*valid_actions[1][j]*valid_actions[2][i] for i in range(5) for j in range(5)] +\
               [valid_actions[0][i]*valid_actions[1][j]*valid_actions[3][j] for i in range(5) for j in range(5)] + \
               [valid_actions[0][i] * valid_actions[1][j]*valid_actions[2][i] * valid_actions[3][j] for i in range(5) for j in range(5)]
        return mask

    def step_test(self):
        # state: vehicle location, capacity, request num * maximum waiting time *4, candidate schedules
        state = [0]*18

        state[self.vehicles[0].loc.x * self.size + self.vehicles[0].loc.y] = 1
        state[self.vehicles[1].loc.x * self.size + self.vehicles[1].loc.y + 9] = 1

        to_serve_state = [0] * 8
        for i in range(4):
            for request in self.current_on_board[i].queue:
                if request[1] == 0:
                    to_serve_state[i] += 1
                else:
                    to_serve_state[4 + i] += 1

        state += to_serve_state

        request_state = [0] * (len(self.generate_rate) * (1 + self.max_waiting_time))
        for key, value in self.current_request.items():
            for (et, lt) in value.queue:
                request_state[key + len(self.request_type) * (et - self.t)] += 1
        state += request_state

        # generate mask
        mask = self.generate_mask()
        action = self.qlearner.select_action(torch.FloatTensor(state), 400, mask)
        self.actions[0], self.actions[1], self.actions[2] = (action % 25) // 5, (action % 25) % 5, (action // 25)

        # print(state)

        k = 0
        for vehicle in self.vehicles:
            vehicle.move(vehicle.get_action2(self.actions[k]))
            if self.actions[2] in [1, 3] and k == 0:
                vehicle.willing_to_pickup = True
            else:
                vehicle.willing_to_pickup = False
            if self.actions[2] in [2, 3] and k == 1:
                vehicle.willing_to_pickup = True
            else:
                vehicle.willing_to_pickup = False
            if (vehicle.loc.x, vehicle.loc.y) in self.pickup_ind.keys() and vehicle.willing_to_pickup:
                while self.current_request[self.pickup_ind[(vehicle.loc.x, vehicle.loc.y)]].qsize() > 0:
                    if (vehicle.load < vehicle.capacity):
                        vehicle.load += 1
                        request = self.current_request[self.pickup_ind[(vehicle.loc.x, vehicle.loc.y)]].get()
                        # add_on_board vehicle
                        self.current_on_board[self.pickup_ind[(vehicle.loc.x, vehicle.loc.y)]].put(
                            (request[1], k))
                        vehicle.income += 10
                        continue
                    break

            elif ((vehicle.loc.x, vehicle.loc.y) in self.drop_off_ind.keys()):
                while self.current_on_board[self.drop_off_ind[(vehicle.loc.x, vehicle.loc.y)]].qsize() > 0:
                    if k == self.current_on_board[self.drop_off_ind[(vehicle.loc.x, vehicle.loc.y)]].queue[0][1]:
                        if (vehicle.load > 0):
                            vehicle.load -= 1
                            self.current_on_board[self.drop_off_ind[(vehicle.loc.x, vehicle.loc.y)]].get()
                            continue
                    break

            k += 1
        # collect rewards
        # new_reward = sum([vehicle.income for vehicle in self.vehicles]) - sum([vehicle.get_plan_cost() for vehicle in self.vehicles])
        for i in range(4):
            while (self.current_request[i].qsize() > 0):
                if (self.t > self.current_request[i].queue[0][0]):
                    self.current_request[i].get()
                    continue
                break
        # pre decision stage of t+1
        # generate new request
        if self.t + 1 < self.T:
            for new_request in self.future_request[self.t + 1]:
                self.current_request[new_request].put(
                    (self.t + self.max_waiting_time, self.t + self.max_waiting_time + self.max_travel_time))
        # print(self.future_request[self.t + 1])
        # save this idea for later
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             for l in range(3):
        # generate candidate plans of serving existing requests
        # four for loops, schedule is decorated as 16 elements vector, we shaded out the invalid options
        # for i in range(len(self.request_type))
        # ...
        # is_valid
        # updated the final schedule
        # vehicle.update_schedule
        self.t += 1
        self.incomes.append(np.sum([vehicle.income for vehicle in self.vehicles]))
        self.wages.append(np.sum([vehicle.wage for vehicle in self.vehicles]))

    def save_res(self, name):
        print("FINISH")
        import pandas as pd
        result = pd.DataFrame({"income": self.incomes, "wage": self.wages})
        result.to_csv(name)
