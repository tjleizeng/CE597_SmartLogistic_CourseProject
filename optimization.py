import cplex
import numpy as np
def route_optimization (num_request, num_to_serve, num_veh, incidence_mat,
                       earliest_pickup_time, latest_delivery_time,
                       latest_to_serve_time,
                       initial_loads,
                       load_change,
                       travel_cost_mat):
    # print("OPT INPUT")
    # print(num_request)
    # print(num_to_serve)
    # print(num_veh)
    # print(incidence_mat)
    # print(earliest_pickup_time)
    # print(latest_delivery_time)
    # print(latest_to_serve_time)
    # print(initial_loads)
    # print(load_change)
    # print(travel_cost_mat)


    myProblem = cplex.Cplex()
    myProblem.set_log_stream(None)
    myProblem.set_error_stream(None)
    myProblem.set_warning_stream(None)
    myProblem.set_results_stream(None)
    myProblem.parameters.mip.tolerances.mipgap.set(1e-3)
    myProblem.variables.add(
        names = ["x"+str(k)+"_"+str(i)+"_"+str(j) for k in range(num_veh)\
                 for i in range(2*num_request+num_to_serve)\
                 for j in range(2*num_request+num_to_serve) if i != j])
    myProblem.variables.add(names = ["x" + str(i) + "_" + str(j) \
             for i in range(num_veh)\
             for j in range(num_request + num_to_serve  + num_veh)])

    myProblem.variables.add(names = ["T_"+str(k)+"_"+str(i) for k in range(num_veh)\
                                     for i in range(2*num_request+num_to_serve)])
    myProblem.variables.add(names = ["c_"+str(k)+"_"+str(i) for k in range(num_veh)\
                                     for i in range(2*num_request+num_to_serve)])

    for k in range(num_veh):
        for i in range(2*num_request+num_to_serve):
            for j in range(2*num_request+num_to_serve):
                if i!=j:
                    myProblem.variables.set_types("x"+str(k)+"_"+str(i)+"_"+str(j),\
                                                  myProblem.variables.type.binary)

    for i in range(num_veh):
        for j in range(num_request+num_to_serve + num_veh):
            myProblem.variables.set_types("x" + str(i) + "_" + str(j),\
                                          myProblem.variables.type.binary)

    for k in range(num_veh):
        for i in range(2*num_request+num_to_serve):
            myProblem.variables.set_lower_bounds("T_"+str(k)+"_"+str(i), (earliest_pickup_time+[0]*num_to_serve)[i])
            # myProblem.variables.set_upper_bounds("T_" + str(k) + "_" + str(i), (latest_delivery_time+latest_to_serve_time)[i])
            myProblem.variables.set_types("T_"+str(k)+"_"+str(i), myProblem.variables.type.continuous)
            myProblem.variables.set_lower_bounds("c_"+str(k)+"_"+str(i), 0.0)
            myProblem.variables.set_upper_bounds("c_"+str(k)+"_"+str(i), 4)
            myProblem.variables.set_types("c_"+str(k)+"_"+str(i), myProblem.variables.type.continuous)

    constraint_count = 0

    # all nodes are visited at most once
    for i in range(2*num_request+num_to_serve):
        index = ["x"+str(k)+"_"+str(j)+"_"+str(i) for k in range(num_veh)\
                 for j in range(2*num_request+num_to_serve) if i != j] + \
                ["x" + str(j) + "_" + str(i) \
                 for j in range(num_veh) if i < num_request+num_to_serve]
        myProblem.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = index, val = [1]*len(index))],
            rhs = [1],
            names = ["c"+str(constraint_count)],
            senses = ['L']
        )
        constraint_count +=1

    # pickup and dropoff is visited at the same time by the same veh
    for k in range(num_veh):
        for i in range(num_request):
            index1 = ["x" + str(k) + "_" + str(j) + "_" + str(i) \
                     for j in range(2 * num_request + num_to_serve) if i != j] + \
                     ["x" + str(k) + "_" + str(i)]
            index2 = ["x" + str(k) +"_" + str(j) + "_" + str(i+num_request) \
                     for j in range(2 * num_request + num_to_serve) if i+num_request != j]
            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=index1+index2, val=[1] * len(index1)+[-1]*len(index2))],
                rhs=[0],
                names=["c" + str(constraint_count)],
                senses=['E']
            )
            constraint_count += 1

    # to serve request is served by the assigned veh
    for k in range(num_veh):
        for i in range(num_to_serve):
            value = incidence_mat[i][k]
            if value:
                index = ["x" +str(k) + "_"+str(j) +"_" + str(2 * num_request+i) \
                          for j in range(2 * num_request + num_to_serve) if 2 * num_request+i != j] + \
                         ["x" + str(k) + "_" + str(num_request+i)]

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=index, val=[1]*len(index))],
                    rhs=[1],
                    names=["c" + str(constraint_count)],
                    senses=['E']
                )
                constraint_count += 1

    # flow conservation
    for k in range(num_veh):
        for i in range(num_request*2+num_to_serve):
            index1 = ["x" + str(k) + "_" + str(j) + "_" + str(i) \
                     for j in range(2 * num_request + num_to_serve) if i != j]
            if i < num_request:
                index1 += ["x" + str(k) + "_" + str(i)]
            elif i >= 2*num_request:
                index1 += ["x" + str(k) + "_" + str(i-num_request)]

            index2 = ["x" + str(k) + "_" + str(i) + "_" + str(j) \
                     for j in range(2 * num_request + num_to_serve) if i != j]

            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=index1+index2, val=[1] * len(index1) + [-1] * len(index2))],
                rhs=[0],
                names=["c" + str(constraint_count)],
                senses=['G']
            )
            constraint_count += 1

    for k in range(num_veh):
        index = ["x" +str(k) + "_"+str(i)  \
                          for i in range(num_request + num_to_serve + num_veh)]

        myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=index, val=[1]*len(index))],
                    rhs=[1],
                    names=["c" + str(constraint_count)],
                    senses=['E']
                )
        constraint_count += 1

    # shortest time constraint
    for k in range(num_veh):
        for i in range(2 * num_request + num_to_serve):
            for j in range(2 * num_request + num_to_serve):
                if i != j:
                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=["x"+str(k)+"_"+str(i)+"_"+str(j)]+\
                                                   ["T_"+str(k)+"_"+str(i)]+\
                                                   ["T_"+str(k)+"_"+str(j)], val=[1000,1,-1])],
                        rhs=[-travel_cost_mat[i][j]+1000],
                        names=["c" + str(constraint_count)],
                        senses=['L']
                    )
                    constraint_count += 1
        for i in range(num_request):
            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i)] + \
                                               ["T_" + str(k) + "_" + str(i)], val=[1000, -1])],
                rhs=[-travel_cost_mat[k + num_request * 2 + num_to_serve][i] + 1000],
                names=["c" + str(constraint_count)],
                senses=['L']
            )
            constraint_count += 1
        for i in range(num_to_serve):
            if incidence_mat[i][k]:
                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i + num_request)] + \
                                                   ["T_" + str(k) + "_" + str(i + num_request * 2)],
                                               val=[1000, -1])],
                    rhs=[-travel_cost_mat[k + num_request * 2 + num_to_serve][i + num_request * 2] + 1000],
                    names=["c" + str(constraint_count)],
                    senses=['L']
                )
                constraint_count += 1

    # max delay constraint,  (latest_delivery_time+latest_to_serve_time)[i]
    for k in range(num_veh):
        for i in range(2 * num_request + num_to_serve):
            for j in range(2 * num_request + num_to_serve):
                if i != j:
                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i) + "_" + str(j)] + \
                                                       ["T_" + str(k) + "_" + str(j)],
                                                   val=[1000, 1])],
                        rhs=[1000+(latest_delivery_time+latest_to_serve_time)[i]],
                        names=["c" + str(constraint_count)],
                        senses=['L']
                    )
                    constraint_count += 1
        for i in range(num_request):
            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i)] + \
                                               ["T_" + str(k) + "_" + str(i)], val=[1000, 1])],
                rhs=[(latest_delivery_time+latest_to_serve_time)[i] + 1000],
                names=["c" + str(constraint_count)],
                senses=['L']
            )
            constraint_count += 1
        for i in range(num_to_serve):
            if incidence_mat[i][k]:
                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i+num_request)] + \
                                                   ["T_" + str(k) + "_" + str(i+num_request*2)], val=[1000, 1])],
                    rhs=[(latest_delivery_time+latest_to_serve_time)[num_request*2+i] + 1000],
                    names=["c" + str(constraint_count)],
                    senses=['L']
                )
                constraint_count += 1




    for k in range(num_veh):
        for i in range(2 * num_request + num_to_serve):
            for j in range(2 * num_request + num_to_serve):
                if i != j:
                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i) + "_" + str(j)] + \
                                                       ["c_" + str(k) + "_" + str(i)] + \
                                                       ["c_" + str(k) + "_" + str(j)], val=[1000, 1, -1])],
                        rhs=[-load_change[j] + 1000],
                        names=["c" + str(constraint_count)],
                        senses=['L']
                    )
                    constraint_count += 1
                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i) + "_" + str(j)] + \
                                                       ["c_" + str(k) + "_" + str(i)] + \
                                                       ["c_" + str(k) + "_" + str(j)], val=[-1000, 1, -1])],
                        rhs=[-load_change[j] - 1000],
                        names=["c" + str(constraint_count)],
                        senses=['G']
                    )
                    constraint_count += 1

    for k in range(num_veh):
        for i in range(num_request):
            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i)] + \
                                               ["c_" + str(k) + "_" + str(i)], val=[1000, -1])],
                rhs=[-initial_loads[k]-load_change[i] + 1000],
                names=["c" + str(constraint_count)],
                senses=['L']
            )
            constraint_count += 1
            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i)] + \
                                               ["c_" + str(k) + "_" + str(i)], val=[-1000, -1])],
                rhs=[-initial_loads[k] - load_change[i] - 1000],
                names=["c" + str(constraint_count)],
                senses=['G']
            )
            constraint_count += 1
        for i in range(num_to_serve):
            if incidence_mat[i][k]:
                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i+num_request)] + \
                                                   ["c_" + str(k) + "_" + str(i+num_request*2)], val=[1000, -1])],
                    rhs=[-initial_loads[k] - load_change[i+num_request*2] + 1000],
                    names=["c" + str(constraint_count)],
                    senses=['L']
                )
                constraint_count += 1
                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=["x" + str(k) + "_" + str(i + num_request)] + \
                                                   ["c_" + str(k) + "_" + str(i + num_request * 2)], val=[-1000, -1])],
                    rhs=[-initial_loads[k] - load_change[i + num_request * 2] - 1000],
                    names=["c" + str(constraint_count)],
                    senses=['G']
                )
                constraint_count += 1

    index1 = ["x"+str(k)+"_"+str(i)+"_"+str(j) for k in range(num_veh) for i in range(num_request*2+num_to_serve) for j in range(num_request) if i!=j]+\
        ["x"+str(k)+"_"+str(i) for k in range(num_veh) for i in range(num_request)]

    index2 = ["x"+str(k)+"_"+str(i)+"_"+str(j) for k in range(num_veh) for i in range(num_request*2+num_to_serve) for j in range(num_request*2+num_to_serve) if i!=j]+\
        ["x"+str(k)+"_"+str(i) for k in range(num_veh) for i in range(num_request+num_to_serve)]

    for index in index2:
        if index in index1:
            myProblem.objective.set_linear(index, 4)
        else:
            myProblem.objective.set_linear(index, -1)

    myProblem.objective.set_sense(myProblem.objective.sense.maximize)
    #myProblem.parameters.timelimit = 60
    myProblem.solve()

    # return vehicle plan as a list
    res = []
    for k in range(num_veh):
        one_res = []
        start_pt = myProblem.solution.get_values(["x" + str(k) + "_" + str(j) \
           for j in range(num_request + num_to_serve)])
        #print(start_pt)
        if(1 in start_pt):
            start_index = np.where(np.abs(np.array(start_pt)-1)<0.1)[0][0]
            if(start_index>=num_request):
                start_index += num_request
            one_res.append(start_index)
            inter_pt = np.array([myProblem.solution.get_values(["x" + str(k) + "_" + str(i) + "_" + str(j)])[0]  if i != j else 0 \
                          for i in range(2 * num_request + num_to_serve) \
                          for j in range(2 * num_request + num_to_serve)]).reshape(2 * num_request + num_to_serve,
                                            2 * num_request + num_to_serve)
            # print(inter_pt)
            while(1 in inter_pt[one_res[-1],:]):
                one_res.append(np.where(inter_pt[one_res[-1],:] == 1)[0][0])
        res.append(one_res)
    return(res)


    # print(myProblem.solution.get_values(["x" + str(i) + "_" + str(j) \
    #      for i in range(num_veh)\
    #      for j in range(num_request + num_to_serve)]))
    #
    # print(np.array(myProblem.solution.get_values(["x" + str(0) + "_" + str(i) + "_" + str(j)  if\
    #                                               i != j else 0 \
    #                                               for i in range(2 * num_request + num_to_serve) \
    #                                               for j in range(2 * num_request + num_to_serve)])).reshape(2 * num_request + num_to_serve,
    #                                                                 2 * num_request + num_to_serve ))
    # print(np.array(myProblem.solution.get_values(["x" + str(1) + "_" + str(i) + "_" + str(j) if i!=j else 0 \
    #                                               for i in range(2 * num_request + num_to_serve) \
    #                                               for j in range(2 * num_request + num_to_serve)])).reshape(2 * num_request + num_to_serve,
    #                                                                 2 * num_request + num_to_serve ))
    # print(myProblem.solution.get_objective_value())
if __name__ == '__main__':
    #num_request, num_to_serve, num_veh, incidence_mat,
    #                   earliest_pickup_time, latest_delivery_time,
    #                   latest_to_serve_time,
     #                  initial_loads,
     #                  load_change,
     #                  travel_cost_mat
    print(route_optimization(2, 2, 2, [[1,0],[0,1]], [0,0,4,4], [5, 5, 8, 8], [4, 4], [1,1],
                       [1,1,-1,-1,-1,-1],
                       [[0,2,2,2,3,3,1,1],
                         [2,0,2,2,3,1,1,3],
                         [2,2,0,2,1,1,3,3],
                         [2,2,2,0,1,3,3,1],
                         [3,3,1,1,0,2,4,2],
                        [3,1,1,3,2,0,2,4],
                        [1,1,3,3,4,2,0,2],
                        [1,3,3,1,2,4,2,0]]))



