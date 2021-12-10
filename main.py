# Main class for CE597: smart logistic course project
# Author: Zengxiang Lei

# Press the green button in the gutter to run the script.
from env import Environment

if __name__ == '__main__':
    # env = Environment(precision=0, horizon=0, generate_rate=[0.06, 0.03, 0.06, 0.03])
    # for t in range(env.T):
    #     env.step_optimization()
    # env.save_res("opt_baseline_under.csv")
    # for precision in [0,0.05,0.1, 0.15]:
    #     env = Environment(precision = precision, horizon = 3, generate_rate=[0.14, 0.07, 0.14, 0.07])
    #     for t in range(env.T):
    #         env.step_optimization()
    #     env.save_res("opt_"+str(int(precision*100))+"_"+str(3)+"_above.csv")
    for precision in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        env = Environment(precision=precision, generate_rate=[0.1,0.05,0.1,0.05])
        #env.train()
        env.reset()
        env.qlearner.load_models("model"+str(int(precision * 10)))
        for t in range(env.T):
            env.step_test()
        env.save_res("result/qlearning_" + str(int(precision * 10))+ "_above.csv")

