import random
import time

from probability_table import entropy
from test_simulator import TestSimulator
from probabilistic_bisection import ProbabilisticBisection
import cProfile
import pstats


def run(sim, bis, c = 0.95):
    steps = 0
    print('Bad point', sim.bad_point)
    while True:
        steps += 1
        t = bis.best_test_point()
        r = sim.run_test(t)
        # print('entropy before', bis.entropy())
        bis.report_test_result(t, r == 'Pass')
        # print('entropy after', bis.entropy())

        print(t, r, [bis.prob_point_bad(i) for i in range(bis.num_points)])
        # print('\t', bis.theta_given_evidence())
        if any(bis.prob_point_bad(i) >= c for i in range(bis.num_points)):
            return max(range(bis.num_points), key=bis.prob_point_bad)

def random_test():
    for i in range(10000):
        N = random.randint(3, 20)
        sim = TestSimulator(N, 1.0, random.random() * 0.99 + 0.01)
        bis = ProbabilisticBisection2(N)
        print(N, sim.true_negative_rate)

        if sim.bad_point == run(sim, bis):
            print('right')
        else:
            print('wrong')

t = time.time()
run(TestSimulator(50, 1.0, 0.05, seed=0), ProbabilisticBisection(20))
print(time.time() - t)
