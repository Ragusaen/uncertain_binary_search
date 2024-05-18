from probabilistic_bisection2 import ProbabilisticBisection2
from probability_table import entropy
from test_simulator import TestSimulator
from probabilistic_bisection import ProbabilisticBisection
import cProfile
import pstats

N = 6
c = 0.95

sim = TestSimulator(N, 1.0, 0.1, seed=0)
bis = ProbabilisticBisection2(N, discretisation_steps=10)
print('Actual:', sim.true_positive_rate, sim.true_negative_rate, sim.bad_point)

def run():
    steps = 0
    while True:
        steps += 1
        t = bis.best_test_point()
        r = sim.run_test(t)
        print('entropy before', bis.entropy())
        bis.report_test_result(t, r == 'Pass')
        print('entropy after', bis.entropy())

        print(t, r, [bis.prob_point_bad(i) for i in range(N)])
        print('\t', bis.theta_given_evidence())
        if any(bis.prob_point_bad(i) >= c for i in range(N)):
            assert max(range(N), key=bis.prob_point_bad) == sim.bad_point
            break

    print('Total tests: ', steps)

run()