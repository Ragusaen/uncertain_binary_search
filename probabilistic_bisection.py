from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, BeliefPropagation, DBNInference, ApproxInference
from pgmpy.inference.EliminationOrder import BaseEliminationOrder
from pgmpy.models import BayesianNetwork
import numpy as np


TabularCPD._truncate_strtable = lambda self, x: x

GOOD = 'Good'
BAD = 'Bad'
PASS = 'Pass'
FAIL = 'Fail'

def point_var_idx(i: int):
    # return i
    return f'X{i}'

def test_var_idx(i: int, j: int):
    # return  i + 1000 * (j + 1)
    return f'T{i}-{j}'

def error_var_idx():
    # return -1
    return 'E'

def theta_var_idx():
    # return -2
    return 'θ'


def discrete_factor_log2(self):
    with np.errstate(divide='ignore'):
        self.values = np.nan_to_num(np.log2(self.values))

DiscreteFactor.log2 = discrete_factor_log2


def entropy(p: DiscreteFactor, q: DiscreteFactor = DiscreteFactor([], [], [1.]), with_vars: set = None):
    """
    Computes H(X|Y) from P(X,Y) and P(Y)
    """
    p_g = p / q
    p_g.log2()
    j = p * p_g
    j.marginalize(set(p.variables) - (with_vars if with_vars else set()))
    return -j.values


class ProbabilisticBisection:
    def __init__(self, num_points: int, discretisation_steps: int = 20):
        self.theta_discretisation = [(j + 1) / discretisation_steps for j in range(discretisation_steps)]
        self.num_points = num_points
        self.tests = {i: {PASS: 0, FAIL: 0} for i in range(num_points)}

        self.G = BayesianNetwork()

        self.G.add_node(error_var_idx())
        self.G.add_cpds(
            TabularCPD(error_var_idx(),
               num_points,
               [[1 / num_points] for _ in range(num_points)],
               state_names={error_var_idx(): [j for j in range(num_points)]}
               )
        )

        self.G.add_node(theta_var_idx())
        self.G.add_cpds(
            TabularCPD(theta_var_idx(),
                   discretisation_steps,
                   [[1 / discretisation_steps] for _ in range(discretisation_steps)],
                   state_names={theta_var_idx(): self.theta_discretisation}
               )
        )

        for i in range(num_points):
            self.G.add_node(point_var_idx(i))
            self.G.add_edge(error_var_idx(), point_var_idx(i))
            self.G.add_cpds(
                TabularCPD(point_var_idx(i),
                    2,
                    [[int(o) if i < j else int(not o) for j in range(num_points)] for o in [True, False]],
                    evidence=[error_var_idx()],
                    evidence_card=[num_points],
                    state_names={point_var_idx(i): [GOOD, BAD], error_var_idx(): [j for j in range(num_points)]}
               )
            )

        self.G.check_model()

        self.test_nodes = [0 for _ in range(num_points)]

        self.inference = VariableElimination(self.G)

        self.query_cache = {}

    def query(self, variables: list):
        tupvar = tuple(variables)
        if tupvar in self.query_cache:
            return self.query_cache[tupvar]
        res = self.inference.query(
            variables,
            evidence=self.get_test_evidence(),
            # elimination_order=self.get_elimination_order(set(self.G.nodes) - set(variables) - set(self.get_test_evidence().keys())),
            show_progress=False
        )
        self.query_cache[tupvar] = res
        return res

    def add_test_node(self, point: int):
        self.query_cache.clear()
        test_index = self.test_nodes[point]
        self.test_nodes[point] += 1
        self.G.add_node(test_var_idx(point, test_index))
        self.G.add_edge(point_var_idx(point), test_var_idx(point, test_index))
        self.G.add_edge(theta_var_idx(), test_var_idx(point, test_index))
        self.G.add_cpds(
            TabularCPD(
                test_var_idx(point, test_index),
                2,
                [[1 if good else 1 - s for s in self.theta_discretisation for good in [True, False]],  # PASS
                 [0 if good else s for s in self.theta_discretisation for good in [True, False]]],  # FAIL
                [theta_var_idx(), point_var_idx(point)],
                [len(self.theta_discretisation), 2],
                state_names={test_var_idx(point, test_index): [PASS, FAIL], point_var_idx(point): [GOOD, BAD],
                             theta_var_idx(): self.theta_discretisation}
            )
        )
        return test_var_idx(point, test_index)

    def remove_test_node(self, point: int):
        self.query_cache.clear()
        self.test_nodes[point] -= 1
        test_index = self.test_nodes[point]
        self.G.remove_node(test_var_idx(point, test_index))

    def report_test_result(self, point: int, passed: bool):
        self.add_test_node(point)
        self.tests[point][PASS if passed else FAIL] += 1

    def get_test_evidence(self):
        return {test_var_idx(i, j): PASS if j < self.tests[i][PASS] else FAIL for i in range(self.num_points) for j in range(self.tests[i][PASS] + self.tests[i][FAIL])}

    def prob_point_bad(self, point: int):
        q = self.query([error_var_idx()])
        return q.values[point]

    def entropy(self):
        p = self.query([error_var_idx()])
        e = entropy(p)
        return e

    def information_gain(self, point: int, with_test_var: bool = False):
        test_var = self.add_test_node(point)
        p = self.query([error_var_idx(), test_var])
        ig = (entropy(p.marginalize([test_var], inplace=False), with_vars={test_var} if with_test_var else set())
              - entropy(p, p.marginalize([error_var_idx()], inplace=False), with_vars={test_var} if with_test_var else set()))
        self.remove_test_node(point)
        return ig

    def best_test_point(self):
        # print('IG', [self.information_gain(i, True) for i in range(self.num_points)])
        return max(range(self.num_points), key=self.information_gain)

    def theta_given_evidence(self):
        p = self.query([theta_var_idx()])
        return p
