import numpy as np

from probability_table import Variable, ProbabilityTable, entropy


# GOOD = 0
# BAD = 1
# PASS = 10
# FAIL = 11
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

class ProbabilisticBisection:
    def __init__(self, num_points: int, p_bad: float = 0.9, good_certainty: float = 0.9, discretisation_steps: int = 6):
        """

        :param num_points: number of points to test
        :param p_bad: probability that there actually is a reproducible error
        """
        self.good_offset = (num_points * 1 / (1 - good_certainty)) if good_certainty < 1 else None

        self.error_var = Variable(error_var_idx(), list(range(num_points)))
        self.error_pt = ProbabilityTable.from_dict(
                    [self.error_var],
                    {i: p_bad * (1 / num_points) if i < num_points else (1 - p_bad) for i in range(num_points + 1)},
                    []
        )

        self.points = [Variable(point_var_idx(i), [GOOD, BAD]) for i in range(num_points)]
        self.points_pts = [
            ProbabilityTable.from_dict([self.error_var, self.points[i]], {
                j: ({GOOD: 1, BAD: 0} if i < j or j == num_points else {GOOD: 0, BAD: 1}) for j in range(num_points + 1)
            }, [self.error_var]) for i in range(num_points)
        ]

        self.theta_var = Variable(theta_var_idx(), [np.exp(-1/discretisation_steps * i**2) for i in range(discretisation_steps)])
        self.theta_pt = ProbabilityTable.from_dict([self.theta_var], {s: 1 / discretisation_steps for s in self.theta_var.domain})

        self.tests = [{PASS: 0, FAIL: 0} for _ in range(num_points)]

    def report_test_result(self, point: int, passed: bool):
        self.tests[point][PASS if passed else FAIL] += 1
        if hasattr(self, 'prob_point_cache'):
            delattr(self, 'prob_point_cache')
        if hasattr(self, 'joint_probability_with_test_evidence_cache'):
            delattr(self, 'joint_probability_with_test_evidence_cache')

    def _test_probability_table(self, point: int, test: Variable):
        return ProbabilityTable.from_dict([self.theta_var, self.points[point], test], {
            s: {
                GOOD: {
                    PASS: 1,
                    FAIL: 0,
                },
                BAD: {
                    PASS: 1 - s,
                    FAIL: s
                }
            } for s in self.theta_var.domain
        }, [self.theta_var, self.points[point]])

    def joint_probability(self):
        """P(P_0,P_1,...,P_n) = P(P_0) * P(P_1|P_0) * ... * P(P_n|P_{n-1})"""
        if hasattr(self, 'joint_probability_cache'):
            return getattr(self, 'joint_probability_cache')

        p = ProbabilityTable.unit()
        for i in range(len(self.points)):
            p *= self.points_pts[i]
        self.joint_probability_cache = p
        assert False
        return p

    def joint_probability_with_test_evidence(self, extra_test_point: int = -1):
        """P(P_0,...,P_n|T_0=,T_1=,...)"""
        p = self.error_pt
        for i in range(len(self.points)):
            p *= self.points_pts[i]
            for j, result in enumerate([PASS] * self.tests[i][PASS] + [FAIL] * self.tests[i][FAIL]):
                test_var = Variable(test_var_idx(i, j), [PASS, FAIL])
                p *= self._test_probability_table(i, test_var)
                q = p.marginalize_but(test_var, self.theta_var)
                p = (p / q).evidence(test_var, result)

            if extra_test_point == i:
                test_var = Variable(test_var_idx(i, self.tests[i][PASS] + self.tests[i][FAIL]), [PASS, FAIL])
                p *= self._test_probability_table(i, test_var)
            p = p.marginalize(self.points[i])

        p = (p * self.theta_pt).marginalize(self.theta_var)
        return p

    def information_gain(self, point: int):
        # Return the information gain of running another test on this point
        p_E_Ti = self.joint_probability_with_test_evidence(point)

        return entropy(p_E_Ti.marginalize_but(self.error_var)) - entropy(p_E_Ti, p_E_Ti.marginalize(self.error_var))

    def theta_given_evidence(self):
        p_theta = self.theta_pt
        for i in range(len(self.points)):
            for j, result in enumerate([PASS] * self.tests[i][PASS] + [FAIL] * self.tests[i][FAIL]):
                p_E_Xi = self.error_pt * self.points_pts[i]
                test_var = Variable(test_var_idx(i, j), [PASS, FAIL])
                p_E_Xi_Tij_theta = p_E_Xi * self._test_probability_table(i, test_var) * p_theta
                p_E_theta_given_Tij = p_E_Xi_Tij_theta.marginalize(self.points[i]) / p_E_Xi_Tij_theta.marginalize_but(test_var)
                p_theta = p_E_theta_given_Tij.evidence(test_var, result).marginalize(self.error_var)

        return p_theta

    def best_test_point(self):
        print('IG', [self.information_gain(i) for i in range(len(self.points))])
        return max(range(len(self.points)), key=self.information_gain)

    def prob_point(self, point: int):
        if hasattr(self, 'prob_point_cache') and point in getattr(self, 'prob_point_cache'):
            return getattr(self, 'prob_point_cache')[point]

        p_E_theta_given_T = self.joint_probability_with_test_evidence()
        p_E_Xi_given_T = p_E_theta_given_T * self.points_pts[point]
        p_Xi_given_T = p_E_Xi_given_T.marginalize(self.error_var, self.theta_var)

        if not hasattr(self, 'prob_point_cache'):
            self.prob_point_cache = {}
        self.prob_point_cache[point] = p_Xi_given_T
        return p_Xi_given_T

    def prob_point_bad(self, point: int):
        p_E_given_T = self.joint_probability_with_test_evidence()
        return p_E_given_T.index({self.error_var: point})

    def update_theta(self):
        if self.good_offset is not None:
            self.true_positive_rate = (self.good_offset + sum(self.prob_point(i).index({self.points[i]: GOOD}) * (self.tests[i][PASS] + 1) for i in range(len(self.points)))) \
                                  / (self.good_offset + sum(self.prob_point(i).index({self.points[i]: GOOD}) * (self.tests[i][PASS] + self.tests[i][FAIL] + 2) for i in range(len(self.points))))
        else:
            self.true_positive_rate = 1.0
        self.true_negative_rate = sum(self.prob_point(i).index({self.points[i]: BAD}) * (self.tests[i][FAIL] + 1) for i in range(len(self.points))) \
                                  / sum(self.prob_point(i).index({self.points[i]: BAD}) * (self.tests[i][PASS] + self.tests[i][FAIL] + 2) for i in range(len(self.points)))

        if hasattr(self, 'prob_point_cache'):
            delattr(self, 'prob_point_cache')
        if hasattr(self, 'joint_probability_with_test_evidence_cache'):
            delattr(self, 'joint_probability_with_test_evidence_cache')
        if hasattr(self, 'joint_probability_cache'):
            delattr(self, 'joint_probability_cache')

if __name__ == '__main__':
    b = ProbabilisticBisection(3, 0.9)
    print(b.joint_probability())

