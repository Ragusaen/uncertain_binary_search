import random


class TestSimulator:
    def __init__(self, points: int, true_positive_rate: float, true_negative_rate: float, seed=None):
        if seed is not None:
            random.seed(seed)

        self.true_positive_rate = true_positive_rate
        self.true_negative_rate = true_negative_rate
        self.bad_point = random.randint(0, points - 1)


    def run_test(self, point: int):
        return random.choices(['Pass', 'Fail'],
                              [1 - self.true_negative_rate, self.true_negative_rate]
                              if point >= self.bad_point else
                              [self.true_positive_rate, 1 - self.true_positive_rate])[0]