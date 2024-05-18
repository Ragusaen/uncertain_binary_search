import copy
import functools
from dataclasses import dataclass
from typing import Iterable, TypeVar, List, Sequence, Any

import numpy as np

IdT = TypeVar('IdT')
DomainT = TypeVar('DomainT')
@dataclass
class Variable:
    id: IdT
    domain: Sequence[DomainT]

    def __hash__(self):
        return id.__hash__()


NDDict = dict[Any, 'NDDict | float']


class ProbabilityTable:
    EPSILON = 1e-6
    def __init__(self, variables: Iterable[Variable], probabilities: np._typing.NDArray, conditional_variables: Iterable[Variable] = None, factor=False, fixed=None):
        assert not isinstance(probabilities, dict)
        self.factor = factor
        self.fixed = {} if fixed is None else fixed
        self.variables = list(variables)
        self.conditional_variables = list(conditional_variables) if conditional_variables else []
        assert set(self.conditional_variables).issubset(set(variables))
        self.array = probabilities

        # self.check_consistency()

    @classmethod
    def from_dict(cls, variables: Iterable[Variable], probabilities: NDDict, conditional_variables: Iterable[Variable] = None):
        array = np.zeros(tuple(len(x.domain) for x in variables))

        with np.nditer(array, flags=['multi_index'], op_flags=[['writeonly']]) as it:
            for x in it:
                lookup = probabilities
                index = it.multi_index
                for v in variables:
                    # print(lookup, v.domain[index[0]], index)
                    lookup = lookup[v.domain[index[0]]]
                    index = index[1:]
                x[...] = lookup
        return ProbabilityTable(variables, array, conditional_variables)

    @classmethod
    def unit(cls):
        return ProbabilityTable([], np.int_(1), [])

    def get_unconditional_indices(self):
        return tuple(i for i, v in enumerate(self.variables) if v not in self.conditional_variables)

    def check_consistency(self):
        if not self.factor:
            sum = np.sum(self.array, axis=self.get_unconditional_indices())
            if not ((1 - self.EPSILON < sum) & (sum < 1 + self.EPSILON)).all():
                raise Exception(f'Probability table is inconsistent, the axis sum to: {list(zip(self.conditional_variables, list(sum))) if len(self.conditional_variables) > 0 else sum}')

    def __mul__(self, other: 'ProbabilityTable'):
        if not self.factor and not other.factor:
            assert len((set(self.variables).difference(self.conditional_variables))
                       .intersection((set(other.variables).difference(other.conditional_variables)))) == 0
            # assert set(self.conditional_variables).issubset(set(other.variables))
            # assert set(other.conditional_variables).issubset(set(self.variables))
        variables = list(set(self.variables + other.variables))
        unconditional_variables = (set(self.variables).difference(self.conditional_variables)).union(set(other.variables).difference(other.conditional_variables))
        conditional_variables = (set(self.conditional_variables).union(other.conditional_variables)).difference(unconditional_variables)

        extra_zeros = range(len(self.variables), len(variables)).__iter__()
        left_array = np.transpose(
            np.expand_dims(self.array, axis=tuple(copy.copy(extra_zeros))),
            axes=tuple(self.variables.index(variables[i])
                       if variables[i] in self.variables
                       else next(extra_zeros) for i in range(len(variables))),
        )
        extra_zeros = range(len(other.variables), len(variables)).__iter__()
        right_array = np.transpose(
            np.expand_dims(other.array, axis=tuple(copy.copy(extra_zeros))),
            axes=tuple(other.variables.index(variables[i])
                       if variables[i] in other.variables
                       else next(extra_zeros) for i in range(len(variables))),
        )

        prod = left_array * right_array

        new_fixed = copy.copy(self.fixed)
        new_fixed.update(other.fixed)

        prod = np.nan_to_num(prod)
        return ProbabilityTable(variables, prod, conditional_variables, factor=self.factor or other.factor, fixed=new_fixed)

    def __truediv__(self, other: 'ProbabilityTable'):
        if not self.factor and other.factor:
            assert ((set(other.variables).difference(set(other.conditional_variables)))
                    .issubset(set(self.variables).difference(self.conditional_variables)))
            assert set(self.conditional_variables) == set(other.conditional_variables)
        variables = list(set(self.variables + other.variables))
        conditional_variables = (set(self.conditional_variables).union(other.variables))

        extra_zeros = range(len(self.variables), len(variables)).__iter__()
        left_array = np.transpose(
            np.expand_dims(self.array, axis=tuple(copy.copy(extra_zeros))),
            axes=tuple(self.variables.index(variables[i])
                       if variables[i] in self.variables
                       else next(extra_zeros) for i in range(len(variables))),
        )
        extra_zeros = range(len(other.variables), len(variables)).__iter__()
        right_array = np.transpose(
            np.expand_dims(other.array, axis=tuple(copy.copy(extra_zeros))),
            axes=tuple(other.variables.index(variables[i])
                       if variables[i] in other.variables
                       else next(extra_zeros) for i in range(len(variables))),
        )

        prod = left_array / right_array

        new_fixed = copy.copy(self.fixed)
        new_fixed.update(other.fixed)
        return ProbabilityTable(variables, prod, conditional_variables, factor=self.factor or other.factor, fixed=new_fixed)

    def marginalize(self, *marginalize: Variable):
        assert all(v in self.variables and v not in self.conditional_variables for v in marginalize)
        variables = [v for v in self.variables if v not in marginalize]
        conditional_variables = self.conditional_variables
        array = np.sum(self.array, axis=tuple(self.variables.index(v) for v in marginalize))

        return ProbabilityTable(variables, array, conditional_variables, factor=self.factor, fixed=self.fixed)

    def marginalize_but(self, *keep: Variable) -> 'ProbabilityTable':
        return self.marginalize(*set(self.variables).difference(keep))

    def evidence(self, variable: Variable, value: DomainT):
        assert variable in self.conditional_variables
        i = self.variables.index(variable)
        j = variable.domain.index(value)
        new_variables = copy.copy(self.variables)
        new_variables.remove(variable)
        new_conditionals = list(self.conditional_variables)
        new_conditionals.remove(variable)
        array = self.array[(slice(None),) * i + (j,)]
        new_fixed = copy.copy(self.fixed)
        new_fixed[variable] = value
        return ProbabilityTable(new_variables, array, new_conditionals, factor=self.factor, fixed=new_fixed)

    def numpy_apply(self, func, factor: bool = False):
        return ProbabilityTable(self.variables, func(self.array), self.conditional_variables, factor=factor or self.factor)

    def index(self, values: dict):
        idx = tuple(v.domain.index(values[v]) for v in self.variables)
        return self.array[idx]

    def name(self):
        return f'P({",".join(str(x.id) for x in self.variables if x not in self.conditional_variables)}|{",".join([str(x.id) for x in self.conditional_variables] + [f"{x.id}={v}" for x, v in self.fixed.items()])})'

    def __str__(self):
        # mid = len(self.variables) // 2 + 1
        # top_widths = functools.reduce(lambda a, x: [max(max(len(str(d)) for d in self.variables[x].domain), 1 + len(self.variables[x].domain) * a[0])] + a, range(mid), [0])[:-1]
        # left_widths = functools.reduce(lambda a, x: [max(max(len(str(d)) for d in self.variables[x].domain), 1 + len(self.variables[x].domain) * a[0])] + a, range(mid, len(self.variables)), [0])[:-1]
        # max_var_length = max(len(str(v.id)) for v in self.variables)
        #
        # repeat = 1
        # top_header = ''
        # for i in range(mid):
        #     for _ in range(repeat):
        #         for d in self.variables[i].domain:
        #             top_header += '|' + str(d).center(top_widths[i] - 1)
        #     repeat *= len(self.variables[i].domain)
        #     top_header += '| ' + str(self.variables[i].id) + '\n'
        #
        # left_header = [str(self.variables[i].id).center(max_var_length) for i in range(mid, len(self.variables))]
        # repeat = 1
        # for i in

        return self.name() + f' = \n{self.array}'

    def __repr__(self):
        return self.__str__()

def entropy(joint_probability: ProbabilityTable, conditional_joint_probability: ProbabilityTable = ProbabilityTable.unit()):
    assert len(joint_probability.conditional_variables) == 0
    assert len(conditional_joint_probability.conditional_variables) == 0
    assert set(conditional_joint_probability.variables).issubset(joint_probability.variables)

    a = joint_probability / conditional_joint_probability
    b = a.numpy_apply(np.log2, factor=True).numpy_apply(np.nan_to_num)
    f = joint_probability * b
    return - np.sum(f.array)



if __name__ == '__main__':
    X = Variable('X', ['true', 'false'])
    Y = Variable('Y', ['low', 'mid', 'high'])
    p = ProbabilityTable.from_dict([X, Y], {
        'true': {
            'low': 0.3,
            'mid': 0.5,
            'high': 0.9,
        },
        'false': {
            'low': 0.7,
            'mid': 0.5,
            'high': 0.1,
        }
    }, [Y])

    q = ProbabilityTable.from_dict([Y], {
        'low': 0.5,
        'mid': 0.3,
        'high': 0.2,
    })

    print(p.evidence(Y, 'low'))