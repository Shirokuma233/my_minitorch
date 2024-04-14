from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    my_vals = list(vals)
    my_vals[arg] += epsilon
    m = f(*my_vals)
    my_vals[arg] -= 2 * epsilon
    n = f(*my_vals)
    return (m - n) / (2 * epsilon)
    raise NotImplementedError('Need to implement for Task 1.1')


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = []
    result = []
    def visit(n: Variable):
        if n.is_constant():
            return
        if n.unique_id in visited:
            return
        if not n.is_leaf():
            for input in n.history.inputs:
                visit(input)
        visited.append(n.unique_id)
        result.insert(0, n)
    visit(variable)
    return result
    raise NotImplementedError('Need to implement for Task 1.4')


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    graph = topological_sort(variable)
    # 注意这个字典存放的是当前节点后面给它自己算出来的d_output
    node2deriv = {}
    node2deriv[variable.unique_id] = deriv
    # 下面遍历整个拓扑顺序图
    for n in graph:
        if n.is_leaf():
            continue
        # 如果这个节点的已在字典出现过，代表他的d_output已经被算过
        if n.unique_id in node2deriv.keys():
            deriv = node2deriv[n.unique_id]
        deriv_tmp = n.chain_rule(deriv)
        for k, v in deriv_tmp:
            if k.is_leaf():
                k.accumulate_derivative(v)
                continue
            if k.unique_id in node2deriv.keys():
                node2deriv[k.unique_id] += v
            else:
                node2deriv[k.unique_id] = v
    return
    raise NotImplementedError('Need to implement for Task 1.4')


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
