"""
calculator.py
-------------
Branch: feature/calculator-tool

A small, safe calculator that evaluates a mathematical expression
and returns the result.

Security note
-------------
Python's built-in eval() executes arbitrary code, so it is NOT used here.
Instead, the expression is parsed into an Abstract Syntax Tree (ast) and
only a safe, known set of node types is allowed.
Anything outside that set raises a ValueError immediately.

Supported operations
--------------------
  +   addition
  -   subtraction
  *   multiplication
  /   division
  ()  parentheses
  unary + and -  (e.g. -5, +3)
  decimal numbers (e.g. 10.5)

Usage
-----
  from backend.tools.calculator import calculate

  result = calculate("25 * 18")       # -> 450
  result = calculate("(10 + 5) * 2")  # -> 30
"""

import ast
import operator


# Map each allowed AST binary-operator node to its Python function.
# Any operator NOT listed here is automatically rejected.
OPERATORS = {
    ast.Add:  operator.add,       # +
    ast.Sub:  operator.sub,       # -
    ast.Mult: operator.mul,       # *
    ast.Div:  operator.truediv,   # /
}


def calculate(expression):
    """
    Evaluate a safe mathematical expression and return the result.

    Parameters
    ----------
    expression : str
        A mathematical expression, e.g. "25 * 18" or "(10 + 5) / 3".

    Returns
    -------
    int or float
        The calculated result.

    Raises
    ------
    ValueError
        If the expression is empty, invalid, contains unsupported
        operators, or attempts division by zero.
    """

    # Guard: reject empty input
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    try:
        # Parse the expression into an Abstract Syntax Tree.
        # mode="eval" means only a single expression is allowed -
        # statements like assignments or imports are rejected here.
        tree = ast.parse(expression.strip(), mode="eval")

        return _evaluate(tree.body)

    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")

    except ValueError:
        # Re-raise ValueErrors from _evaluate (they already have clear messages).
        raise

    except SyntaxError:
        raise ValueError("Invalid mathematical expression.")

    except Exception:
        raise ValueError("Invalid mathematical expression.")


def _evaluate(node):
    """
    Recursively walk an AST node and compute its value.

    Only numeric constants, binary operations (+, -, *, /),
    and unary operations (+, -) are allowed.

    Any other node type is rejected immediately - this is what
    prevents arbitrary code execution.
    """

    # Numeric constant: 42, 3.14, etc.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        # Strings, booleans, or other literals are not allowed.
        raise ValueError("Only numeric values are allowed.")

    # Binary operation: left OP right
    if isinstance(node, ast.BinOp):
        op_func = OPERATORS.get(type(node.op))

        if op_func is None:
            raise ValueError(
                f"Unsupported operator: {type(node.op).__name__}. "
                "Only +, -, *, / are allowed."
            )

        left  = _evaluate(node.left)
        right = _evaluate(node.right)

        return op_func(left, right)

    # Unary operation: -5 or +3
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand)

        if isinstance(node.op, ast.UAdd):
            return +value

        if isinstance(node.op, ast.USub):
            return -value

        raise ValueError("Unsupported unary operator.")

    # Anything else (function calls, names, imports, etc.) is rejected.
    raise ValueError("Invalid mathematical expression.")
