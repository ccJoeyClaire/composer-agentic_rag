from typing import Annotated

from pydantic import Field

from tools.registry import local_tool


@local_tool
def integrate_function(
    func_str: Annotated[
        str, Field(description="可被 sympy.sympify 解析的表达式，如 x**2 + sin(x)")
    ],
    a: Annotated[float, Field(description="积分下限")],
    b: Annotated[float, Field(description="积分上限")],
) -> float:
    """积分计算工具：对 func_str 在 [a, b] 上定积分。"""
    print(f"正在输入表达式 {func_str}")
    import sympy as sp

    x = sp.Symbol("x")
    try:
        expr = sp.sympify(func_str)
        integral = sp.integrate(expr, (x, a, b))
        return float(integral.evalf())
    except Exception as e:
        return f"表达式解析失败: {e}"


def _main() -> None:
    """Offline smoke: definite integral via sympy.

    Run (from repo root):
      python -m tools.LocalTool.math_tool "x**2" 0 1
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Integrate func_str over [a, b].")
    parser.add_argument("func_str", help="SymPy expression, e.g. x**2 + sin(x)")
    parser.add_argument("a", type=float, help="Lower bound")
    parser.add_argument("b", type=float, help="Upper bound")
    args = parser.parse_args()
    result = integrate_function(args.func_str, args.a, args.b)
    print(result)
    if isinstance(result, str):
        sys.exit(1)


if __name__ == "__main__":
    _main()
