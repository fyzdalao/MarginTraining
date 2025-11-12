"""
使用 Spearman 与 Kendall 检验评估不同改进方法与模型深度的关联。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass(frozen=True)
class ModelResult:
    name: str
    depth: int
    base: float
    improved_a: float
    improved_b: float
    improved_ab: float

    def deltas(self) -> Dict[str, float]:
        """返回各改进方法相对于基线的增益。"""
        return {
            "A": self.improved_a - self.base,
            "B": self.improved_b - self.base,
            "A+B": self.improved_ab - self.base,
        }


def build_dataset() -> List[ModelResult]:
    """构建模型结果列表，深度按模型 a→f 递增记为 1→6。"""
    raw_values: Iterable[Tuple[str, float, float, float, float]] = (
        ("a", 22.60, 26.13, 26.33, 40.47),
        ("b", 23.60, 29.47, 31.00, 42.60),
        ("c", 25.00, 34.40, 30.00, 51.67),
        ("d", 20.73, 30.73, 26.60, 54.27),
        ("e", 26.20, 29.27, 27.87, 57.13),
        ("f", 23.87, 30.33, 39.60, 63.47),
    )

    dataset: List[ModelResult] = [
        ModelResult(
            name=name,
            depth=index + 1,
            base=base,
            improved_a=improved_a,
            improved_b=improved_b,
            improved_ab=improved_ab,
        )
        for index, (name, base, improved_a, improved_b, improved_ab) in enumerate(
            raw_values
        )
    ]
    return dataset


def compute_rank_correlations(
    depths: np.ndarray, deltas: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """计算 Spearman ρ 及 Kendall τ 与对应 p 值。"""
    rho, p_rho = spearmanr(depths, deltas)
    tau, p_tau = kendalltau(depths, deltas)
    return (rho, p_rho), (tau, p_tau)


def main() -> None:
    dataset = build_dataset()
    depths = np.array([result.depth for result in dataset], dtype=float)

    methods = ("A", "B", "A+B")
    deltas_by_method: Dict[str, np.ndarray] = {
        method: np.array([result.deltas()[method] for result in dataset], dtype=float)
        for method in methods
    }

    print("模型深度：a→f 映射为 1→6，样本量 =", depths.size)
    for method in methods:
        deltas = deltas_by_method[method]
        (rho, p_rho), (tau, p_tau) = compute_rank_correlations(depths, deltas)

        print(f"\n方法 {method} 的改进（改进-基线）：")
        print("增益序列：", ", ".join(f"{value:.2f}" for value in deltas))
        print(f"Spearman ρ = {rho:.4f}, p = {p_rho:.4f}")
        print(f"Kendall τ  = {tau:.4f}, p = {p_tau:.4f}")

    print(
        "\n结论提示：p 值 < 0.05 时可视为在 95% 置信水平下显著；"
        "当前样本量仅 6，解释时需注意小样本带来的不确定性。"
    )


if __name__ == "__main__":
    main()

