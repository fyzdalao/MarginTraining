"""
使用 ART (Aligned Rank Transform) 方法对 2x2 因子设计执行方差分析，评估改进方法 Rsm、LM 以及它们的交互作用对攻击准确率的影响。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import pandas as pd
from scipy.stats import rankdata
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class Condition:
    apply_rsm: int
    apply_lm: int
    accuracy: float


RAW_RESULTS: Dict[str, Dict[str, Condition]] = {
    "a": {
        "none": Condition(0, 0, 22.60),
        "Rsm": Condition(1, 0, 26.13),
        "LM": Condition(0, 1, 26.33),
        "Rsm+LM": Condition(1, 1, 40.47),
    },
    "b": {
        "none": Condition(0, 0, 23.60),
        "Rsm": Condition(1, 0, 29.47),
        "LM": Condition(0, 1, 31.00),
        "Rsm+LM": Condition(1, 1, 42.60),
    },
    "c": {
        "none": Condition(0, 0, 25.00),
        "Rsm": Condition(1, 0, 34.40),
        "LM": Condition(0, 1, 30.00),
        "Rsm+LM": Condition(1, 1, 51.67),
    },
    "d": {
        "none": Condition(0, 0, 20.73),
        "Rsm": Condition(1, 0, 30.73),
        "LM": Condition(0, 1, 26.60),
        "Rsm+LM": Condition(1, 1, 54.27),
    },
    "e": {
        "none": Condition(0, 0, 26.20),
        "Rsm": Condition(1, 0, 29.27),
        "LM": Condition(0, 1, 27.87),
        "Rsm+LM": Condition(1, 1, 57.13),
    },
    "f": {
        "none": Condition(0, 0, 23.87),
        "Rsm": Condition(1, 0, 30.33),
        "LM": Condition(0, 1, 39.60),
        "Rsm+LM": Condition(1, 1, 63.47),
    },
    "g": {
        "none": Condition(0, 0, 16.93),
        "Rsm": Condition(1, 0, 22.40),
        "LM": Condition(0, 1, 29.60),
        "Rsm+LM": Condition(1, 1, 31.20),
    },
    "h": {
        "none": Condition(0, 0, 15.67),
        "Rsm": Condition(1, 0, 31.93),
        "LM": Condition(0, 1, 22.47),
        "Rsm+LM": Condition(1, 1, 32.73),
    },
    "i": {
        "none": Condition(0, 0, 21.87),
        "Rsm": Condition(1, 0, 29.20),
        "LM": Condition(0, 1, 28.20),
        "Rsm+LM": Condition(1, 1, 43.00),
    },
}


def build_dataframe(raw: Dict[str, Dict[str, Condition]]) -> pd.DataFrame:
    rows: Iterable[Tuple[str, Condition]] = (
        (model, condition)
        for model, conds in raw.items()
        for condition in conds.values()
    )
    df = pd.DataFrame(
        (
            {
                "model": model,
                "Rsm": cond.apply_rsm,
                "LM": cond.apply_lm,
                "accuracy": cond.accuracy,
            }
            for model, cond in rows
        )
    )
    df["model"] = df["model"].astype("category")
    df["Rsm"] = df["Rsm"].astype("category")
    df["LM"] = df["LM"].astype("category")
    return df


def align_for_effect(df: pd.DataFrame, effect: str) -> pd.Series:
    grand_mean = df["accuracy"].mean()
    means_rsm = df.groupby("Rsm", observed=False)["accuracy"].mean()
    means_lm = df.groupby("LM", observed=False)["accuracy"].mean()
    means_interaction = df.groupby(["Rsm", "LM"], observed=False)["accuracy"].mean()
    means_model = df.groupby("model", observed=False)["accuracy"].mean()

    def align_row(row: pd.Series) -> float:
        acc = row["accuracy"]
        mean_interaction = means_interaction.loc[row["Rsm"], row["LM"]]
        if effect == "Rsm":
            mean_rsm = means_rsm.loc[row["Rsm"]]
            return acc - mean_interaction + mean_rsm
        if effect == "LM":
            mean_lm = means_lm.loc[row["LM"]]
            return acc - mean_interaction + mean_lm
        if effect == "Rsm:LM":
            mean_rsm = means_rsm.loc[row["Rsm"]]
            mean_lm = means_lm.loc[row["LM"]]
            return acc - mean_rsm - mean_lm + grand_mean
        if effect == "model":
            mean_model = means_model.loc[row["model"]]
            return acc - mean_model + grand_mean
        raise ValueError(f"Unsupported effect: {effect}")

    return df.apply(align_row, axis=1)


def art_anova(df: pd.DataFrame) -> Dict[str, pd.Series]:
    results: Dict[str, pd.Series] = {}
    for effect in ("Rsm", "LM", "Rsm:LM"):
        aligned = align_for_effect(df, effect)
        ranked = rankdata(aligned)
        data = df.copy()
        data["aligned"] = aligned
        data["rank"] = ranked

        model = smf.ols("rank ~ C(Rsm) * C(LM) + C(model)", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        effect_name = {
            "Rsm": "C(Rsm)",
            "LM": "C(LM)",
            "Rsm:LM": "C(Rsm):C(LM)",
        }[effect]
        results[effect] = anova_table.loc[effect_name]

    return results


def art_combo_effect(df: pd.DataFrame) -> pd.Series:
    combo_df = df[df["Rsm"] == df["LM"]].copy()
    combo_df["combo"] = combo_df["Rsm"].cat.codes.astype(int)
    combo_df["combo"] = combo_df["combo"].astype("category")
    combo_df["rank"] = rankdata(combo_df["accuracy"])
    model = smf.ols("rank ~ C(combo) + C(model)", data=combo_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table.loc["C(combo)"]


def art_model_effect(df: pd.DataFrame) -> pd.Series:
    aligned = align_for_effect(df, "model")
    ranked = rankdata(aligned)
    data = df.copy()
    data["aligned"] = aligned
    data["rank"] = ranked

    model = smf.ols("rank ~ C(model) + C(Rsm) * C(LM)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table.loc["C(model)"]


def main() -> None:
    df = build_dataframe(RAW_RESULTS)
    results = art_anova(df)
    combo = art_combo_effect(df)
    model_effect = art_model_effect(df)

    print("ART 方差分析结果（Typ II ANOVA，秩统计量）")
    print("-" * 60)
    for effect in ("Rsm", "LM", "Rsm:LM"):
        row = results[effect]
        print(
            f"{effect:>6} -> sum_sq={row['sum_sq']:.3f}, "
            f"df={row['df']:.0f}, F={row['F']:.3f}, p={row['PR(>F)']:.10f}"
        )
    print("-" * 60)
    print(
        "Rsm+LM 联合条件（与无改进对比） -> "
        f"sum_sq={combo['sum_sq']:.3f}, df={combo['df']:.0f}, "
        f"F={combo['F']:.3f}, p={combo['PR(>F)']:.10f}"
    )
    print("-" * 60)
    print(
        "模型主效应 -> "
        f"sum_sq={model_effect['sum_sq']:.3f}, df={model_effect['df']:.0f}, "
        f"F={model_effect['F']:.3f}, p={model_effect['PR(>F)']:.10f}"
    )


if __name__ == "__main__":
    main()
