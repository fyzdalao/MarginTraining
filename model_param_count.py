import argparse
import sys
from typing import Callable, Optional

try:
    import networks
except ImportError as exc:
    raise SystemExit("未找到`networks`包，请确认项目结构或 PYTHONPATH 配置。") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算指定 WideResNet 模型的参数总量。"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="WideResNet 模型名称，如 wideresnet50_5。",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="分类数，默认为 10。"
    )
    parser.add_argument(
        "--mode",
        default="",
        help="构造模型时的 mode 参数，可选。"
    )
    parser.add_argument(
        "--weight",
        default=None,
        help="构造模型时的 weight 参数，可选。"
    )
    return parser.parse_args()


def format_param_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def resolve_model_fn(model_name: str) -> Callable[..., object]:
    model_fn: Optional[Callable[..., object]] = getattr(networks, model_name, None)
    if model_fn is None:
        available = sorted(
            name for name in dir(networks)
            if not name.startswith("_") and callable(getattr(networks, name))
        )
        raise SystemExit(
            f"未找到模型 `{model_name}`。可用模型包括：{', '.join(available)}。"
        )
    if not callable(model_fn):
        raise SystemExit(f"`{model_name}` 不是可调用的模型构造函数。")
    return model_fn


def main() -> None:
    args = parse_args()

    model_fn = resolve_model_fn(args.model)

    try:
        model = model_fn(num_classes=args.num_classes, mode=args.mode, weight=args.weight)
    except TypeError as exc:
        raise SystemExit(f"模型构造函数参数不匹配：{exc}") from exc

    total_params = sum(p.numel() for p in model.parameters())
    print(format_param_count(total_params))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"程序执行出错：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


# python model_param_count.py --model=wideresnet22_10 26.8M
# python model_param_count.py --model=wideresnet28_8 23.4M
# python model_param_count.py --model=wideresnet40_6 20.1M
# python model_param_count.py --model=wideresnet50_5 16.4M


# python model_param_count.py --model=wideresnet52_5 18.8M
# python model_param_count.py --model=wideresnet76_4 18.3M
# python model_param_count.py --model=wideresnet58_5 21.3M
# python model_param_count.py --model=wideresnet46_6 23.6M
# python model_param_count.py --model=wideresnet82_4 19.8M

#final
# 22-10 26.8 yes
# 28-8 23.4 yes

# 46-6 23.6 yet
# 58-5 21.3 yet
# 52-5 18.8 yet
# 82-4 19.8 yet


