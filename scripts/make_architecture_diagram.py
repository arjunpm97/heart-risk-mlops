"""Render the system architecture diagram to reports/figures/architecture.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

LANES = [  # (y, title, colour)
    (7.6, "1  Data & experimentation", "#e9f5f3"),
    (5.1, "2  CI/CD - GitHub Actions", "#fdf1ec"),
    (2.6, "3  Serving - Kubernetes", "#eef1f8"),
]

BOXES = {
    # key: (x, y, w, h, label)
    "uci": (0.3, 7.9, 2.0, 1.1, "UCI repository\nprocessed.cleveland"),
    "dl": (2.9, 7.9, 2.2, 1.1, "download_data.py\nclean + validate"),
    "eda": (5.7, 7.9, 2.0, 1.1, "EDA\nnotebook / figures"),
    "train": (8.3, 7.9, 2.6, 1.1, "train.py\nGridSearchCV x 3 models"),
    "mlflow": (11.5, 7.9, 2.4, 1.1, "MLflow tracking\nparams, metrics, plots"),
    "repo": (0.3, 5.4, 2.0, 1.1, "GitHub repo\npush / PR"),
    "lint": (2.9, 5.4, 1.7, 1.1, "lint\nflake8"),
    "test": (5.1, 5.4, 1.7, 1.1, "test\npytest + cov"),
    "ci_train": (7.3, 5.4, 2.2, 1.1, "train\nquality gate AUC"),
    "docker": (10.0, 5.4, 2.0, 1.1, "docker build\n+ smoke test"),
    "ghcr": (12.5, 5.4, 1.4, 1.1, "GHCR\nimage"),
    "client": (0.3, 2.9, 2.0, 1.1, "Client\ncurl / app"),
    "lb": (2.9, 2.9, 2.2, 1.1, "Ingress / LoadBalancer\nService :80"),
    "pods": (5.7, 2.9, 2.8, 1.1, "Deployment (2-5 pods, HPA)\nFastAPI + model.joblib"),
    "prom": (9.1, 2.9, 2.0, 1.1, "Prometheus\nscrapes /metrics"),
    "graf": (11.7, 2.9, 2.2, 1.1, "Grafana\ndashboard"),
    "logs": (5.7, 0.6, 2.8, 1.0, "JSON request logs\nkubectl logs / stdout"),
}

ARROWS = [
    ("uci", "dl"), ("dl", "eda"), ("eda", "train"), ("train", "mlflow"),
    ("repo", "lint"), ("lint", "test"), ("test", "ci_train"), ("ci_train", "docker"),
    ("docker", "ghcr"), ("client", "lb"), ("lb", "pods"), ("pods", "prom"), ("prom", "graf"),
]


def centre(key, side):
    x, y, w, h, _ = BOXES[key]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y),
            "t": (x + w / 2, y + h)}[side]


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 9.6))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0.2, 10.2)
    ax.axis("off")
    for y, title, colour in LANES:
        ax.add_patch(FancyBboxPatch((0.1, y), 14.0, 1.9, boxstyle="round,pad=0.02",
                                    fc=colour, ec="#cccccc"))
        ax.text(0.2, y + 1.75, title, fontsize=11, weight="bold", va="top", color="#333")
    for key, (x, y, w, h, label) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    fc="white", ec="#264653", lw=1.4))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for a, b in ARROWS:
        ax.add_patch(FancyArrowPatch(centre(a, "r"), centre(b, "l"), arrowstyle="-|>",
                                     mutation_scale=14, color="#264653", lw=1.3))
    extra = [  # cross-lane arrows
        (centre("train", "b"), (8.4, 6.5), "model.joblib\n+ metadata"),
        (centre("ghcr", "b"), (7.1, 4.0), "kubectl apply\nk8s/*.yaml"),
        (centre("pods", "b"), centre("logs", "t"), ""),
    ]
    for start, end, label in extra:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     color="#e76f51", lw=1.5, linestyle="--",
                                     connectionstyle="arc3,rad=0.15"))
        if label:
            ax.text((start[0] + end[0]) / 2 + 0.15, (start[1] + end[1]) / 2, label,
                    fontsize=8, color="#e76f51")
    ax.set_title("Heart-disease risk MLOps platform - architecture", fontsize=14, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
