from pathlib import Path
import matplotlib.pyplot as plt

class PlotManager:
    """
    Centralized plot saving utility.
    Ensures consistent resolution, naming, and directory structure.
    """

    def __init__(
        self,
        base_dir: str = "outputs/figures/shap",
        dpi: int = 300,
        fmt: str = "png",
        tight_layout: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.fmt = fmt
        self.tight_layout = tight_layout

    def save(self, filename: str):
        """
        Save current matplotlib figure.
        """
        path = self.base_dir / f"{filename}.{self.fmt}"
        if self.tight_layout:
            plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        print(f"[saved] {path.resolve()}")
