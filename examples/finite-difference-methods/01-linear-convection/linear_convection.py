from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth Gaussian pulse used as the initial condition."""
    return np.exp(-120.0 * (x - 0.35) ** 2)


def analytic_solution(
    x: np.ndarray,
    t: float,
    c: float,
    domain_length: float,
) -> np.ndarray:
    """Periodic analytic solution u(x,t) = u0(x - c t)."""
    shifted_x = (x - c * t) % domain_length
    return initial_condition(shifted_x)


def upwind_step(u: np.ndarray, cfl: float) -> np.ndarray:
    """
    One explicit first-order upwind step for positive convection speed.

    For c > 0:
        u_i^{n+1} = u_i^n - CFL * (u_i^n - u_{i-1}^n)

    Periodic boundary conditions are handled by np.roll.
    """
    return u - cfl * (u - np.roll(u, 1))


def compute_l2_error(u_num: np.ndarray, u_exact: np.ndarray) -> float:
    """Root-mean-square L2 error."""
    return float(np.sqrt(np.mean((u_num - u_exact) ** 2)))


def run_simulation(
    *,
    nx: int,
    c: float,
    cfl: float,
    final_time: float,
    domain_length: float,
) -> dict:
    """Run the 1-D linear convection problem for one CFL number."""
    x = np.linspace(0.0, domain_length, nx, endpoint=False)
    dx = domain_length / nx
    dt = cfl * dx / c

    # With final_time = 0.96, nx = 100, and CFL in [0.8, 1.0, 1.2],
    # every case lands exactly on the same final physical time.
    n_steps = int(round(final_time / dt))
    actual_final_time = n_steps * dt

    u = initial_condition(x)

    for _ in range(n_steps):
        u = upwind_step(u, cfl)

    u_exact = analytic_solution(x, actual_final_time, c, domain_length)
    error = compute_l2_error(u, u_exact)

    stability = "Stable" if cfl <= 1.0 else "Unstable: CFL > 1"

    return {
        "cfl": cfl,
        "x": x,
        "u": u,
        "u_exact": u_exact,
        "dx": dx,
        "dt": dt,
        "n_steps": n_steps,
        "actual_final_time": actual_final_time,
        "error": error,
        "stability": stability,
        "min_u": float(np.min(u)),
        "max_u": float(np.max(u)),
    }


def plot_cfl_sweep(results: list[dict]) -> Path:
    """Create one readable comparison plot for all CFL values."""
    example_dir = Path(__file__).resolve().parent
    output_dir = example_dir / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "linear_convection_cfl_sweep.png"

    fig, ax = plt.subplots(figsize=(12, 7))

    reference = results[0]

    visible_min = -2.0
    visible_max = 2.0

    styles = {
        0.8: {
            "color": "tab:blue",
            "linewidth": 2.2,
            "linestyle": "-",
            "label": "Numerical CFL = 0.8, stable but diffusive",
            "zorder": 3,
        },
        1.0: {
            "color": "tab:orange",
            "linewidth": 2.4,
            "linestyle": "-",
            "label": "Numerical CFL = 1.0, stable",
            "zorder": 6,
        },
        1.2: {
            "color": "tab:red",
            "linewidth": 1.35,
            "linestyle": ":",
            "label": "Numerical CFL = 1.2, unstable, clipped to [-2, 2]",
            "zorder": 2,
        },
    }

    # Plot all numerical solutions as lines.
    for result in results:
        cfl = result["cfl"]
        u_plot = result["u"].copy()

        if cfl > 1.0:
            u_plot = np.clip(u_plot, visible_min, visible_max)

        ax.plot(
            result["x"],
            u_plot,
            **styles[cfl],
        )

    # Plot analytic solution as markers only so it does not hide CFL = 1.0.
    marker_stride = 4
    ax.plot(
        reference["x"][::marker_stride],
        reference["u_exact"][::marker_stride],
        color="black",
        marker="x",
        linestyle="None",
        markersize=6,
        markeredgewidth=1.6,
        label=f"Analytic solution, t = {reference['actual_final_time']:.2f}",
        zorder=10,
    )

    ax.set_title("1-D Linear Convection: CFL Number Comparison", fontsize=16)
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("u(x,t)", fontsize=13)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(visible_min, visible_max)

    ax.grid(True, alpha=0.28)

    ax.legend(
        loc="upper right",
        framealpha=0.95,
        fontsize=10,
    )

    ax.text(
        0.02,
        0.05,
        "CFL = 1.2 violates the stability limit and grows rapidly.\n"
        "Its plotted values are clipped to [-2, 2] so the stable solutions remain readable.",
        transform=ax.transAxes,
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    nx = 100
    c = 1.0
    domain_length = 1.0
    final_time = 0.96
    cfl_values = [0.8, 1.0, 1.2]

    results = [
        run_simulation(
            nx=nx,
            c=c,
            cfl=cfl,
            final_time=final_time,
            domain_length=domain_length,
        )
        for cfl in cfl_values
    ]

    print("--- 1-D Linear Convection: CFL Sweep ---")
    print(f"Number of cells: {nx}")
    print(f"Target final time: {final_time:.4f}")
    print()
    print(
        f"{'CFL':>6} | {'Steps':>6} | {'dt':>10} | "
        f"{'Final t':>10} | {'L2 error':>12} | {'min(u)':>10} | {'max(u)':>10} | Stability"
    )
    print("-" * 104)

    for result in results:
        print(
            f"{result['cfl']:6.1f} | "
            f"{result['n_steps']:6d} | "
            f"{result['dt']:10.4e} | "
            f"{result['actual_final_time']:10.4f} | "
            f"{result['error']:12.4e} | "
            f"{result['min_u']:10.4e} | "
            f"{result['max_u']:10.4e} | "
            f"{result['stability']}"
        )

    output_path = plot_cfl_sweep(results)
    example_dir = Path(__file__).resolve().parent

    print()
    print(f"Plot saved to: {output_path.relative_to(example_dir)}")


if __name__ == "__main__":
    main()