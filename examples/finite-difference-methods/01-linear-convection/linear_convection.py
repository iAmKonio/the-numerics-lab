import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def initial_condition(x):
    """
    Returns a smooth Gaussian pulse.
    """
    # Gaussian centered at x = 0.5 with width roughly 0.1
    return np.exp(-100.0 * (x - 0.5)**2)

def analytic_solution(x, t, c, domain_length):
    """
    Analytic solution for periodic boundary conditions.
    The pulse translates by c*t, wrapping around the domain.
    """
    # Find the shifted coordinate, wrapped to the domain [0, domain_length)
    x_shifted = (x - c * t) % domain_length
    return initial_condition(x_shifted)

def upwind_step(u, sigma):
    """
    Performs one explicit first-order upwind time step.
    u_i^{n+1} = u_i^n - sigma * (u_i^n - u_{i-1}^n)
    Periodic boundary conditions are applied.
    """
    # np.roll(u, 1) shifts elements to the right, so index 0 gets index -1
    u_new = u - sigma * (u - np.roll(u, 1))
    return u_new

def compute_l2_error(u_num, u_exact):
    """
    Computes the L2 norm of the error.
    """
    return np.sqrt(np.sum((u_num - u_exact)**2) / len(u_num))

def run_simulation(nx, c, cfl, target_final_time, domain_length=1.0):
    """
    Runs the linear convection simulation for a fixed number of steps
    to maintain a constant CFL number throughout the run.
    """
    # Spatial grid
    x = np.linspace(0, domain_length, nx, endpoint=False)
    dx = domain_length / nx
    
    # Time step from CFL condition: sigma = c * dt / dx
    dt = cfl * dx / c
    
    # Number of steps to reach roughly target_final_time
    n_steps = max(1, round(target_final_time / dt))
    actual_final_time = n_steps * dt
    
    # Initial condition
    u = initial_condition(x)
    
    for _ in range(n_steps):
        u = upwind_step(u, cfl)
        
    return x, u, n_steps, dt, actual_final_time

def main():
    # Parameters
    nx = 100               # Number of cells
    c = 1.0                # Wave speed
    cfl_values = [0.8, 1.0, 1.2]
    target_final_time = 0.5
    domain_length = 1.0
    
    print(f"--- 1-D Linear Convection (Explicit Upwind FDM) ---")
    print(f"{'CFL':<5} | {'Steps':<5} | {'dt':<8} | {'Final t':<7} | {'L2 Error':<10} | {'Stability'}")
    print("-" * 65)
    
    plt.figure(figsize=(10, 6))
    
    # Spatial grid for analytic/initial plotting
    x = np.linspace(0, domain_length, nx, endpoint=False)
    plt.plot(x, initial_condition(x), '--', color='gray', label='Initial Condition')
    
    # Plot Analytic Solution at the target time as a reference
    u_exact_target = analytic_solution(x, target_final_time, c, domain_length)
    plt.plot(x, u_exact_target, '-k', label=f'Analytic (t={target_final_time})', linewidth=2)
    
    colors = {0.8: 'b', 1.0: 'g', 1.2: 'r'}
    markers = {0.8: 's', 1.0: '^', 1.2: 'x'}
    
    for cfl in cfl_values:
        x_num, u_num, n_steps, dt, actual_final_time = run_simulation(
            nx, c, cfl, target_final_time, domain_length
        )
        
        # Compute exact solution at the ACTUAL final time reached
        u_exact = analytic_solution(x_num, actual_final_time, c, domain_length)
        
        # Compute error
        error = compute_l2_error(u_num, u_exact)
        
        stability = "Stable" if cfl <= 1.0 else "Unstable"
        
        print(f"{cfl:<5.1f} | {n_steps:<5} | {dt:<8.4f} | {actual_final_time:<7.4f} | {error:<10.4e} | {stability}")
        
        plt.plot(x_num, u_num, label=f'CFL = {cfl} ({stability})', 
                 color=colors[cfl], marker=markers[cfl], markerfacecolor='none', linestyle='-', alpha=0.7)

    plt.title(f'1-D Linear Convection CFL Sweep')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    
    # Clip y-axis to avoid unstable CFL=1.2 blowing up the plot
    plt.ylim(-0.2, 1.2)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save output
    example_dir = Path(__file__).resolve().parent
    output_dir = example_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "linear_convection_cfl_sweep.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {output_path.relative_to(example_dir)}")

if __name__ == '__main__':
    main()
