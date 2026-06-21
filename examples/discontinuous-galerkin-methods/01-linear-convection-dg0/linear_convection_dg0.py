import numpy as np
import matplotlib.pyplot as plt
import os

def initial_condition(x):
    """
    Smooth Gaussian pulse as the initial condition.
    """
    return np.exp(-200.0 * (x - 0.5)**2)

def analytic_solution(x, t, c, domain_length):
    """
    Analytic solution for 1-D linear convection with periodic boundary conditions.
    """
    x_shifted = (x - c * t) % domain_length
    return initial_condition(x_shifted)

def numerical_flux_upwind(u, c):
    """
    Compute the upwind numerical flux at interfaces for c > 0.
    F_{i+1/2} = c * u_i
    Returns the flux array of size nx, representing fluxes entering each cell from the left.
    """
    nx = len(u)
    fluxes = np.zeros(nx)
    
    for i in range(nx):
        # periodic boundary condition handles left-most interface
        u_left = u[i - 1] if i > 0 else u[-1]
        fluxes[i] = c * u_left
        
    return fluxes

def dg0_step(u, c, dt, dx):
    """
    Perform a single DG0 time step (piecewise-constant, upwind flux).
    """
    nx = len(u)
    fluxes = numerical_flux_upwind(u, c)
    u_new = np.zeros_like(u)
    
    for i in range(nx):
        f_in = fluxes[i]
        f_out = fluxes[(i + 1) % nx]
        # DG0 weak form update: u_new = u_old - (dt/dx) * (f_out - f_in)
        u_new[i] = u[i] - (dt / dx) * (f_out - f_in)
        
    return u_new

def compute_l2_error(u_num, u_exact):
    """
    Compute the L2 norm of the error.
    """
    return np.sqrt(np.mean((u_num - u_exact)**2))

def run_simulation(nx, c, domain_length, final_time, cfl):
    """
    Run the DG0 linear convection simulation for a specific CFL number.
    Returns spatial coordinates, final numerical solution, and statistics.
    """
    dx = domain_length / nx
    x = np.linspace(dx/2, domain_length - dx/2, nx)
    u = initial_condition(x)
    
    dt = cfl * dx / c
    n_steps = int(np.ceil(final_time / dt))
    actual_dt = final_time / n_steps
    
    time = 0.0
    for _ in range(n_steps):
        u = dg0_step(u, c, actual_dt, dx)
        time += actual_dt
        
    u_exact = analytic_solution(x, time, c, domain_length)
    l2_err = compute_l2_error(u, u_exact)
    u_min = np.min(u)
    u_max = np.max(u)
    is_stable = (cfl <= 1.0 and u_max < 2.0 and not np.isnan(u_max))
    
    stats = {
        "cfl": cfl,
        "steps": n_steps,
        "dt": actual_dt,
        "final_time": time,
        "l2_error": l2_err,
        "min": u_min,
        "max": u_max,
        "stable": is_stable
    }
    
    return x, u, u_exact, stats

def plot_cfl_sweep(results, output_file):
    """
    Plot the results of the CFL sweep and save to output_file.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {0.8: 'blue', 1.0: 'green', 1.2: 'red'}
    
    for cfl, (x, u, u_exact) in results.items():
        if cfl == 1.2:
            # Clip unstable values for readability only
            u_plot = np.clip(u, -2.0, 2.0)
            plt.plot(x, u_plot, label=f"DG0 CFL = {cfl}", color=colors.get(cfl, 'red'))
        else:
            plt.plot(x, u, label=f"DG0 CFL = {cfl}", color=colors.get(cfl, 'black'))
            
    # Plot analytic solution from the CFL=1.0 result
    x_exact = results[1.0][0]
    u_exact = results[1.0][2]
    plt.plot(x_exact, u_exact, 'kx', label="Analytic Solution", markevery=2)
    
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('1-D Linear Convection: DG0 (Piecewise-Constant)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    nx = 100
    c = 1.0
    domain_length = 1.0
    final_time = 0.96
    
    cfl_values = [0.8, 1.0, 1.2]
    results = {}
    
    print(f"{'CFL':<5} | {'Steps':<6} | {'dt':<8} | {'Time':<6} | {'L2 Error':<10} | {'Min(u)':<10} | {'Max(u)':<10} | {'Stable'}")
    print("-" * 80)
    
    for cfl in cfl_values:
        x, u, u_exact, stats = run_simulation(nx, c, domain_length, final_time, cfl)
        results[cfl] = (x, u, u_exact)
        
        stable_str = "Yes" if stats["stable"] else "No"
        print(f"{stats['cfl']:<5.1f} | {stats['steps']:<6d} | {stats['dt']:<8.4f} | {stats['final_time']:<6.2f} | {stats['l2_error']:<10.2e} | {stats['min']:<10.2f} | {stats['max']:<10.2f} | {stable_str}")
        
    output_path = os.path.join(os.path.dirname(__file__), "output", "linear_convection_dg0_cfl_sweep.png")
    plot_cfl_sweep(results, output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
