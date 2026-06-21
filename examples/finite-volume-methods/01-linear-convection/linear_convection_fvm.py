import os
import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    """Smooth Gaussian pulse."""
    return np.exp(-100.0 * (x - 0.5)**2)

def analytic_solution(x, t, c, domain_length):
    """Analytic solution is the initial condition shifted by c*t."""
    x_shifted = (x - c * t) % domain_length
    return initial_condition(x_shifted)

def compute_flux(u, c):
    """Compute first-order upwind numerical flux for c > 0."""
    return c * u

def finite_volume_step(u, c, dt, dx):
    """
    Perform one explicit upwind finite volume step.
    Updates cell averages based on fluxes through cell faces.
    """
    nx = len(u)
    u_new = np.zeros_like(u)
    
    for i in range(nx):
        # With periodic boundaries, the left neighbor of i=0 is i=nx-1
        im1 = i - 1 if i > 0 else nx - 1
        
        # Upwind flux at right face i+1/2 is based on cell i
        F_right = compute_flux(u[i], c)
        
        # Upwind flux at left face i-1/2 is based on cell i-1
        F_left = compute_flux(u[im1], c)
        
        # Finite volume update
        u_new[i] = u[i] - (dt / dx) * (F_right - F_left)
        
    return u_new

def compute_l2_error(u_num, u_exact):
    """Compute the discrete L2 error norm."""
    return np.sqrt(np.mean((u_num - u_exact)**2))

def run_simulation(nx, c, target_cfl, final_time, domain_length):
    """Run the 1D linear convection simulation using FVM."""
    dx = domain_length / nx
    x = np.linspace(dx/2, domain_length - dx/2, nx) # Cell centers
    
    dt = target_cfl * dx / c
    num_steps = int(np.ceil(final_time / dt))
    
    # Recompute actual dt to exactly hit final_time
    dt = final_time / num_steps
    actual_cfl = c * dt / dx
    
    u = initial_condition(x)
    
    for _ in range(num_steps):
        u = finite_volume_step(u, c, dt, dx)
        
    u_exact = analytic_solution(x, final_time, c, domain_length)
    l2_error = compute_l2_error(u, u_exact)
    min_u = np.min(u)
    max_u = np.max(u)
    
    stability = "Stable" if actual_cfl <= 1.0 and max_u < 2.0 else "Unstable"
    
    return {
        'x': x,
        'u': u,
        'u_exact': u_exact,
        'cfl': actual_cfl,
        'steps': num_steps,
        'dt': dt,
        'final_time': num_steps * dt,
        'l2_error': l2_error,
        'min_u': min_u,
        'max_u': max_u,
        'stability': stability
    }

def plot_cfl_sweep(results, output_dir):
    """Plot the results of the CFL sweep in a single figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot numerical results
    colors = {0.8: 'blue', 1.0: 'green', 1.2: 'red'}
    for target_cfl, res in results.items():
        u_plot = np.clip(res['u'], -2, 2) if target_cfl == 1.2 else res['u']
        plt.plot(res['x'], u_plot, label=f"Numerical (CFL = {target_cfl})", 
                 color=colors.get(target_cfl, 'black'), 
                 linewidth=2 if target_cfl != 1.0 else 1.5)
                 
    # Plot analytic solution from one of the results (they all have same exact solution)
    res_ref = next(iter(results.values()))
    plt.plot(res_ref['x'], res_ref['u_exact'], 'kx', label="Analytic Solution", markersize=6)
    
    plt.title("1-D Linear Convection FVM: CFL Sweep", fontsize=14)
    plt.xlabel("Domain x", fontsize=12)
    plt.ylabel("Velocity u", fontsize=12)
    plt.ylim(-2, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    output_path = os.path.join(output_dir, 'linear_convection_fvm_cfl_sweep.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_path}")

def main():
    domain_length = 1.0
    nx = 100
    c = 1.0
    final_time = 0.96
    
    cfl_values = [0.8, 1.0, 1.2]
    results = {}
    
    print("-" * 105)
    print(f"{'CFL':<8} | {'Steps':<8} | {'dt':<10} | {'Final t':<10} | {'L2 Error':<12} | {'min(u)':<10} | {'max(u)':<10} | {'Stability':<10}")
    print("-" * 105)
    
    for cfl in cfl_values:
        res = run_simulation(nx, c, cfl, final_time, domain_length)
        results[cfl] = res
        print(f"{res['cfl']:<8.3f} | {res['steps']:<8d} | {res['dt']:<10.5f} | {res['final_time']:<10.5f} | "
              f"{res['l2_error']:<12.2e} | {res['min_u']:<10.5f} | {res['max_u']:<10.5f} | {res['stability']:<10}")
    print("-" * 105)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    plot_cfl_sweep(results, output_dir)

if __name__ == "__main__":
    main()
