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

def project_initial_condition(nx, domain_length):
    """
    Project the initial condition onto the DG1 basis using 3-point Gaussian quadrature.
    Returns the arrays a and b representing the mean and slope in each element.
    """
    dx = domain_length / nx
    x_centers = np.linspace(dx/2, domain_length - dx/2, nx)
    
    # 3-point Gauss-Legendre quadrature points and weights on [-1, 1]
    xi_q = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    w_q = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    
    a = np.zeros(nx)
    b = np.zeros(nx)
    
    for i in range(nx):
        xc = x_centers[i]
        
        # Compute integral for a_i (projection onto phi_0 = 1)
        # a_i = 1/2 * int_{-1}^{1} u0(xc + xi * dx/2) * 1 dxi
        sum_a = 0.0
        # Compute integral for b_i (projection onto phi_1 = xi)
        # b_i = 3/2 * int_{-1}^{1} u0(xc + xi * dx/2) * xi dxi
        sum_b = 0.0
        
        for k in range(3):
            x_eval = xc + xi_q[k] * dx / 2.0
            u_val = initial_condition(x_eval)
            sum_a += w_q[k] * u_val
            sum_b += w_q[k] * u_val * xi_q[k]
            
        a[i] = 0.5 * sum_a
        b[i] = 1.5 * sum_b
        
    return a, b

def compute_rhs_dg1(a, b, c, dx):
    """
    Compute the right-hand side (spatial operator) for the DG1 semi-discrete form.
    Returns da/dt and db/dt.
    """
    nx = len(a)
    da_dt = np.zeros(nx)
    db_dt = np.zeros(nx)
    
    # Evaluate solution at element boundaries (xi = 1 for right, xi = -1 for left)
    # The left boundary of cell i is the right boundary of cell i-1.
    # Because c > 0, the upwind numerical flux is always the state from the left side.
    
    # State at the right interface of each cell (xi = 1)
    u_right = a + b
    
    for i in range(nx):
        # Periodic boundary condition handles the left-most interface
        im1 = i - 1 if i > 0 else nx - 1
        
        # Upwind fluxes
        f_in = c * u_right[im1]
        f_out = c * u_right[i]
        
        # Update equations derived from the weak form
        da_dt[i] = -(1.0 / dx) * (f_out - f_in)
        db_dt[i] = (3.0 / dx) * (2.0 * c * a[i] - (f_out + f_in))
        
    return da_dt, db_dt

def ssp_rk3_step(a, b, c, dt, dx):
    """
    Perform a single SSP RK3 time step.
    """
    # Stage 1
    da_dt1, db_dt1 = compute_rhs_dg1(a, b, c, dx)
    a1 = a + dt * da_dt1
    b1 = b + dt * db_dt1
    
    # Stage 2
    da_dt2, db_dt2 = compute_rhs_dg1(a1, b1, c, dx)
    a2 = 0.75 * a + 0.25 * a1 + 0.25 * dt * da_dt2
    b2 = 0.75 * b + 0.25 * b1 + 0.25 * dt * db_dt2
    
    # Stage 3
    da_dt3, db_dt3 = compute_rhs_dg1(a2, b2, c, dx)
    a_new = (1.0 / 3.0) * a + (2.0 / 3.0) * a2 + (2.0 / 3.0) * dt * da_dt3
    b_new = (1.0 / 3.0) * b + (2.0 / 3.0) * b2 + (2.0 / 3.0) * dt * db_dt3
    
    return a_new, b_new

def evaluate_solution(a, b, nx, domain_length, points_per_element=3):
    """
    Evaluate the DG1 polynomial on a fine grid for plotting and error computation.
    """
    dx = domain_length / nx
    x_centers = np.linspace(dx/2, domain_length - dx/2, nx)
    
    xi_eval = np.linspace(-1, 1, points_per_element)
    
    x_fine = np.zeros(nx * points_per_element)
    u_fine = np.zeros(nx * points_per_element)
    
    idx = 0
    for i in range(nx):
        for xi in xi_eval:
            x_fine[idx] = x_centers[i] + xi * dx / 2.0
            u_fine[idx] = a[i] + b[i] * xi
            idx += 1
            
    return x_fine, u_fine

def compute_l2_error(u_num, u_exact):
    """
    Compute the L2 norm of the error on the fine grid.
    """
    return np.sqrt(np.mean((u_num - u_exact)**2))

def run_simulation(nx, c, domain_length, final_time, cfl):
    """
    Run the DG1 linear convection simulation for a specific CFL number.
    Returns spatial coordinates, final numerical solution, and statistics.
    """
    dx = domain_length / nx
    
    # For stability with SSP-RK3 and DG1, the theoretical maximum CFL is 1/3 (0.333)
    # when using the standard definitions. However, the exact limit depends on the
    # definition of CFL. Here we use CFL = c * dt / dx.
    # Note: DG1 is more restrictive than DG0.
    
    dt = cfl * dx / c
    n_steps = int(np.ceil(final_time / dt))
    actual_dt = final_time / n_steps
    
    a, b = project_initial_condition(nx, domain_length)
    
    time = 0.0
    for _ in range(n_steps):
        a, b = ssp_rk3_step(a, b, c, actual_dt, dx)
        time += actual_dt
        
    x_fine, u_fine = evaluate_solution(a, b, nx, domain_length, points_per_element=5)
    u_exact = analytic_solution(x_fine, time, c, domain_length)
    
    l2_err = compute_l2_error(u_fine, u_exact)
    u_min = np.min(u_fine)
    u_max = np.max(u_fine)
    is_stable = (u_max < 2.0 and not np.isnan(u_max))
    
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
    
    return x_fine, u_fine, u_exact, stats

def plot_cfl_sweep(results, output_file):
    """
    Plot the results of the CFL sweep and save to output_file.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {0.1: 'cyan', 0.2: 'blue', 0.3: 'green', 0.4: 'orange', 0.8: 'red'}
    
    for cfl, (x, u, u_exact) in results.items():
        if cfl >= 0.4:
            # Clip unstable values for readability only
            u_plot = np.clip(u, -2.0, 2.0)
            plt.plot(x, u_plot, label=f"DG1 CFL = {cfl}", color=colors.get(cfl, 'red'))
        else:
            plt.plot(x, u, label=f"DG1 CFL = {cfl}", color=colors.get(cfl, 'black'))
            
    # Plot analytic solution from the lowest CFL result
    cfl_ref = list(results.keys())[0]
    x_exact = results[cfl_ref][0]
    u_exact = results[cfl_ref][2]
    plt.plot(x_exact, u_exact, 'kx', label="Analytic Solution", markevery=5)
    
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('1-D Linear Convection: DG1 (Linear Polynomial)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    nx = 60
    c = 1.0
    domain_length = 1.0
    final_time = 0.8  # Slightly smaller to ensure nicely aligned output for demonstration
    
    cfl_values = [0.2, 0.4, 0.8]
    results = {}
    
    print(f"{'CFL':<5} | {'Steps':<6} | {'dt':<8} | {'Time':<6} | {'L2 Error':<10} | {'Min(u)':<10} | {'Max(u)':<10} | {'Stable'}")
    print("-" * 80)
    
    for cfl in cfl_values:
        x, u, u_exact, stats = run_simulation(nx, c, domain_length, final_time, cfl)
        results[cfl] = (x, u, u_exact)
        
        stable_str = "Yes" if stats["stable"] else "No"
        print(f"{stats['cfl']:<5.1f} | {stats['steps']:<6d} | {stats['dt']:<8.4f} | {stats['final_time']:<6.2f} | {stats['l2_error']:<10.2e} | {stats['min']:<10.2f} | {stats['max']:<10.2f} | {stable_str}")
        
    output_path = os.path.join(os.path.dirname(__file__), "output", "linear_convection_dg1_cfl_sweep.png")
    plot_cfl_sweep(results, output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
