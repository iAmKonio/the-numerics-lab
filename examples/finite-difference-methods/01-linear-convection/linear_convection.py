import numpy as np
import matplotlib.pyplot as plt

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

def run_simulation(nx, c, cfl, final_time, domain_length=1.0):
    """
    Runs the linear convection simulation.
    """
    # Spatial grid
    x = np.linspace(0, domain_length, nx, endpoint=False)
    dx = domain_length / nx
    
    # Time step from CFL condition: sigma = c * dt / dx
    dt = cfl * dx / c
    
    # Initial condition
    u = initial_condition(x)
    
    t = 0.0
    while t < final_time:
        # Adjust last time step to hit exactly final_time if needed
        if t + dt > final_time:
            dt = final_time - t
            sigma = c * dt / dx
        else:
            sigma = cfl
            
        u = upwind_step(u, sigma)
        t += dt
        
    return x, u

def main():
    # Parameters
    nx = 100            # Number of cells
    c = 1.0             # Wave speed
    cfl = 0.8           # CFL number (sigma)
    final_time = 1.0    # One full period for length=1.0, c=1.0
    domain_length = 1.0
    
    # Run simulation
    x, u_num = run_simulation(nx, c, cfl, final_time, domain_length)
    
    # Compute exact solution
    u_exact = analytic_solution(x, final_time, c, domain_length)
    
    # Compute error
    error = compute_l2_error(u_num, u_exact)
    
    # Print results
    print(f"--- 1-D Linear Convection (Explicit Upwind FDM) ---")
    print(f"Number of cells: {nx}")
    print(f"CFL number:      {cfl}")
    print(f"Final time:      {final_time:.2f}")
    print(f"L2 Error:        {error:.4e}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, initial_condition(x), '--', color='gray', label='Initial Condition')
    plt.plot(x, u_exact, '-k', label='Analytic Solution')
    plt.plot(x, u_num, 'o-r', markerfacecolor='none', label='Upwind FDM')
    plt.title(f'1-D Linear Convection (CFL = {cfl})')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/linear_convection_comparison.png')
    print("Plot saved to output/linear_convection_comparison.png")

if __name__ == '__main__':
    main()
