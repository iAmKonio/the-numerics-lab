import numpy as np
import matplotlib.pyplot as plt
import os

def exact_solution(x):
    """
    Analytic manufactured solution for the 1-D Poisson equation.
    u(x) = sin(pi * x)
    """
    return np.sin(np.pi * x)

def source_term(x):
    """
    Source term corresponding to the manufactured solution.
    If u(x) = sin(pi * x), then -u''(x) = pi^2 * sin(pi * x).
    """
    return np.pi**2 * np.sin(np.pi * x)

def make_mesh(num_elements, domain_length=1.0):
    """
    Generate a 1-D uniform mesh.
    Returns the node coordinates and the element connectivity.
    """
    num_nodes = num_elements + 1
    nodes = np.linspace(0, domain_length, num_nodes)
    
    # Element connectivity: element i connects node i and i+1
    elements = np.zeros((num_elements, 2), dtype=int)
    for i in range(num_elements):
        elements[i, 0] = i
        elements[i, 1] = i + 1
        
    return nodes, elements

def element_stiffness(h):
    """
    Compute the element stiffness matrix for linear Lagrange elements.
    K_e = (1/h) * [[1, -1],
                   [-1, 1]]
    """
    return (1.0 / h) * np.array([[1.0, -1.0],
                                 [-1.0, 1.0]])

def element_load_vector(node_left, node_right):
    """
    Compute the element load vector using two-point Gauss quadrature.
    """
    h = node_right - node_left
    
    # 2-point Gauss quadrature on [-1, 1]
    xi_q = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    w_q = np.array([1.0, 1.0])
    
    f_e = np.zeros(2)
    
    # Map [-1, 1] to [node_left, node_right]
    # x(xi) = node_left * (1 - xi)/2 + node_right * (1 + xi)/2
    for q in range(2):
        xi = xi_q[q]
        weight = w_q[q]
        
        # Physical coordinate of quadrature point
        x_q = node_left * 0.5 * (1.0 - xi) + node_right * 0.5 * (1.0 + xi)
        
        # Shape functions evaluated at xi
        phi_0 = 0.5 * (1.0 - xi)
        phi_1 = 0.5 * (1.0 + xi)
        
        # Source term at quadrature point
        f_val = source_term(x_q)
        
        # Integration: dx = (h/2) * dxi
        det_J = h / 2.0
        
        f_e[0] += weight * f_val * phi_0 * det_J
        f_e[1] += weight * f_val * phi_1 * det_J
        
    return f_e

def assemble_system(nodes, elements):
    """
    Assemble the global stiffness matrix and load vector.
    """
    num_nodes = len(nodes)
    num_elements = len(elements)
    
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    
    for i in range(num_elements):
        n0 = elements[i, 0]
        n1 = elements[i, 1]
        
        x0 = nodes[n0]
        x1 = nodes[n1]
        h = x1 - x0
        
        # Local matrices
        K_e = element_stiffness(h)
        F_e = element_load_vector(x0, x1)
        
        # Assembly
        K_global[n0, n0] += K_e[0, 0]
        K_global[n0, n1] += K_e[0, 1]
        K_global[n1, n0] += K_e[1, 0]
        K_global[n1, n1] += K_e[1, 1]
        
        F_global[n0] += F_e[0]
        F_global[n1] += F_e[1]
        
    return K_global, F_global

def apply_dirichlet_boundary_conditions(K, F, boundary_nodes):
    """
    Apply homogeneous Dirichlet boundary conditions by eliminating 
    the boundary nodes from the system (solving for interior unknowns).
    """
    num_nodes = len(F)
    interior_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes)
    
    # Extract the interior portion of the matrix and vector
    K_interior = K[np.ix_(interior_nodes, interior_nodes)]
    F_interior = F[interior_nodes]
    
    return K_interior, F_interior, interior_nodes

def solve_poisson_1d(num_elements):
    """
    Solves the 1-D Poisson equation for a given number of elements.
    Returns the mesh nodes and the numerical solution.
    """
    nodes, elements = make_mesh(num_elements)
    
    K_global, F_global = assemble_system(nodes, elements)
    
    # Apply homogeneous Dirichlet BCs at x=0 and x=1
    boundary_nodes = [0, len(nodes) - 1]
    K_interior, F_interior, interior_nodes = apply_dirichlet_boundary_conditions(K_global, F_global, boundary_nodes)
    
    # Solve the linear system K_interior * U_interior = F_interior
    U_interior = np.linalg.solve(K_interior, F_interior)
    
    # Reconstruct the full solution vector
    U_full = np.zeros(len(nodes))
    U_full[interior_nodes] = U_interior
    # Boundary nodes remain 0.0, which matches homogeneous Dirichlet
    
    return nodes, U_full

def compute_errors(nodes, U_num):
    """
    Compute the L2 and max errors of the numerical solution.
    Uses dense sampling within each element to approximate the L2 norm.
    """
    num_elements = len(nodes) - 1
    
    max_err = 0.0
    l2_err_sq = 0.0
    
    # 10 sample points per element for accurate error integration
    samples_per_elem = 10
    xi_sample = np.linspace(-1.0, 1.0, samples_per_elem)
    
    for i in range(num_elements):
        x0 = nodes[i]
        x1 = nodes[i+1]
        u0 = U_num[i]
        u1 = U_num[i+1]
        h = x1 - x0
        
        # Trapezoidal rule for integration over the element
        dx_sample = 2.0 / (samples_per_elem - 1)
        
        for j, xi in enumerate(xi_sample):
            # Physical coordinate
            x_val = x0 * 0.5 * (1.0 - xi) + x1 * 0.5 * (1.0 + xi)
            
            # Exact and numerical solutions at x_val
            u_exact_val = exact_solution(x_val)
            
            # Linear interpolation of numerical solution
            phi_0 = 0.5 * (1.0 - xi)
            phi_1 = 0.5 * (1.0 + xi)
            u_num_val = u0 * phi_0 + u1 * phi_1
            
            err_val = abs(u_num_val - u_exact_val)
            max_err = max(max_err, err_val)
            
            # Integration weight for trapezoidal rule
            weight = 0.5 if (j == 0 or j == samples_per_elem - 1) else 1.0
            
            l2_err_sq += weight * (err_val**2) * dx_sample * (h / 2.0)
            
    l2_err = np.sqrt(l2_err_sq)
    return l2_err, max_err

def run_convergence_study(elements_list):
    """
    Run the FEM solver for a list of element counts and measure errors.
    """
    results = []
    
    print(f"{'Elements':<10} | {'h':<10} | {'L2 Error':<12} | {'Max Error':<12}")
    print("-" * 52)
    
    for n in elements_list:
        nodes, U_num = solve_poisson_1d(n)
        h = 1.0 / n
        l2_err, max_err = compute_errors(nodes, U_num)
        
        print(f"{n:<10d} | {h:<10.4f} | {l2_err:<12.2e} | {max_err:<12.2e}")
        
        results.append({
            'elements': n,
            'h': h,
            'l2_error': l2_err,
            'max_error': max_err
        })
        
    return results

def plot_solution(nodes, U_num, output_file):
    """
    Plot the FEM solution against the exact solution.
    """
    x_exact = np.linspace(0, 1, 200)
    u_exact = exact_solution(x_exact)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, u_exact, 'k-', linewidth=2, label='Analytic Solution')
    plt.plot(nodes, U_num, 'bo--', markersize=6, label=f'FEM (Linear, N={len(nodes)-1})')
    
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('1-D Poisson Equation: Finite Element Method')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def plot_convergence(results, output_file):
    """
    Plot the convergence behavior (Error vs. h) on a log-log scale.
    """
    h_vals = [res['h'] for res in results]
    l2_errs = [res['l2_error'] for res in results]
    max_errs = [res['max_error'] for res in results]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(h_vals, l2_errs, 'bo-', linewidth=2, label='L2 Error')
    plt.loglog(h_vals, max_errs, 'rs-', linewidth=2, label='Max Error')
    
    # Add an O(h^2) reference line
    h_ref = np.array([h_vals[-1], h_vals[0]])
    err_ref = l2_errs[-1] * (h_ref / h_vals[-1])**2
    plt.loglog(h_ref, err_ref, 'k--', label=r'$\mathcal{O}(h^2)$ reference')
    
    plt.xlabel('Mesh size (h)')
    plt.ylabel('Error')
    plt.title('FEM Convergence for 1-D Poisson Equation')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Solve default case
    n_default = 20
    print(f"Solving 1-D Poisson Equation with {n_default} linear elements...\n")
    nodes, U_num = solve_poisson_1d(n_default)
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    sol_plot_path = os.path.join(output_dir, "poisson_1d_fem_solution.png")
    plot_solution(nodes, U_num, sol_plot_path)
    print(f"Solution plot saved to {sol_plot_path}\n")
    
    # Run convergence study
    print("Running convergence study:")
    elements_list = [10, 20, 40, 80]
    results = run_convergence_study(elements_list)
    
    conv_plot_path = os.path.join(output_dir, "poisson_1d_fem_convergence.png")
    plot_convergence(results, conv_plot_path)
    print(f"\nConvergence plot saved to {conv_plot_path}")

if __name__ == "__main__":
    main()
