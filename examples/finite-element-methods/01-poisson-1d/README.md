# 1-D Poisson Equation with the Finite Element Method

This folder contains a foundational Python script demonstrating the Finite Element Method (FEM) applied to the 1-D Poisson equation using linear Lagrange elements.

## What this example solves

This example numerically solves the 1-D Poisson equation over the domain $x \in [0, 1]$ with a specified source term and homogeneous Dirichlet boundary conditions. The Poisson equation is the fundamental model for diffusion, steady-state heat conduction, and electrostatic potential.

## Why Poisson is a good first FEM example

While previous Numerics Lab examples focused on the 1-D linear convection equation, the Poisson equation is the ideal starting point for the Finite Element Method. Convection problems are mathematically hyperbolic and challenging for standard Galerkin FEM without special stabilization techniques (like streamline-upwind Petrov-Galerkin or Discontinuous Galerkin). 

The Poisson equation is an elliptic boundary-value problem. It is mathematically symmetric, positive-definite, and perfectly suited for the standard Galerkin formulation. This makes it an excellent vehicle to introduce the core mechanics of FEM: the weak form, basis functions, element integration, and matrix assembly.

## Strong form

The strong form of the problem is the classic differential equation:

$$
-u''(x) = f(x) \quad \text{for} \quad x \in (0, 1)
$$

## Boundary conditions

We apply homogeneous Dirichlet boundary conditions, meaning the solution is fixed to zero at both ends of the domain:

$$
u(0) = 0, \quad u(1) = 0
$$

## Manufactured analytic solution

To verify our numerical implementation, we use the method of manufactured solutions. We propose an exact solution:

$$
u_{\text{exact}}(x) = \sin(\pi x)
$$

This perfectly satisfies our boundary conditions. Plugging it into the left side of the strong form gives us the required source term:

$$
f(x) = \pi^2 \sin(\pi x)
$$

## From strong form to weak form

A crucial distinction of FEM is that it starts from a **weak form** rather than directly approximating derivatives (like Finite Differences) or flux balances (like Finite Volumes).

We multiply the strong form by an arbitrary test function $v(x)$ (which satisfies $v(0)=v(1)=0$) and integrate over the domain:

$$
\int_0^1 -u''(x)v(x) \, dx = \int_0^1 f(x)v(x) \, dx
$$

Using integration by parts on the left side, we shift one spatial derivative from $u$ onto the test function $v$. The boundary terms vanish due to our homogeneous boundary conditions, resulting in the weak form:

$$
\int_0^1 u'(x)v'(x) \, dx = \int_0^1 f(x)v(x) \, dx
$$

We now seek a function $u_h$ in our finite element space such that this integral equality holds for all test functions $v_h$ in that same space.

## Linear Lagrange basis functions

We partition the domain into a uniform mesh of elements. Over each element, we approximate the solution as a **piecewise-linear function** using linear Lagrange basis functions. In a standard local coordinate system $\xi \in [-1, 1]$, these functions are:

- $\phi_0(\xi) = \frac{1 - \xi}{2}$
- $\phi_1(\xi) = \frac{1 + \xi}{2}$

The numerical solution is constructed by attaching these local "hat" functions to the nodes of our mesh.

## Element stiffness matrix

The stiffness matrix comes from integrating the derivatives of these basis functions over the element. For a 1-D element of length $h$, the element stiffness matrix is given by:

$$
K_e = \frac{1}{h} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

The element load vector $F_e$ is computed by integrating the source term $f(x)$ multiplied by the basis functions, which we approximate using two-point Gauss-Legendre quadrature.

## Assembly into the global system

A core mechanic of FEM is **assembly**. We loop over all individual elements and add their local stiffness matrices $K_e$ and load vectors $F_e$ into a large global stiffness matrix $K$ and global load vector $F$. The overlap of basis functions at shared nodes naturally connects the elements together.

## Applying Dirichlet boundary conditions

Once assembled, the global system is singular. We must enforce our boundary conditions. Homogeneous Dirichlet boundary conditions are enforced by fixing the values at the end nodes to zero. In this script, we achieve this by simply removing the rows and columns corresponding to the boundary nodes, solving a reduced system strictly for the interior unknowns.

## Solving the linear system

After applying the boundary conditions, we are left with a symmetric, positive-definite linear system:

$$
K_{\text{interior}} U_{\text{interior}} = F_{\text{interior}}
$$

We solve this using standard dense linear algebra routines (`numpy.linalg.solve`). *Note: This is an educational foundation example. A production solver would leverage the matrix's sparsity and use iterative solvers or sparse direct methods.*

## Error measurement

To quantify the accuracy, the script evaluates the $L_2$ error and maximum error by densely sampling the linear polynomial solution inside each element and comparing it to the analytic continuous sine wave.

## Convergence study

The script automatically runs a convergence study using meshes of 10, 20, 40, and 80 elements. By observing how the error drops as the mesh size $h$ decreases, we confirm that linear elements achieve optimal $\mathcal{O}(h^2)$ convergence in the $L_2$ norm for the Poisson equation.

## Running the example

You can run the script directly from your terminal. It requires only standard Python data science libraries:

```bash
python3 poisson_1d_fem.py
```

## Expected result

The script will output a table demonstrating the convergence:

```text
Elements   | h          | L2 Error     | Max Error   
----------------------------------------------------
10         | 0.1000     | 8.16e-03     | 1.30e-02
20         | 0.0500     | 2.05e-03     | 3.25e-03
40         | 0.0250     | 5.14e-04     | 8.13e-04
80         | 0.0125     | 1.28e-04     | 2.03e-04
```

It will also save two plots in the local `output/` directory (which is ignored by Git):
1. `poisson_1d_fem_solution.png`: Overlays the discrete linear piecewise FEM solution against the exact continuous sine wave.
2. `poisson_1d_fem_convergence.png`: A log-log plot demonstrating the $\mathcal{O}(h^2)$ error convergence.

## Relationship to the previous Numerics Lab examples

- **FDM (Finite Differences)** discretizes the strong form directly, evaluating derivatives at grid points.
- **FVM (Finite Volumes)** discretizes the integral conservation law, focusing on fluxes crossing cell faces.
- **DG (Discontinuous Galerkin)** uses a weak formulation on independent elements, coupled by numerical interface fluxes.
- **FEM (Continuous Galerkin)** uses a global weak formulation where elements are strongly coupled by shared nodes, requiring a global matrix solve.

## What comes next: FEM for diffusion, advection-diffusion, and stabilized convection

With the weak form, element integration, matrix assembly, and boundary conditions established via this Poisson foundation, we are ready to tackle more complex physics. The natural next steps include adding transient diffusion, introducing a velocity field for advection-diffusion, and eventually utilizing specialized stabilization to handle purely convective problems within the continuous FEM framework.
