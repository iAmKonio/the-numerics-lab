# 1-D Linear Convection with DG1

## What this example solves

This example solves the 1-D linear convection equation:

$$ \frac{\partial u}{\partial t} + \frac{\partial (cu)}{\partial x} = 0 $$

using the Discontinuous Galerkin method with linear polynomials (DG1). It assumes a constant, positive wave speed ($c > 0$) and uses periodic boundary conditions.

## Why move from DG0 to DG1?

DG1 is the next step after DG0. While DG0 only stores one constant value per element, DG1 stores a mean and a slope per element. This means the solution can vary linearly inside each element. By moving to higher-order polynomials, we can capture the solution with significantly less numerical diffusion. Interface fluxes still couple neighboring elements just as they did in DG0, but now the internal representation is much richer. 

## Elements and local coordinates

We partition our 1-D domain into elements of uniform size $\Delta x$. For each element, we define a local coordinate $\xi \in [-1, 1]$, where $\xi = -1$ is the left boundary, $\xi = 1$ is the right boundary, and $\xi = 0$ is the element center. 

## Linear polynomial approximation

Inside each element, the solution is approximated by a linear polynomial:

$$ u_h(\xi, t) = a_i(t) + b_i(t) \xi $$

Where $a_i(t)$ represents the cell average (mean mode) and $b_i(t)$ represents the slope (slope mode).

## Modal basis functions

This approximation uses a simple modal basis:

- $\phi_0(\xi) = 1$
- $\phi_1(\xi) = \xi$

Because these basis functions are orthogonal on $[-1, 1]$, the equations for $\dot{a}_i$ and $\dot{b}_i$ decouple cleanly in the mass matrix, simplifying the resulting system.

## Projection of the initial condition

To start the simulation, we must map our initial analytical condition onto the DG1 basis. We use a 3-point Gaussian quadrature to numerically integrate the initial condition over each element and compute the initial values for $a_i$ and $b_i$.

## Weak form and interface fluxes

Just like in DG0, we multiply the governing equation by our test functions ($\phi_0$ and $\phi_1$) and integrate over the element. Integrating the spatial derivative by parts moves the derivative onto the test function and yields boundary terms. These boundary terms are evaluated using a numerical flux, which couples the adjacent, discontinuous elements together.

## Upwind numerical flux

Because information travels strictly from left to right ($c > 0$), we use the upwind numerical flux. The state at an interface $x_{i+1/2}$ is determined entirely by the value of the polynomial approaching the interface from the left (i.e., from inside element $i$ at $\xi = 1$):

$$ F_{i+1/2} = c u_i(\xi=1) = c(a_i + b_i) $$

## Time integration with SSP RK3

Because we now have a linear spatial variation inside the element, the maximum stable time step drops relative to DG0. Simple forward Euler time stepping would be severely restricted and potentially unstable. Instead, we use the explicit 3-stage Strong Stability Preserving Runge-Kutta method (SSP RK3).

## CFL condition for DG1

The time step is constrained by the Courant-Friedrichs-Lewy (CFL) number:

$$ \text{CFL} = \frac{c \Delta t}{\Delta x} $$

DG1 has a stricter practical stability limit than the first-order DG0/FVM update. Higher-order polynomial content requires more cautious time stepping. For our specific formulation and time integrator, $\text{CFL} \le 1/3$ is the theoretical stable limit under standard definitions.

## Step-by-step implementation

1. **Domain Setup**: Create a periodic domain and discretize it into elements.
2. **Initial Condition**: Evaluate the smooth Gaussian pulse and project it onto $a_i$ and $b_i$ using Gaussian quadrature.
3. **Flux Computation**: Compute the upwind numerical fluxes using the right-side values of each element.
4. **RHS Evaluation**: Compute the time derivatives $\dot{a}_i$ and $\dot{b}_i$ using the weak form equations.
5. **Time Integration**: Step the solution forward using SSP RK3.
6. **Evaluation**: Reconstruct the linear polynomial on a fine grid for visualization and compare it against the analytical solution.

## Running the example

Run the script directly using Python:

```bash
python3 linear_convection_dg1.py
```

## Expected result

The script simulates linear convection for different CFL numbers (e.g., 0.2, 0.4, and an intentionally unstable 0.8). It prints a table with L2 error and stability metrics to the console and generates a plot saved in the `output/` directory:

`output/linear_convection_dg1_cfl_sweep.png`

## Relationship to DG0, FVM, and FDM

- **FDM** stores point values and approximates derivatives.
- **FVM** stores cell averages and updates them through fluxes.
- **DG0** stores one constant coefficient per element and is equivalent to first-order FVM here.
- **DG1** stores two coefficients per element: a mean and a slope. It begins to show why DG can become high-order and less diffusive. Interface fluxes still couple neighboring elements, but the internal representation tracks gradients explicitly.

This example is educational, not a production solver.

## What comes next: higher-order DG and DGSEM

The jump from DG0 to DG1 demonstrates the core mechanics of Discontinuous Galerkin. To achieve spectral-like accuracy, the next evolution is the Discontinuous Galerkin Spectral Element Method (DGSEM), which employs arbitrarily high-order nodal polynomials, Legendre-Gauss-Lobatto (LGL) quadrature nodes, and collocation to build a highly efficient and accurate solver.
