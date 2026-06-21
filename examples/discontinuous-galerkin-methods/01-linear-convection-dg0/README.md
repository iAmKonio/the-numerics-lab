# 1-D Linear Convection with DG0

## What this example solves

This example demonstrates how to solve the 1-D Linear Convection equation using the Discontinuous Galerkin (DG) method with polynomial degree $p = 0$, also known as DG0. The equation is:

$$ \frac{\partial u}{\partial t} + \frac{\partial (c u)}{\partial x} = 0 $$

where $u(x,t)$ is the transported quantity, and $c$ is the constant wave speed.

## Why start DG with linear convection?

Linear convection is the simplest hyperbolic partial differential equation. It describes pure transport without diffusion or source terms. Because the analytic solution simply shifts the initial condition at speed $c$, it isolates the numerical transport properties (like numerical diffusion and dispersion) of the spatial discretization method.

## DG idea in one dimension

The Discontinuous Galerkin method combines the local, element-based nature of Finite Volume Methods (FVM) with the higher-order basis functions of Finite Element Methods (FEM). In 1-D, we:
1. Partition the domain into non-overlapping elements (or cells).
2. Approximate the solution within each element using a local polynomial space.
3. Allow the solution to be discontinuous at the boundaries (interfaces) between elements.
4. Couple the elements together by evaluating numerical fluxes at these interfaces.

## Elements and piecewise-constant approximation

DG0 is the lowest-order DG method. In DG0, the local polynomial degree is $p=0$. This means the solution is approximated as a constant value inside each element. 

$$ u_h|_{K_i} = u_i $$

where $K_i$ is the $i$-th element, and $u_i$ is the constant value within that element. Because there are no continuity restrictions between elements, the solution can jump discontinuously at the interfaces.

## Weak form and flux balance

To derive the DG method, we multiply the PDE by a test function and integrate over an element $K_i$. For DG0, the test function is just a constant (1).

$$ \int_{K_i} \left( \frac{\partial u}{\partial t} + \frac{\partial (c u)}{\partial x} \right) dx = 0 $$

Integrating the spatial derivative gives:

$$ \Delta x \frac{d u_i}{dt} + \left[ c u \right]_{x_{i-1/2}}^{x_{i+1/2}} = 0 $$

Because the solution is discontinuous at the interfaces $x_{i-1/2}$ and $x_{i+1/2}$, the term $cu$ is multi-valued. We resolve this by replacing the physical flux with a single-valued **numerical flux** $F$.

## Upwind numerical flux

For a constant positive wave speed ($c > 0$), information flows from left to right. The upwind numerical flux takes the value from the "upwind" side of the interface:

$$ F_{i+1/2} = c u_i $$

## DG0 update

Substituting the upwind flux into the flux balance and using forward Euler time-stepping gives the explicit DG0 update:

$$ u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x} \left( F_{i+1/2} - F_{i-1/2} \right) $$

## CFL condition

Explicit time-stepping requires a stability limit on the time step $\Delta t$, governed by the Courant-Friedrichs-Lewy (CFL) number:

$$ \text{CFL} = \frac{c \Delta t}{\Delta x} $$

For this specific DG0 formulation and forward Euler time integration, stability requires $\text{CFL} \le 1$.

## Step-by-step implementation

1. **Domain Setup**: Define a 1-D periodic domain and discretize it into elements of size $\Delta x$.
2. **Initial Condition**: Assign cell-average (constant) values to each element based on a smooth Gaussian pulse.
3. **Flux Computation**: Loop over elements and compute the upwind numerical flux at each interface.
4. **Time Integration**: Update the constant value in each element using the flux balance equation.
5. **Evaluation**: Compare the numerical solution to the analytic shifted Gaussian pulse.

## Running the example

Run the script directly using Python:

```bash
python3 linear_convection_dg0.py
```

## Expected result

The script simulates linear convection for three different CFL numbers: 0.8, 1.0, and 1.2. It prints a table with L2 error and stability metrics to the console and generates a plot saved in the `output/` directory:

`output/linear_convection_dg0_cfl_sweep.png`

## What the CFL sweep teaches

- **CFL = 0.8**: The scheme is stable but introduces significant numerical diffusion, heavily smearing the initial Gaussian pulse.
- **CFL = 1.0**: The scheme is stable and, for this exact setup on a uniform mesh, transports the profile perfectly without diffusion (a special artifact of shifting exactly one element per time step).
- **CFL = 1.2**: The stability limit is violated. The solution exhibits rapid, exponentially growing oscillations. The plotted values for this case are clipped to $[-2, 2]$ to maintain visual readability, but the console output shows massive numerical divergence.

## Relationship to FVM and FDM

For 1-D linear convection with constant positive speed on a uniform mesh, DG0 with upwind flux gives the same algebraic update as first-order upwind FVM. Furthermore, on this simple grid, it matches the first-order upwind FDM formula.

The purpose of this example is to introduce DG language and structure before moving to higher-order DG. We conceptualize the domain as elements, write a weak formulation, and compute interface fluxes—concepts that seamlessly extend to arbitrary higher-order polynomials and complex geometries.

## What comes next: DG1, higher-order DG, DGSEM

In upcoming examples, we will increase the polynomial degree to $p=1$ (DG1) and beyond. This will require local basis functions (like Legendre polynomials), volume integrals that no longer vanish, and more sophisticated numerical integration and time-stepping schemes (such as Runge-Kutta). Eventually, this builds up to the Discontinuous Galerkin Spectral Element Method (DGSEM).
