# 1-D Linear Convection with the Finite Volume Method

## What this example solves
This example numerically solves the 1-D linear convection equation, the simplest model of wave propagation and transport. It is the same equation solved in the Finite Difference Method (FDM) example, but approached from the perspective of Finite Volume Methods (FVM).

## Why use finite volumes for linear convection?
While FDM directly approximates the derivatives of the governing PDE at grid points, FVM is built upon the integral form of the conservation laws. Instead of storing point values, FVM stores cell averages. The values are updated by computing fluxes across the boundaries (faces) of the cells. This guarantees exact conservation of the transported quantity up to machine precision, a critical property for fluid dynamics and heat transfer.

## Conservative form
The 1-D linear convection equation in conservative form is written as:

∂u/∂t + ∂f(u)/∂x = 0

where `u` is the conserved quantity and `f(u)` is the flux. For linear convection with a constant wave speed `c`:

f(u) = c u

## Cell averages and control volumes
In FVM, the domain is divided into control volumes or cells. The variable `u_i` represents the average value of `u` over the `i`-th cell. 

## Numerical flux
To update the cell averages over time, we must compute the amount of `u` flowing into and out of each cell. This flow is called the numerical flux, denoted `F_{i+1/2}` for the right face of cell `i`, and `F_{i-1/2}` for the left face.

## Upwind flux for positive convection speed
The choice of numerical flux determines the stability and accuracy of the scheme. For a strictly positive convection speed (`c > 0`), information only flows from left to right. Therefore, the flux at the face `i+1/2` depends only on the cell to its left, `i`:

F_{i+1/2} = c u_i

This is known as the first-order upwind numerical flux.

## Finite volume update
Integrating the conservative form over a cell and applying the definition of the numerical fluxes yields the explicit FVM update equation:

u_i^{n+1} = u_i^n - (Δt / Δx) (F_{i+1/2} - F_{i-1/2})

## CFL condition
For the explicit first-order upwind finite volume scheme to remain stable, the Courant-Friedrichs-Lewy (CFL) condition must be met:

0 <= CFL <= 1

where CFL = c Δt / Δx. If the CFL number exceeds 1, the numerical scheme attempts to transport fluxes further than one cell per time step, leading to non-physical oscillations and instability.

## Step-by-step implementation
1. **Initialize the grid**: Divide the domain into uniform cells of width `Δx`.
2. **Initial Condition**: Set the initial cell averages `u_i^0`.
3. **Compute time step**: Choose `Δt` to satisfy the desired CFL number.
4. **Compute fluxes**: Calculate the numerical flux `F` at all cell faces using the upwind method.
5. **Update cells**: Apply the finite volume update formula to find the new cell averages `u_i^{n+1}`.
6. **Enforce boundaries**: Apply periodic boundary conditions to the boundaries of the domain.
7. **Repeat**: March forward in time until the final time is reached.

## Running the example
To run the Python script, ensure you have `numpy` and `matplotlib` installed:

```bash
python3 linear_convection_fvm.py
```

## Expected result
The script will output a table comparing the simulation metrics for three different CFL numbers (0.8, 1.0, and 1.2) against the analytic solution at `t = 0.96`. It will also generate a plot saved in `output/linear_convection_fvm_cfl_sweep.png`.

## What the CFL sweep teaches
- **CFL = 0.8**: The scheme is stable but the numerical pulse is smeared due to numerical diffusion introduced by the first-order upwind flux.
- **CFL = 1.0**: The scheme perfectly shifts the cell averages by exactly one cell width per time step, resulting in zero numerical diffusion. This is a special property of this specific setup and not a general rule.
- **CFL = 1.2**: The CFL condition is violated, causing the numerical fluxes to overshoot and the solution to become wildly unstable.

## Relationship to the FDM example
For a constant positive wave speed `c` on a uniform grid, the first-order upwind FVM scheme mathematically reduces to the exact same algebraic update as the first-order upwind FDM scheme. However, the conceptual difference is profound:
- **FDM** is an approximation of mathematical derivatives at discrete points.
- **FVM** is a physical balance of fluxes across finite control volumes, explicitly enforcing conservation.

## What to try next
- Change the initial condition to a step function (square wave) to see how numerical diffusion affects sharp gradients.
- Implement a higher-order numerical flux, such as Lax-Wendroff, to reduce numerical diffusion.
- Test a negative wave speed (`c < 0`) to see how the upwind flux formulation must change to remain stable.
