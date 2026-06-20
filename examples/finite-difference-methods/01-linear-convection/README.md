# 1-D Linear Convection with the Finite Difference Method

## What this example solves
This example implements a numerical solver for the 1-D Linear Convection equation using an explicit first-order upwind Finite Difference Method (FDM). It models a scalar profile being transported at a constant speed across a periodic domain.

## Why linear convection matters
The linear convection (or advection) equation is a fundamental prototype in computational fluid dynamics (CFD). It is often the first transport equation studied because it introduces:
- Numerical transport (advection) without diffusion.
- The concept of the Courant-Friedrichs-Lewy (CFL) stability condition.
- Numerical errors such as numerical diffusion (smearing) and dispersion (oscillations).

Mastering this simple equation is essential before tackling more complex, non-linear problems like the Navier-Stokes equations.

## Governing equation
The 1-D continuous linear convection equation is:

$$ \frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0 $$

where:
- $u$ is the scalar field (e.g., concentration, temperature).
- $c$ is the constant wave propagation speed ($c > 0$).
- $x$ is the spatial coordinate.
- $t$ is time.

## Analytic solution
For an initial condition $u(x, 0) = u_0(x)$ on an infinite or periodic domain, the exact analytic solution is simply the initial profile translated by a distance $ct$:

$$ u(x,t) = u_0(x - c t) $$

In this example, we use a smooth Gaussian pulse and periodic boundary conditions, meaning the pulse wraps around the domain.

## Discretization
We discretize the domain into uniform cells of size $\Delta x$, so $x_i = i \Delta x$. Time is discretized into levels $t^n = n \Delta t$.

Using a forward difference in time and a backward difference in space (upwind, since $c > 0$), we get the explicit update scheme:

$$ \frac{u_i^{n+1} - u_i^n}{\Delta t} + c \frac{u_i^n - u_{i-1}^n}{\Delta x} = 0 $$

Rearranging for the new time level $u_i^{n+1}$:

$$ u_i^{n+1} = u_i^n - \sigma (u_i^n - u_{i-1}^n) $$

where $\sigma$ is the Courant number.

## CFL condition
The Courant number (CFL) is defined as:

$$ \sigma = \frac{c \Delta t}{\Delta x} $$

For the first-order upwind explicit scheme to remain stable, the CFL condition requires:

$$ 0 \le \sigma \le 1 $$

If $\sigma > 1$, the numerical scheme will become unstable and diverge. If $\sigma$ is small, the scheme is stable but introduces artificial numerical diffusion, smearing the pulse. For $\sigma = 1$, the exact analytic shift is recovered without numerical diffusion, provided the grid aligns perfectly.

## Step-by-step implementation
1. **Setup parameters**: Choose the domain size, number of cells (`nx`), wave speed (`c`), and final time.
2. **Initial Condition**: Initialize the grid with a Gaussian pulse.
3. **Time step**: Compute $\Delta t$ based on a chosen stable CFL number.
4. **Time Marching**: Iterate over time, applying the upwind update formula to compute the next time step.
5. **Periodic Boundary**: Use array rolling to enforce $u_0^n - u_{-1}^n = u_0^n - u_{nx-1}^n$.
6. **Compare**: At the final time, compute the analytic solution and compare the numerical result.

## Running the example

Ensure you have `numpy` and `matplotlib` installed.

```bash
# From this directory:
python3 linear_convection.py
```

## Expected result
The script will output a table comparing the simulation parameters, L2 error, and stability for three different CFL numbers (0.8, 1.0, and 1.2). It will also generate a plot saved in the `output/` directory that visualizes these three runs alongside the analytic solution.

```
--- 1-D Linear Convection (Explicit Upwind FDM) ---
CFL   | Steps | dt       | Final t | L2 Error   | Stability
-----------------------------------------------------------------
0.8   | 62    | 0.0080   | 0.4960  | 2.7124e-02 | Stable
1.0   | 50    | 0.0100   | 0.5000  | 1.4569e-16 | Stable
1.2   | 42    | 0.0120   | 0.5040  | 3.5186e-02 | Unstable

Plot saved to output/linear_convection_cfl_sweep.png
```

## Numerical behavior and comparison with analytic solution
- **CFL = 0.8 (Stable, Diffusive):** The numerical solution exhibits numerical diffusion due to the first-order truncation error of the upwind scheme. The peak of the pulse will appear shorter and wider.
- **CFL = 1.0 (Exact, "Magic" step):** For the upwind scheme, a CFL of exactly 1.0 results in zero numerical diffusion and dispersion. The numerical solution perfectly matches the analytic solution because the wave travels exactly one grid cell per time step.
- **CFL = 1.2 (Unstable):** The numerical scheme violates the CFL condition ($\sigma \le 1$), leading to instability. Oscillations grow unboundedly, destroying the solution.

## What to try next
- Change the initial condition to a square wave to see severe numerical smearing on discontinuities for stable CFLs.
- Explore a higher-order scheme (like Lax-Wendroff) to see how it handles numerical diffusion and dispersion differently.

## Connection to The Numerics Lab
This is an educational example created as part of The Numerics Lab module.
