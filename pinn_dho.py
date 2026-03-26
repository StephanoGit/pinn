import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Physics Informed Neural Networks (PINN)

    This file will use `PyTorch` and PINNs to solve simulation and inversion problems related to the damped harmonic oscillator.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt 
    import plotly.graph_objects as go

    import marimo as mo

    return go, mo, nn, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem Overview

    We are going to use a PINN to solve problems related to the **damped harmonic oscillator**:

    ![Oscillator](https://benmoseley.blog/wp-content/uploads/2021/08/oscillator.gif)

    We are interested in modelling the displacement of the mass on a spring (green box) over time

    This is a canonical physics problem, where the displacement, $u(t)$, of the oscillator as a function of time can be described by the following differential equation:

    \[

    m\frac{d^2u}{dt^2} + \mu \frac{du}{dt} + ku = 0,

    \]

    where:
    - $m$ is the mass of the oscillator
    - $\mu$ is the coefficient of friction
    - $k$ is the spring constant


    We will focus on solving the problem in the **under-dumped state**, i.e. where the oscillation is slowly damped by friction (as displayed in the animation above).

    Mathematically, this occurs when:

    \[

    \delta < \omega_0, \text{ where } \delta = \frac{\mu}{2m}, \omega_0 = \sqrt{\frac{k}{m}}

    \]

    where:
    - $\omega_0$ (Natural Angular Frequency) depends on how stiff the spring is ($k$) and how heavy the mass is ($m$). This represents the "perfect" scenario. If there were absolutely zero friction, this is the exact speed or frequency at which the mass would bounce up and down endlessly.
    - $\delta$ (Damping Factor or Decay Rate) tells you how fast the system loses energy due to friction ($\mu$). This represents the stopping power. The higher the friction, the larger this factor becomes, pulling energy out of the system and causing the bounces to shrink rapidly over time.

    ### The "Under-damped" State ($\delta < \omega_0$)

    When the math says $\delta < \omega_0$, it translates to a very specific physical reality: the system's desire to bounce is stronger than friction's ability to stop it.

    Because the natural frequency ($\omega_0$) is larger than the decay rate ($\delta$), the mass manages to swing back and forth across its resting point several times before friction finally kills the movement entirely. You get a classic decaying wave—the mass overshoots the center, comes back, overshoots again by a little less, and eventually settles to a stop.

    ### Tying it Back to the Real-World Frequency ($\omega$)

    Because the system is fighting friction the entire time it bounces, it can't bounce quite as fast as its perfect natural frequency. Instead, it bounces at the Damped Angular Frequency ($\omega$):

    $$ \omega = \sqrt{\omega_0^2 - \delta^2} $$

    This formula elegantly shows the reality of the situation: the actual bouncing speed ($\omega$) is just the perfect bouncing speed ($\omega_0$) visibly slowed down by the friction ($\delta$).


    Furthermore, we are given the following initial conditions:

    $$ u(t = 0) = 1 \text{ , } \frac{du}{dt}(t = 0) = 0 $$

    where:
    - $u(t = 0) = 1$: We pull the mass to a starting position of 1.
    - $\frac{du}{dt}(t = 0) = 0$: We release it from a complete standstill (zero starting velocity).

    ### The Exact Solution:

    $$ u(t) = e^{-\delta t} (2A \cos(\phi + \omega t)) $$

    This equation perfectly describes the movement by multiplying two distinct physical behaviors together:
    1. The Bouncing Part: $(2A \cos(\phi + \omega t))$
        - This is the standard math for a wave. The cosine function ($\cos$) creates the endless up-and-down, back-and-forth movement.
        - $\omega$ is our damped frequency from earlier—it dictates exactly how fast the wave wiggles in reality.
        - $A$ (Amplitude) and $\phi$ (Phase shift) are just constants dictated by our initial conditions. They exist simply to ensure the math curve perfectly matches the fact that we started exactly at position 1 with zero velocity.

    2. The Stopping Part: $e^{-\delta t}$
        - This is an exponential decay function. Think of it as a shrinking envelope.
        - Because time ($t$) keeps growing, and $\delta$ (our friction factor) is positive, this entire $e^{-\delta t}$ term gets smaller and smaller as the clock ticks forward, eventually approaching zero.

    The mass tries to bounce endlessly ($\cos$), but the shrinking exponential term ($e^{-\delta t}$) squishes the height of every single swing until it comes to a complete halt.

    For a more detailed mathematical explanation, check out the blog post: https://beltoforion.de/en/harmonic_oscillator/
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Workflow overview

    There are 2 scientific tasks related to the harmonic oscillator we will use a PINN for:
    1. We will simulate the system using a PINN, given the initial conditions **(forward problem)**
    2. We will invert the underlying paramters of the system using a PINN, given some noisy observations of the oscillator's displacement **(inverse problem)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions and Configuration Variables
    """)
    return


@app.cell
def _(nn):
    DEVICE = "mps"

    N_INPUT = 1
    N_OUTPUT = 1
    N_HIDDEN = 32
    N_LAYERS = 3

    LR = 1e-3
    d, w0 = 2, 20
    mu, k = 2**d, w0**2
    m = 1

    ACTIVATION = nn.Tanh

    ITER = 15001
    return (
        ACTIVATION,
        ITER,
        LR,
        N_HIDDEN,
        N_INPUT,
        N_LAYERS,
        N_OUTPUT,
        d,
        k,
        m,
        mu,
        w0,
    )


@app.cell
def _(np):
    def exact_solution(d, w0, t):
        assert d < w0
        w = np.sqrt(w0**2 - d**2)
        phi = np.arctan(-d / w)
        A = 1/(2 * np.cos(phi))
        cos = np.cos(phi + w * t)
        exp = np.exp(-d * t)
        u = exp * 2 * A * cos
        return u

    return (exact_solution,)


@app.cell
def _(nn):
    class FCN(nn.Module):
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation=nn.Tanh):
            super().__init__()
            self.net = nn.Sequential(

                # in
                nn.Linear(N_INPUT, N_HIDDEN), activation(), # in

                # hidden
                *[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS - 1)],

                # out
                nn.Linear(N_HIDDEN, N_OUTPUT)
            )

        def forward(self, x):
            return self.net(x)


    return (FCN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1: trian a PINN to simulate the system

    - Inputs: underlying differential equations and the initial considtions of the system
    - Outputs: estimate of the solution, $u(t)$

    ### Approach
    The PINN is trained to directly approximate the solution to the differential equation, i.e.

    $$
    NN(t; \theta ) \approx u(t)
    $$

    For this task we use $\delta = 2$, $\omega_0 = 20$, $m = 1$, and try to learn the solution over the domain $t \in [0, 1]$

    ### Loss function

    To simulate the system, the PINN is trained with the following loss function:

    $$
    \mathcal{L}(\theta) = (NN(0; \theta) - 1)^2 + \lambda_1 \left( \frac{d NN}{dt}(0; \theta) - 0 \right)^2 + \frac{\lambda_2}{N} \sum_{i}^{N} \left( \left[ m\frac{d^2}{dt^2} + \mu\frac{d}{dt} + k \right] NN(t_i; \theta) \right)^2
    $$

    ### Computing gradients

    To compute gradients of the network with respect to its inputs, we will use `torch.autograd.grad`:

    ```
    torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
    create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)
    Compute and return the sum of gradients of outputs with respect to the inputs.
    ```
    """)
    return


@app.cell
def _(
    ACTIVATION,
    FCN,
    ITER,
    LR,
    N_HIDDEN,
    N_INPUT,
    N_LAYERS,
    N_OUTPUT,
    d,
    exact_solution,
    go,
    k,
    m,
    mo,
    mu,
    np,
    torch,
    w0,
):
    torch.manual_seed(123)

    PINN = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, ACTIVATION)

    # boundary point: single point at t=0 with u(0)=1
    t_boundary = torch.tensor(0., requires_grad=True).view(-1, 1)
    # physics points: 30 evenly spaced points over [0,1] where ODE must hold
    t_physics  = torch.linspace(0, 1, 60,  requires_grad=True).view(-1, 1)
    # test points: 300 points to evaluate and plot the solution
    t_test     = torch.linspace(0, 1, 600).view(-1, 1)

    u_exact = exact_solution(d, w0, t_test)
    optimiser = torch.optim.Adam(PINN.parameters(), lr=LR)




    t_np      = t_test[:, 0].numpy()
    t_phys_np = t_physics.detach()[:, 0].numpy()
    losses    = []
    fig = go.Figure([
        go.Scatter(x=t_np,      y=u_exact[:, 0],      name="Exact solution", line=dict(color="grey")),
        go.Scatter(x=t_np,      y=np.zeros_like(t_np), name="PINN solution",  line=dict(color="green")),
        go.Scatter(x=t_phys_np, y=np.zeros(60), mode="markers", name="Physics pts", marker=dict(color="green", size=5)),
        go.Scatter(x=[0],       y=[0],          mode="markers", name="Boundary pt",  marker=dict(color="red",   size=8)),
    ])
    fig.update_layout(xaxis_title="t", yaxis_title="u(t)")




    for i in range(ITER):
        optimiser.zero_grad()
        lambda1, lambda2 = 1e-1, 1e-4

        # --- boundary loss ---
        # loss1: enforce u(0) = 1
        u     = PINN(t_boundary)
        loss1 = (u - 1)**2
        # loss2: enforce u'(0) = 0 (initial velocity)
        dudt  = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
        loss2 = dudt**2

        # --- physics loss ---
        # enforce the ODE: m*u'' + mu*u' + k*u = 0 across the domain
        u      = PINN(t_physics)
        dudt   = torch.autograd.grad(u,    t_physics, torch.ones_like(u),    create_graph=True)[0]
        d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
        loss3  = torch.mean((m*d2udt2 + mu*dudt + k*u)**2)

        loss = loss1 + lambda1*loss2 + lambda2*loss3
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    
        if i % 100 == 0:
            with torch.no_grad():
                u_pred = PINN(t_test).numpy()
            fig.update_traces(y=u_pred[:, 0], selector=dict(name="PINN solution"))
            fig.update_layout(title=f"Training step {i} | Loss: {loss.item():.2e}")
            mo.output.replace(mo.ui.plotly(fig))
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PINNs for solving wave equation, can it be used to simulate how the photons from fNIRS travel within the skull and brain tissue? from source to detector?

    uisng PINN to solve inverse problems
    """)
    return


if __name__ == "__main__":
    app.run()
