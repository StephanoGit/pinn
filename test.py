import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # import torch
    # import torch.nn as nn
    # import numpy as np
    # import plotly.graph_objects as go
    # import marimo as mo

    # # ---------------------------------------------------------------------------
    # # device
    # # ---------------------------------------------------------------------------
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # print(f"using {device}")

    # # ---------------------------------------------------------------------------
    # # network
    # # ---------------------------------------------------------------------------
    # class FCN(nn.Module):
    #     def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation=nn.Tanh):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(N_INPUT, N_HIDDEN),
    #             activation(),
    #             *[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS - 1)],
    #             nn.Linear(N_HIDDEN, N_OUTPUT)
    #         )

    #     def forward(self, x):
    #         return self.net(x)

    # # ---------------------------------------------------------------------------
    # # geometry
    # # ---------------------------------------------------------------------------
    # Z_SCALP = 0.01
    # Z_SKULL = 0.02
    # Z_BRAIN = 0.03

    # D_SCALP = 1 / (3 * 10.0)
    # D_SKULL = 1 / (3 * 1.0)
    # D_BRAIN = 1 / (3 * 2.0)

    # SOURCE_X = 0.0
    # SOURCE_Z = 0.0

    # DETECTOR_X = torch.tensor([0.01, 0.02, 0.03], device=device).view(-1, 1)
    # DETECTOR_Z = torch.zeros(3, device=device).view(-1, 1)

    # MU_A_TRUE = {"scalp": 0.01, "skull": 0.004, "brain": 0.02}

    # # physiological priors — used for regularization
    # # these represent typical literature values for healthy adult tissue
    # MU_A_PRIOR = {"scalp": 0.015, "skull": 0.008, "brain": 0.018}

    # ITER    = 20000
    # LR      = 1e-3
    # N_PHYS  = 2000
    # N_IFACE = 200
    # N_BC    = 200
    # N_GRID  = 50

    # lambda_pde   = 1.0
    # lambda_data  = 5.0
    # lambda_iface = 1e1
    # lambda_bc    = 1e1
    # lambda_reg   = 5.0    # pulls mu_a toward physiological prior — essential for ill-posed problem

    # # ---------------------------------------------------------------------------
    # # model and softplus-parameterised absorption coefficients
    # # ---------------------------------------------------------------------------
    # torch.manual_seed(42)
    # PINN = FCN(N_INPUT=2, N_OUTPUT=1, N_HIDDEN=64, N_LAYERS=5).to(device)

    # raw_mu_a_scalp = torch.tensor([0.0], requires_grad=True, device=device)
    # raw_mu_a_skull = torch.tensor([0.0], requires_grad=True, device=device)
    # raw_mu_a_brain = torch.tensor([0.0], requires_grad=True, device=device)

    # softplus = nn.Softplus()

    # def get_mu_a():
    #     return softplus(raw_mu_a_scalp), softplus(raw_mu_a_skull), softplus(raw_mu_a_brain)

    # # ---------------------------------------------------------------------------
    # # helpers
    # # ---------------------------------------------------------------------------
    # def get_optical_properties(z):
    #     mu_a_s, mu_a_sk, mu_a_b = get_mu_a()
    #     D    = torch.where(z < Z_SCALP,
    #                 torch.tensor(D_SCALP, device=device),
    #            torch.where(z < Z_SKULL,
    #                 torch.tensor(D_SKULL, device=device),
    #                 torch.tensor(D_BRAIN, device=device)))
    #     mu_a = torch.where(z < Z_SCALP, mu_a_s.expand_as(z),
    #            torch.where(z < Z_SKULL, mu_a_sk.expand_as(z),
    #                                     mu_a_b.expand_as(z)))
    #     return D, mu_a

    # def source(x, z, strength=1.0, sigma=0.002):
    #     return strength * torch.exp(
    #         -((x - SOURCE_X)**2 + (z - SOURCE_Z)**2) / (2 * sigma**2)
    #     )

    # # ---------------------------------------------------------------------------
    # # simulate layered measurements
    # # uses a weighted combination of per-layer analytic solutions
    # # so each detector reading actually encodes information from all three layers
    # # weight by how much photon path at that source-detector separation
    # # samples each layer (deeper detectors = more skull/brain contribution)
    # # ---------------------------------------------------------------------------
    # def simulate_measurements():
    #     with torch.no_grad():
    #         rho = DETECTOR_X.cpu().numpy()  # (3,1) source-detector separations

    #         # per-layer effective attenuation
    #         mu_eff_scalp = np.sqrt(MU_A_TRUE["scalp"] / D_SCALP)
    #         mu_eff_skull = np.sqrt(MU_A_TRUE["skull"] / D_SKULL)
    #         mu_eff_brain = np.sqrt(MU_A_TRUE["brain"] / D_BRAIN)

    #         # analytic fluence from each layer (diffusion Green's function)
    #         Phi_scalp = np.exp(-mu_eff_scalp * rho) / (4 * np.pi * D_SCALP * rho + 1e-8)
    #         Phi_skull = np.exp(-mu_eff_skull * rho) / (4 * np.pi * D_SKULL * rho + 1e-8)
    #         Phi_brain = np.exp(-mu_eff_brain * rho) / (4 * np.pi * D_BRAIN * rho + 1e-8)

    #         # layer weights: closer detectors dominated by scalp,
    #         # farther detectors increasingly sample deeper layers
    #         # (simplified banana-shaped path model)
    #         rho_norm  = rho / rho.max()
    #         w_scalp   = np.exp(-rho_norm * 3)          # exponentially less with distance
    #         w_skull   = rho_norm * np.exp(-rho_norm)    # peaks at mid-range
    #         w_brain   = rho_norm**2                     # grows with separation

    #         # normalise weights to sum to 1
    #         w_total = w_scalp + w_skull + w_brain + 1e-8
    #         w_scalp /= w_total; w_skull /= w_total; w_brain /= w_total

    #         Phi = w_scalp * Phi_scalp + w_skull * Phi_skull + w_brain * Phi_brain
    #         noise = 0.001 * np.random.randn(*Phi.shape)

    #         # log-transform: standard in fNIRS/DOT since intensities are log-normal
    #         return torch.tensor(np.log(np.abs(Phi + noise) + 1e-8),
    #                            dtype=torch.float32, device=device)

    # log_detector_measurements = simulate_measurements()
    # print("detector log-measurements:", log_detector_measurements.cpu().numpy().flatten())

    # # ---------------------------------------------------------------------------
    # # collocation points — sampled once
    # # ---------------------------------------------------------------------------
    # x_phys = torch.FloatTensor(N_PHYS, 1).uniform_(-0.05, 0.05).to(device).requires_grad_(True)
    # z_phys = torch.FloatTensor(N_PHYS, 1).uniform_(0.0, Z_BRAIN).to(device).requires_grad_(True)

    # x_iface    = torch.FloatTensor(N_IFACE, 1).uniform_(-0.05, 0.05).to(device)
    # z_skull_if = torch.full((N_IFACE, 1), Z_SCALP, device=device, requires_grad=True)
    # z_brain_if = torch.full((N_IFACE, 1), Z_SKULL, device=device, requires_grad=True)

    # x_bc = torch.FloatTensor(N_BC, 1).uniform_(-0.05, 0.05).to(device).requires_grad_(True)
    # z_bc = torch.zeros(N_BC, 1, device=device, requires_grad=True)

    # # ---------------------------------------------------------------------------
    # # optimiser
    # # ---------------------------------------------------------------------------
    # optimiser = torch.optim.Adam(
    #     list(PINN.parameters()) + [raw_mu_a_scalp, raw_mu_a_skull, raw_mu_a_brain],
    #     lr=LR
    # )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5000, gamma=0.5)

    # # ---------------------------------------------------------------------------
    # # evaluation grid
    # # ---------------------------------------------------------------------------
    # x_grid = torch.linspace(-0.05, 0.05, N_GRID)
    # z_grid = torch.linspace(0.0, Z_BRAIN, N_GRID)
    # XX, ZZ = torch.meshgrid(x_grid, z_grid, indexing="ij")
    # xz_grid = torch.cat([XX.reshape(-1, 1), ZZ.reshape(-1, 1)], dim=1).to(device)

    # # ---------------------------------------------------------------------------
    # # history
    # # ---------------------------------------------------------------------------
    # losses        = []
    # history_scalp = []
    # history_skull = []
    # history_brain = []

    # # ---------------------------------------------------------------------------
    # # figures
    # # ---------------------------------------------------------------------------
    # fig_field = go.Figure()
    # fig_field.add_heatmap(x=x_grid.numpy(), y=z_grid.numpy(),
    #                       z=np.zeros((N_GRID, N_GRID)),
    #                       colorscale="Hot", colorbar=dict(title="log(Φ)"))
    # for z_line, label in [(Z_SCALP, "scalp/skull"), (Z_SKULL, "skull/brain")]:
    #     fig_field.add_hline(y=z_line, line=dict(color="cyan", dash="dash"),
    #                         annotation_text=label, annotation_position="right")
    # fig_field.add_scatter(x=[SOURCE_X], y=[SOURCE_Z], mode="markers",
    #                       marker=dict(color="yellow", size=12, symbol="star"), name="Source")
    # fig_field.add_scatter(x=DETECTOR_X.cpu()[:, 0].numpy(), y=DETECTOR_Z.cpu()[:, 0].numpy(),
    #                       mode="markers", marker=dict(color="cyan", size=10, symbol="square"), name="Detectors")
    # fig_field.update_layout(xaxis_title="x (m)", yaxis_title="z (m)",
    #                         yaxis=dict(autorange="reversed"), title="log(Φ) fluence field")

    # fig_loss = go.Figure()
    # fig_loss.add_scatter(y=[], mode="lines", name="Total loss", line=dict(color="grey"))
    # fig_loss.update_layout(xaxis_title="Iteration", yaxis_title="Loss",
    #                        yaxis_type="log", title="Training loss")

    # fig_mu = go.Figure()
    # fig_mu.add_scatter(y=[], mode="lines", name="scalp (PINN)", line=dict(color="orange"))
    # fig_mu.add_scatter(y=[], mode="lines", name="skull (PINN)", line=dict(color="royalblue"))
    # fig_mu.add_scatter(y=[], mode="lines", name="brain (PINN)", line=dict(color="mediumseagreen"))
    # fig_mu.add_hline(y=MU_A_TRUE["scalp"], line=dict(color="orange",         dash="dash"), annotation_text="scalp true")
    # fig_mu.add_hline(y=MU_A_TRUE["skull"], line=dict(color="royalblue",      dash="dash"), annotation_text="skull true")
    # fig_mu.add_hline(y=MU_A_TRUE["brain"], line=dict(color="mediumseagreen", dash="dash"), annotation_text="brain true")
    # fig_mu.update_layout(xaxis_title="Iteration", yaxis_title="μₐ (mm⁻¹)",
    #                      title="μₐ convergence vs true values")

    # # ---------------------------------------------------------------------------
    # # training loop
    # # ---------------------------------------------------------------------------
    # for i in range(ITER):
    #     optimiser.zero_grad()
    #     mu_a_scalp, mu_a_skull, mu_a_brain = get_mu_a()

    #     # --- PDE loss: -∇·(D∇Φ) + μₐΦ = S ---
    #     xz_phys = torch.cat([x_phys, z_phys], dim=1)
    #     Phi     = PINN(xz_phys)
    #     D, mu_a = get_optical_properties(z_phys)
    #     S       = source(x_phys, z_phys)

    #     dPhi_dx   = torch.autograd.grad(Phi,     x_phys, torch.ones_like(Phi),     create_graph=True)[0]
    #     dPhi_dz   = torch.autograd.grad(Phi,     z_phys, torch.ones_like(Phi),     create_graph=True)[0]
    #     d2Phi_dx2 = torch.autograd.grad(dPhi_dx, x_phys, torch.ones_like(dPhi_dx), create_graph=True)[0]
    #     d2Phi_dz2 = torch.autograd.grad(dPhi_dz, z_phys, torch.ones_like(dPhi_dz), create_graph=True)[0]
    #     loss_pde  = torch.mean((-D * (d2Phi_dx2 + d2Phi_dz2) + mu_a * Phi - S)**2)

    #     # --- data loss in log domain ---
    #     xz_det      = torch.cat([DETECTOR_X, DETECTOR_Z], dim=1)
    #     Phi_det     = PINN(xz_det)
    #     log_Phi_det = torch.log(torch.abs(Phi_det) + 1e-8)
    #     loss_data   = torch.mean((log_Phi_det - log_detector_measurements)**2)

    #     # --- interface loss ---
    #     loss_iface = torch.mean(
    #         (PINN(torch.cat([x_iface, z_skull_if], dim=1)) -
    #          PINN(torch.cat([x_iface, z_brain_if], dim=1)))**2
    #     )

    #     # --- Robin BC at scalp: Φ + 2D·dΦ/dz = 0 ---
    #     Phi_bc  = PINN(torch.cat([x_bc, z_bc], dim=1))
    #     dPhi_bc = torch.autograd.grad(Phi_bc, z_bc, torch.ones_like(Phi_bc), create_graph=True)[0]
    #     loss_bc = torch.mean((Phi_bc + 2 * D_SCALP * dPhi_bc)**2)

    #     # --- regularization: pull mu_a toward physiological priors ---
    #     # without this the inverse problem is underdetermined —
    #     # many mu_a combinations can explain the same surface measurements
    #     loss_reg = (
    #         (mu_a_scalp - MU_A_PRIOR["scalp"])**2 +
    #         (mu_a_skull - MU_A_PRIOR["skull"])**2 +
    #         (mu_a_brain - MU_A_PRIOR["brain"])**2
    #     )

    #     loss = (lambda_pde  * loss_pde   +
    #             lambda_data * loss_data  +
    #             lambda_iface* loss_iface +
    #             lambda_bc   * loss_bc    +
    #             lambda_reg  * loss_reg)

    #     loss.backward()
    #     optimiser.step()
    #     scheduler.step()

    #     losses.append(loss.cpu().item())
    #     history_scalp.append(mu_a_scalp.cpu().item())
    #     history_skull.append(mu_a_skull.cpu().item())
    #     history_brain.append(mu_a_brain.cpu().item())

    #     if i % 100 == 0:
    #         with torch.no_grad():
    #             Phi_grid = PINN(xz_grid).cpu().reshape(N_GRID, N_GRID).numpy()

    #         fig_field.update_traces(z=np.log1p(np.abs(Phi_grid)).T, selector=dict(type="heatmap"))
    #         fig_field.update_layout(title=f"log(Φ) fluence field — step {i}")
    #         fig_loss.update_traces(y=losses, selector=dict(name="Total loss"))

    #         iters = list(range(len(history_scalp)))
    #         fig_mu.update_traces(x=iters, y=history_scalp, selector=dict(name="scalp (PINN)"))
    #         fig_mu.update_traces(x=iters, y=history_skull, selector=dict(name="skull (PINN)"))
    #         fig_mu.update_traces(x=iters, y=history_brain, selector=dict(name="brain (PINN)"))
    #         fig_mu.update_layout(
    #             title=(
    #                 f"μₐ convergence — step {i} | "
    #                 f"scalp: {mu_a_scalp.cpu().item():.4f} (true {MU_A_TRUE['scalp']}) | "
    #                 f"skull: {mu_a_skull.cpu().item():.4f} (true {MU_A_TRUE['skull']}) | "
    #                 f"brain: {mu_a_brain.cpu().item():.4f} (true {MU_A_TRUE['brain']})"
    #             )
    #         )
    #         mo.output.replace(mo.vstack([
    #             mo.ui.plotly(fig_field),
    #             mo.ui.plotly(fig_mu),
    #             mo.ui.plotly(fig_loss),
    #         ]))
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import plotly.graph_objects as go
    import marimo as mo

    # ---------------------------------------------------------------------------
    # device
    # ---------------------------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using {device}")

    # ---------------------------------------------------------------------------
    # SOTA Architecture: Fourier Feature Network
    # ---------------------------------------------------------------------------
    class FourierFeatureNet(nn.Module):
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, sigma=5.0):
            super().__init__()
            self.B = nn.Parameter(torch.randn(N_INPUT, N_HIDDEN // 2) * sigma, requires_grad=False)
        
            layers = [nn.Linear(N_HIDDEN, N_HIDDEN), nn.SiLU()]
            for _ in range(N_LAYERS - 1):
                layers.extend([nn.Linear(N_HIDDEN, N_HIDDEN), nn.SiLU()])
            layers.append(nn.Linear(N_HIDDEN, N_OUTPUT))
        
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            x_proj = 2.0 * np.pi * x @ self.B
            features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return self.net(features)

    # ---------------------------------------------------------------------------
    # geometry (METERS)
    # ---------------------------------------------------------------------------
    Z_SCALP = 0.01
    Z_SKULL = 0.02
    Z_BRAIN = 0.03

    SOURCE_X = 0.0
    SOURCE_Z = 0.001 

    # High-Density Detector Array
    N_DETECTORS = 15
    DETECTOR_X = torch.linspace(0.005, 0.045, N_DETECTORS, device=device).view(-1, 1)
    DETECTOR_Z = torch.zeros(N_DETECTORS, device=device).view(-1, 1)

    # ---------------------------------------------------------------------------
    # Optical Properties (SCALED TO METERS)
    # ---------------------------------------------------------------------------
    D_SCALP = 1 / (3 * 10000.0) 
    D_SKULL = 1 / (3 * 1000.0)
    D_BRAIN = 1 / (3 * 2000.0)

    MU_A_TRUE  = {"scalp": 10.0, "skull": 4.0, "brain": 20.0}
    MU_A_PRIOR = {"scalp": 15.0, "skull": 8.0, "brain": 18.0}

    PHASE_1_STEPS = 5000
    PHASE_2_STEPS = 15000
    ITER = PHASE_1_STEPS + PHASE_2_STEPS

    N_BC   = 200
    N_GRID = 50

    # Balanced Weights
    lambda_pde   = 1.0
    lambda_data  = 50.0   
    lambda_bc    = 1.0
    lambda_sym   = 10.0   
    lambda_far   = 1.0    
    lambda_reg   = 0.1    

    # ---------------------------------------------------------------------------
    # model and learnable parameters (Unlocked in Phase 2)
    # ---------------------------------------------------------------------------
    torch.manual_seed(42)
    PINN = FourierFeatureNet(N_INPUT=2, N_OUTPUT=1, N_HIDDEN=128, N_LAYERS=5).to(device)

    raw_mu_a_scalp = torch.tensor([15.0], requires_grad=True, device=device)
    raw_mu_a_skull = torch.tensor([8.0], requires_grad=True, device=device)
    raw_mu_a_brain = torch.tensor([18.0], requires_grad=True, device=device)

    softplus = nn.Softplus()

    def get_mu_a(is_phase_1=False):
        if is_phase_1:
            # Phase 1: Force true physics to generate perfect data
            return torch.tensor([MU_A_TRUE["scalp"]], device=device), \
                   torch.tensor([MU_A_TRUE["skull"]], device=device), \
                   torch.tensor([MU_A_TRUE["brain"]], device=device)
        else:
            # Phase 2: Inverse problem using PINN parameters
            return softplus(raw_mu_a_scalp), softplus(raw_mu_a_skull), softplus(raw_mu_a_brain)

    def get_optical_properties(z, is_phase_1):
        mu_a_s, mu_a_sk, mu_a_b = get_mu_a(is_phase_1)
        D    = torch.where(z < Z_SCALP, torch.tensor(D_SCALP, device=device),
               torch.where(z < Z_SKULL, torch.tensor(D_SKULL, device=device),
                                        torch.tensor(D_BRAIN, device=device)))
        mu_a = torch.where(z < Z_SCALP, mu_a_s.expand_as(z),
               torch.where(z < Z_SKULL, mu_a_sk.expand_as(z),
                                        mu_a_b.expand_as(z)))
        return D, mu_a

    def log_source(x, z, strength=100.0, sigma=0.002):
        r2 = (x - SOURCE_X)**2 + (z - SOURCE_Z)**2
        return np.log(strength) - r2 / (2 * sigma**2)

    def sample_collocation_points(n_bg=1500, n_src=500):
        x_bg = torch.empty(n_bg, 1, device=device).uniform_(-0.05, 0.05)
        z_bg = torch.empty(n_bg, 1, device=device).uniform_(0.0, Z_BRAIN)
        x_src = torch.empty(n_src, 1, device=device).normal_(SOURCE_X, 0.005)
        z_src = torch.abs(torch.empty(n_src, 1, device=device).normal_(SOURCE_Z, 0.005))
        x_phys = torch.cat([x_bg, x_src], dim=0).requires_grad_(True)
        z_phys = torch.cat([z_bg, z_src], dim=0).requires_grad_(True)
        return x_phys, z_phys

    # ---------------------------------------------------------------------------
    # static boundaries
    # ---------------------------------------------------------------------------
    x_bc = torch.FloatTensor(N_BC, 1).uniform_(-0.05, 0.05).to(device).requires_grad_(True)
    z_bc = torch.zeros(N_BC, 1, device=device, requires_grad=True)

    x_left   = torch.full((N_BC, 1), -0.05, device=device)
    z_sides  = torch.FloatTensor(N_BC, 1).uniform_(0.0, Z_BRAIN).to(device)
    x_right  = torch.full((N_BC, 1), 0.05, device=device)
    x_bottom = torch.FloatTensor(N_BC, 1).uniform_(-0.05, 0.05).to(device)
    z_bottom = torch.full((N_BC, 1), Z_BRAIN, device=device)

    far_points = torch.cat([
        torch.cat([x_left, z_sides], dim=1),
        torch.cat([x_right, z_sides], dim=1),
        torch.cat([x_bottom, z_bottom], dim=1)
    ], dim=0)

    # ---------------------------------------------------------------------------
    # Optimizers for the Two Phases
    # ---------------------------------------------------------------------------
    # Phase 1 only trains the network
    optimiser_fwd = torch.optim.Adam(PINN.parameters(), lr=1e-3)

    # Phase 2 trains the network AND the parameters
    optimiser_inv = torch.optim.Adam([
        {'params': PINN.parameters(), 'lr': 1e-3},
        {'params': [raw_mu_a_scalp, raw_mu_a_skull, raw_mu_a_brain], 'lr': 5e-3}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser_inv, step_size=5000, gamma=0.5)

    x_grid = torch.linspace(-0.05, 0.05, N_GRID)
    z_grid = torch.linspace(0.0, Z_BRAIN, N_GRID)
    XX, ZZ = torch.meshgrid(x_grid, z_grid, indexing="ij")
    xz_grid = torch.cat([XX.reshape(-1, 1), ZZ.reshape(-1, 1)], dim=1).to(device)

    losses, history_scalp, history_skull, history_brain = [], [], [], []
    log_detector_measurements = None # Will be generated at step 5000

    # ---------------------------------------------------------------------------
    # figures
    # ---------------------------------------------------------------------------
    fig_field = go.Figure()
    fig_field.add_heatmap(x=x_grid.numpy(), y=z_grid.numpy(), z=np.zeros((N_GRID, N_GRID)), colorscale="Hot", colorbar=dict(title="log(Φ)"))
    for z_line, label in [(Z_SCALP, "scalp/skull"), (Z_SKULL, "skull/brain")]:
        fig_field.add_hline(y=z_line, line=dict(color="cyan", dash="dash"), annotation_text=label, annotation_position="right")
    fig_field.add_scatter(x=[SOURCE_X], y=[SOURCE_Z], mode="markers", marker=dict(color="yellow", size=12, symbol="star"), name="Source")
    fig_field.add_scatter(x=DETECTOR_X.cpu()[:, 0].numpy(), y=DETECTOR_Z.cpu()[:, 0].numpy(), mode="markers", marker=dict(color="cyan", size=5, symbol="square"), name="Detectors")
    fig_field.update_layout(xaxis_title="x (m)", yaxis_title="z (m)", yaxis=dict(autorange="reversed"), title="log(Φ) fluence field")

    fig_loss = go.Figure()
    fig_loss.add_scatter(y=[], mode="lines", name="Total loss", line=dict(color="grey"))
    fig_loss.update_layout(xaxis_title="Iteration", yaxis_title="Loss", yaxis_type="log", title="Training loss")

    fig_mu = go.Figure()
    fig_mu.add_scatter(y=[], mode="lines", name="scalp (PINN)", line=dict(color="orange"))
    fig_mu.add_scatter(y=[], mode="lines", name="skull (PINN)", line=dict(color="royalblue"))
    fig_mu.add_scatter(y=[], mode="lines", name="brain (PINN)", line=dict(color="mediumseagreen"))
    fig_mu.add_hline(y=MU_A_TRUE["scalp"], line=dict(color="orange",         dash="dash"), annotation_text="scalp true")
    fig_mu.add_hline(y=MU_A_TRUE["skull"], line=dict(color="royalblue",      dash="dash"), annotation_text="skull true")
    fig_mu.add_hline(y=MU_A_TRUE["brain"], line=dict(color="mediumseagreen", dash="dash"), annotation_text="brain true")
    fig_mu.update_layout(xaxis_title="Iteration", yaxis_title="μₐ (m⁻¹)", title="μₐ convergence vs true values")

    # ---------------------------------------------------------------------------
    # Two-Phase Training Loop
    # ---------------------------------------------------------------------------
    x_phys, z_phys = sample_collocation_points()

    for i in range(ITER):
        is_phase_1 = i < PHASE_1_STEPS
    
        # --- TRANSITION: Generate Ground Truth Data ---
        if i == PHASE_1_STEPS:
            with torch.no_grad():
                log_detector_measurements = PINN(torch.cat([DETECTOR_X, DETECTOR_Z], dim=1)).detach()

        if is_phase_1:
            optimiser_fwd.zero_grad()
        else:
            optimiser_inv.zero_grad()

        if i % 10 == 0:
            x_phys, z_phys = sample_collocation_points()

        mu_a_scalp, mu_a_skull, mu_a_brain = get_mu_a(is_phase_1)
    
        # --- PDE Loss ---
        xz_phys = torch.cat([x_phys, z_phys], dim=1)
        u_phys  = PINN(xz_phys) 
        D, mu_a = get_optical_properties(z_phys, is_phase_1)
    
        du_dx   = torch.autograd.grad(u_phys, x_phys, torch.ones_like(u_phys), create_graph=True)[0]
        du_dz   = torch.autograd.grad(u_phys, z_phys, torch.ones_like(u_phys), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx,  x_phys, torch.ones_like(du_dx),  create_graph=True)[0]
        d2u_dz2 = torch.autograd.grad(du_dz,  z_phys, torch.ones_like(du_dz),  create_graph=True)[0]
    
        log_S = log_source(x_phys, z_phys)
        source_term = torch.exp(log_S - u_phys) 
    
        pde_residual = -D * (d2u_dx2 + d2u_dz2 + du_dx**2 + du_dz**2) + mu_a - source_term
        loss_pde  = torch.mean(pde_residual**2)

        # --- Symmetry & Boundaries ---
        xz_left  = torch.cat([-x_phys, z_phys], dim=1)
        xz_right = torch.cat([x_phys, z_phys], dim=1)
        loss_sym = torch.mean((PINN(xz_left) - PINN(xz_right))**2)

        u_bc  = PINN(torch.cat([x_bc, z_bc], dim=1))
        du_bc = torch.autograd.grad(u_bc, z_bc, torch.ones_like(u_bc), create_graph=True)[0]
        loss_bc = torch.mean((1.0 + 2 * D_SCALP * du_bc)**2)

        u_far    = PINN(far_points)
        loss_far = torch.mean((u_far - (-15.0))**2)

        # --- Combine Losses based on Phase ---
        if is_phase_1:
            loss = lambda_pde * loss_pde + lambda_sym * loss_sym + lambda_far * loss_far + lambda_bc * loss_bc
            loss.backward()
            optimiser_fwd.step()
        else:
            # Inverse Mode: Include Data & Regularization Loss
            u_det     = PINN(torch.cat([DETECTOR_X, DETECTOR_Z], dim=1))
            loss_data = torch.mean((u_det - log_detector_measurements)**2)
        
            loss_reg = (
                (mu_a_scalp - MU_A_PRIOR["scalp"])**2 +
                (mu_a_skull - MU_A_PRIOR["skull"])**2 +
                (mu_a_brain - MU_A_PRIOR["brain"])**2
            )
        
            loss = (lambda_pde * loss_pde   +
                    lambda_data * loss_data  +
                    lambda_sym  * loss_sym   +
                    lambda_far  * loss_far   +
                    lambda_bc   * loss_bc    +
                    lambda_reg  * loss_reg)
                
            loss.backward()
            optimiser_inv.step()
            scheduler.step()

        losses.append(loss.cpu().item())
        history_scalp.append(mu_a_scalp.cpu().item())
        history_skull.append(mu_a_skull.cpu().item())
        history_brain.append(mu_a_brain.cpu().item())

        if i % 100 == 0:
            with torch.no_grad():
                u_grid = PINN(xz_grid).cpu().reshape(N_GRID, N_GRID).numpy()

            fig_field.update_traces(z=u_grid.T, selector=dict(type="heatmap"))
            mode_text = "FORWARD MODE (Generating True Data)" if is_phase_1 else "INVERSE MODE (Recovering Parameters)"
            fig_field.update_layout(title=f"log(Φ) fluence field — step {i} | {mode_text}")
            fig_loss.update_traces(y=losses, selector=dict(name="Total loss"))

            iters = list(range(len(history_scalp)))
            fig_mu.update_traces(x=iters, y=history_scalp, selector=dict(name="scalp (PINN)"))
            fig_mu.update_traces(x=iters, y=history_skull, selector=dict(name="skull (PINN)"))
            fig_mu.update_traces(x=iters, y=history_brain, selector=dict(name="brain (PINN)"))
            fig_mu.update_layout(
                title=(
                    f"μₐ convergence — step {i} | "
                    f"scalp: {mu_a_scalp.cpu().item():.4f} (true {MU_A_TRUE['scalp']}) | "
                    f"skull: {mu_a_skull.cpu().item():.4f} (true {MU_A_TRUE['skull']}) | "
                    f"brain: {mu_a_brain.cpu().item():.4f} (true {MU_A_TRUE['brain']})"
                )
            )
            mo.output.replace(mo.vstack([
                mo.ui.plotly(fig_field),
                mo.ui.plotly(fig_mu),
                mo.ui.plotly(fig_loss),
            ]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
