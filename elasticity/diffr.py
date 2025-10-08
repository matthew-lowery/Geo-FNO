import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timeit import default_timer
import sys
sys.path.append('..')
from utilities3 import *
from Adam import Adam
import torch
import torch.nn as nn
import numpy as np

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True

# class SpectralConv3d(nn.Module):
#     """
#     3D GeoFNO spectral layer with ALL four cases handled:
#       (1) grid -> grid
#       (2) point cloud -> grid
#       (3) grid -> point cloud
#       (4) point cloud -> point cloud
#     Matches the original 2D pattern: split along k1 (+/-), last axis uses rFFT (half-spectrum).
#     """
#     def __init__(self, in_channels, out_channels, modes1, modes2, modes3, s1=32, s2=32, s3=32):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
#         self.s1, self.s2, self.s3 = s1, s2, s3

#         scale = 1.0 / (in_channels * out_channels)
#         self.weights_pos = nn.Parameter(
#             scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
#         )
#         self.weights_neg = nn.Parameter(
#             scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
#         )

#     @staticmethod
#     def _compl_mul3d(u_ft, w):
#         # (B, Cin, K1, K2, K3) x (Cin, Cout, K1, K2, K3) -> (B, Cout, K1, K2, K3)
#         return torch.einsum('bixyz,ioxyz->boxyz', u_ft, w)

#     def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
#         """
#         u:
#           - grid path: (B, Cin, s1, s2, s3)
#           - point-cloud path: (B, Cin, N)
#         x_in:  (B, N_in, 3)  or None
#         x_out: (B, N_out, 3) or None
#         """
#         # rFFT constraint on last axis
#         assert self.modes1 <= self.s1, "modes1 must be <= s1"
#         assert self.modes2 <= self.s2, "modes2 must be <= s2"
#         assert self.modes3 <= (self.s3 // 2 + 1), "modes3 must be <= s3//2+1"

#         B = u.shape[0]

#         # -----------------------
#         # 1) GRID INPUT (x_in is None)
#         # -----------------------
#         if x_in is None:
#             # FFT on grid
#             u_ft = torch.fft.rfftn(u, dim=(-3, -2, -1))  # (B,Cin,s1,s2,s3//2+1)
#             s1, s2, s3 = u.size(-3), u.size(-2), u.size(-1)

#             # Slice low/high k1 bands and apply learned multipliers
#             pos = u_ft[:, :, :self.modes1, :self.modes2, :self.modes3]
#             neg = u_ft[:, :, -self.modes1:, :self.modes2, :self.modes3]
#             factor_pos = self._compl_mul3d(pos, self.weights_pos)  # (B,Cout,m1,m2,m3)
#             factor_neg = self._compl_mul3d(neg, self.weights_neg)  # (B,Cout,m1,m2,m3)

#             # 1a) grid -> grid
#             if x_out is None:
#                 out_ft = torch.zeros(
#                     B, self.out_channels, s1, s2, s3 // 2 + 1,
#                     dtype=torch.cfloat, device=u.device
#                 )
#                 out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = factor_pos
#                 out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = factor_neg
#                 return torch.fft.irfftn(out_ft, s=(s1, s2, s3), dim=(-3, -2, -1))

#             # 1b) grid -> point cloud
#             out_ft_reduced = torch.cat([factor_pos, factor_neg], dim=-3)  # (B,Cout,2*m1,m2,m3)
#             return self.ifft3d_nonuniform(out_ft_reduced, x_out, iphi, code)

#         # -----------------------
#         # 2) POINT-CLOUD INPUT (x_in is not None)
#         # -----------------------
#         u_ft = self.fft3d_nonuniform(u, x_in, iphi, code)  # (B,Cin,2*m1, m2, 2*m3-1)

#         # Learned multipliers on reduced spectrum (keep +/− along k1, keep + along k2, half along k3)
#         pos = u_ft[:, :, :self.modes1, :self.modes2, :self.modes3]
#         neg = u_ft[:, :, -self.modes1:, :self.modes2, :self.modes3]
#         factor_pos = self._compl_mul3d(pos, self.weights_pos)  # (B,Cout,m1,m2,m3)
#         factor_neg = self._compl_mul3d(neg, self.weights_neg)  # (B,Cout,m1,m2,m3)

#         # 2a) point cloud -> grid
#         if x_out is None:
#             out_ft = torch.zeros(
#                 B, self.out_channels, self.s1, self.s2, self.s3 // 2 + 1,
#                 dtype=torch.cfloat, device=u.device
#             )
#             out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = factor_pos
#             out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = factor_neg
#             return torch.fft.irfftn(out_ft, s=(self.s1, self.s2, self.s3), dim=(-3, -2, -1))

#         # 2b) point cloud -> point cloud
#         out_ft_reduced = torch.cat([factor_pos, factor_neg], dim=-3)  # (B,Cout,2*m1,m2,m3)
#         return self.ifft3d_nonuniform(out_ft_reduced, x_out, iphi, code)

#     # ---------- Nonuniform FFT (point cloud) ----------
#     def fft3d_nonuniform(self, u, x_in, iphi=None, code=None):
#         """
#         u:    (B, Cin, N)
#         x_in: (B, N, 3) in [0,1]^3
#         returns: (B, Cin, 2*m1, m2, 2*m3-1)
#         """
#         B, N = x_in.shape[:2]
#         device = x_in.device
#         m1 = 2 * self.modes1
#         m2 = self.modes2
#         m3 = 2 * self.modes3 - 1

#         # k grids (match reduced spectrum layout)
#         k1 = torch.cat([torch.arange(0, self.modes1, device=device),
#                         torch.arange(-self.modes1, 0, device=device)], 0).view(m1, 1, 1).repeat(1, m2, m3)
#         k2 = torch.arange(0, self.modes2, device=device).view(1, m2, 1).repeat(m1, 1, m3)
#         k3 = torch.cat([torch.arange(0, self.modes3, device=device),
#                         torch.arange(-(self.modes3 - 1), 0, device=device)], 0).view(1, 1, m3).repeat(m1, m2, 1)

#         x = x_in if iphi is None else iphi(x_in, code)  # (B,N,3)

#         # <x, k>  -> (B,N,m1,m2,m3)
#         K = (x[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k1 +
#              x[..., 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k2 +
#              x[..., 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k3)

#         basis = torch.exp(-1j * 2 * np.pi * K)  # (B,N,m1,m2,m3)
#         Y = torch.einsum('bcn,bnxyz->bcxyz', u.to(torch.cfloat), basis)  # (B,Cin,m1,m2,m3)
#         return Y

#     def ifft3d_nonuniform(self, out_ft_reduced, x_out, iphi=None, code=None):
#         """
#         out_ft_reduced: (B, Cout, 2*modes1, modes2, modes3)  # reduced last axis
#         x_out:          (B, N_out, 3)
#         returns:        (B, Cout, N_out)
#         """
#         B, _, m1_twice, m2_pos, m3_pos = out_ft_reduced.shape
#         device = out_ft_reduced.device

#         # Mirror along k3 to build full symmetric spectrum: last dim -> 2*modes3 - 1
#         u_ft2 = out_ft_reduced[..., 1:].flip(-1).conj()
#         u_full = torch.cat([out_ft_reduced, u_ft2], dim=-1)  # (B,Cout,2*m1, m2, 2*m3-1)

#         m1 = m1_twice
#         m2 = m2_pos
#         m3_full = 2 * m3_pos - 1  # = 2*modes3 - 1

#         # k grids consistent with u_full
#         k1 = torch.cat([torch.arange(0, m1 // 2, device=device),
#                         torch.arange(-m1 // 2, 0, device=device)], 0).view(m1, 1, 1).repeat(1, m2, m3_full)
#         k2 = torch.arange(0, m2, device=device).view(1, m2, 1).repeat(m1, 1, m3_full)
#         k3 = torch.arange(-(m3_pos - 1), m3_pos, device=device).view(1, 1, m3_full).repeat(m1, m2, 1)

#         x = x_out if iphi is None else iphi(x_out, code)  # (B,N,3)

#         K = (x[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k1 +
#              x[..., 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k2 +
#              x[..., 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k3)  # (B,N,m1,m2,m3_full)

#         basis = torch.exp(1j * 2 * np.pi * K)  # (B,N,m1,m2,m3_full)
#         Y = torch.einsum('bcxyz,bnxyz->bcn', u_full, basis).real  # (B,Cout,N)
#         return Y


import torch
import torch.nn as nn
import numpy as np

class SpectralConv3d(nn.Module):
    """
    3D spectral conv consistent with the classic pattern:
      - rFFT on the LAST axis (k3): store size s3//2+1
      - Learn FOUR blocks in (k1,k2): (+,+), (-,+), (+,-), (-,-)
      - Uses torch.fft.rfftn / irfftn (modern replacement of torch.rfft/irfft)
    Supports exactly the three flows used in your wrapper:
      (A) grid -> grid
      (B) point cloud -> grid
      (C) grid -> point cloud
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, s1=32, s2=32, s3=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.s1, self.s2, self.s3 = s1, s2, s3

        scale = 1.0 / (in_channels * out_channels)
        # Four learned quadrants along (k1,k2); last axis (k3) is reduced (rFFT)
        self.weights_pp = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))  # (+k1,+k2)
        self.weights_np = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))  # (-k1,+k2)
        self.weights_pn = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))  # (+k1,-k2)
        self.weights_nn = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))  # (-k1,-k2)

    @staticmethod
    def _mul(u_ft, w):
        # (B, Cin, K1, K2, K3) x (Cin, Cout, K1, K2, K3) -> (B, Cout, K1, K2, K3)
        return torch.einsum('bixyz,ioxyz->boxyz', u_ft, w)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        """
        u:
          - grid path:        (B, Cin, s1, s2, s3)
          - point-cloud path: (B, Cin, N)
        x_in:  (B, N_in, 3)  or None
        x_out: (B, N_out, 3) or None

        Allowed flows:
          A) grid -> grid          : x_in=None,  x_out=None
          B) point cloud -> grid   : x_in!=None, x_out=None
          C) grid -> point cloud   : x_in=None,  x_out!=None
        """
        # rFFT constraint on last axis
        assert self.modes1 <= self.s1 and self.modes2 <= self.s2 and self.modes3 <= (self.s3 // 2 + 1), \
            "Require: modes1<=s1, modes2<=s2, modes3<=s3//2+1 (rFFT last axis)."

        if x_in is None and x_out is None:
            return self._grid_to_grid(u)
        elif x_in is not None and x_out is None:
            return self._pc_to_grid(u, x_in, iphi, code)
        elif x_in is None and x_out is not None:
            return self._grid_to_pc(u, x_out, iphi, code)
        else:
            raise ValueError("point-cloud -> point-cloud is not supported in this layer (by design).")

    # ---------------- GRID -> GRID ----------------
    def _grid_to_grid(self, u):
        B, Cin, s1, s2, s3 = u.shape
        u_ft = torch.fft.rfftn(u, dim=(-3, -2, -1))  # (B,Cin,s1,s2,s3//2+1)

        # Pick four quadrants (±k1, ±k2), keep reduced last axis
        pp = u_ft[:, :, :self.modes1, :self.modes2, :self.modes3]
        np_ = u_ft[:, :, -self.modes1:, :self.modes2, :self.modes3]
        pn = u_ft[:, :, :self.modes1, -self.modes2:, :self.modes3]
        nn = u_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3]

        f_pp = self._mul(pp, self.weights_pp)
        f_np = self._mul(np_, self.weights_np)
        f_pn = self._mul(pn, self.weights_pn)
        f_nn = self._mul(nn, self.weights_nn)

        out_ft = torch.zeros(B, self.out_channels, s1, s2, s3//2 + 1, dtype=torch.cfloat, device=u.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = f_pp
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = f_np
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = f_pn
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = f_nn

        return torch.fft.irfftn(out_ft, s=(s1, s2, s3), dim=(-3, -2, -1))

    # ------------- POINT CLOUD -> GRID -------------
    # --- Drop-in fix for SpectralConv3d._pc_to_grid to avoid einsum dim bugs ---
    def _pc_to_grid(self, u, x_in, iphi, code):
        """
        u:    (B, Cin, N), x_in: (B,N,3) in [0,1]^3
        Projects to reduced rFFT spectrum (2*m1, 2*m2, m3) with four (±k1, ±k2) quadrants,
        then irFFT to canonical grid (s1,s2,s3). Uses safe matmul instead of einsum.
        """
        # Sanity: ensure (B, Cin, N)
        assert u.dim() == 3, f"u must be (B,Cin,N), got {tuple(u.shape)}"
        B, Cin, N = u.shape
        device = u.device
        m1, m2, m3 = 2 * self.modes1, 2 * self.modes2, self.modes3  # rFFT keeps last axis reduced

        # Frequency sets
        k1_pos = torch.arange(0, self.modes1, device=device)
        k1_neg = torch.arange(-self.modes1, 0, device=device)
        k2_pos = torch.arange(0, self.modes2, device=device)
        k2_neg = torch.arange(-self.modes2, 0, device=device)
        k3_red = torch.arange(0, self.modes3, device=device)

        # Warp coords if provided
        x = x_in if iphi is None else iphi(x_in, code)  # (B,N,3)
        assert x.shape[:2] == (B, N), f"x_in/iphi(x_in) must be (B,N,3), got {tuple(x.shape)}"

        def proj_block(k1_set, k2_set):
            # Build k-grid (|k1|, |k2|, m3)
            K1, K2, K3 = torch.meshgrid(k1_set, k2_set, k3_red, indexing='ij')
            K1 = K1.to(device); K2 = K2.to(device); K3 = K3.to(device)
            # Phase: (B,N,|k1|,|k2|,m3)
            phase = (x[..., 0:1, None, None] * K1[None, None, ...] +
                    x[..., 1:2, None, None] * K2[None, None, ...] +
                    x[..., 2:3, None, None] * K3[None, None, ...])
            basis = torch.exp(-1j * 2 * np.pi * phase)              # (B,N,|k1|,|k2|,m3)

            # Flatten along freq dims for a stable (B,Cin,N) @ (B,N,K) matmul
            K = K1.numel()  # = |k1|*|k2|*m3
            basis_flat = basis.reshape(B, N, K)                      # (B,N,K)
            u_c = u.to(torch.cfloat)                                 # (B,Cin,N)
            # Batched matmul: (B,Cin,N) @ (B,N,K) -> (B,Cin,K)
            Y_flat = torch.matmul(u_c, basis_flat)                   # (B,Cin,K)
            # Reshape back to (B,Cin,|k1|,|k2|,m3)
            return Y_flat.view(B, Cin, k1_set.numel(), k2_set.numel(), m3)

        # Project each quadrant
        pp = proj_block(k1_pos, k2_pos)
        np_ = proj_block(k1_neg, k2_pos)
        pn = proj_block(k1_pos, k2_neg)
        nn = proj_block(k1_neg, k2_neg)

        # Apply learned weights (Cin->Cout) via einsum on matching shapes
        f_pp = torch.einsum('bixyz,ioxyz->boxyz', pp, self.weights_pp)
        f_np = torch.einsum('bixyz,ioxyz->boxyz', np_, self.weights_np)
        f_pn = torch.einsum('bixyz,ioxyz->boxyz', pn, self.weights_pn)
        f_nn = torch.einsum('bixyz,ioxyz->boxyz', nn, self.weights_nn)

        # Place into rFFT tensor and irFFT to grid
        out_ft = torch.zeros(B, self.out_channels, self.s1, self.s2, self.s3 // 2 + 1,
                            dtype=torch.cfloat, device=device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3]  = f_pp
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = f_np
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = f_pn
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = f_nn

        return torch.fft.irfftn(out_ft, s=(self.s1, self.s2, self.s3), dim=(-3, -2, -1))


    # ------------- GRID -> POINT CLOUD -------------
    def _grid_to_pc(self, u, x_out, iphi, code):
        """
        u: (B, Cin, s1, s2, s3) on grid.
        Build reduced spectrum (2*m1, 2*m2, m3) from four FFT quadrants,
        then evaluate nonuniform iFFT at x_out.
        """
        B, Cin, s1, s2, s3 = u.shape
        device = u.device

        u_ft = torch.fft.rfftn(u, dim=(-3, -2, -1))  # (B,Cin,s1,s2,s3//2+1)

        pp = u_ft[:, :, :self.modes1, :self.modes2, :self.modes3]
        np_ = u_ft[:, :, -self.modes1:, :self.modes2, :self.modes3]
        pn = u_ft[:, :, :self.modes1, -self.modes2:, :self.modes3]
        nn = u_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3]

        f_pp = self._mul(pp, self.weights_pp)
        f_np = self._mul(np_, self.weights_np)
        f_pn = self._mul(pn, self.weights_pn)
        f_nn = self._mul(nn, self.weights_nn)

        # Reduced spectrum tensor
        red = torch.zeros(B, self.out_channels, 2*self.modes1, 2*self.modes2, self.modes3, dtype=torch.cfloat, device=device)
        red[:, :, :self.modes1, :self.modes2, :]  = f_pp
        red[:, :, -self.modes1:, :self.modes2, :] = f_np
        red[:, :, :self.modes1, -self.modes2:, :] = f_pn
        red[:, :, -self.modes1:, -self.modes2:, :] = f_nn

        return self._ifft3d_nonuniform(red, x_out, iphi, code)

    # ---------- Nonuniform iFFT for reduced spectrum ----------
    def _ifft3d_nonuniform(self, red, x_out, iphi=None, code=None):
        """
        red:   (B, Cout, 2*m1, 2*m2, m3)    # reduced last axis
        x_out: (B, N_out, 3)
        returns (B, Cout, N_out)
        """
        B, Cout, m1_twice, m2_twice, m3_pos = red.shape
        device = red.device

        # Mirror along last axis to recover full symmetric spectrum: 2*m3_pos - 1
        red_mirror = red[..., 1:].flip(-1).conj()
        full = torch.cat([red, red_mirror], dim=-1)  # (B,Cout,2*m1,2*m2, 2*m3_pos-1)

        m1 = m1_twice
        m2 = m2_twice
        m3_full = 2 * m3_pos - 1  # = 2*modes3 - 1

        # Frequency grids aligned with 'full'
        k1 = torch.cat([torch.arange(0, m1//2, device=device), torch.arange(-m1//2, 0, device=device)], 0)  # (m1,)
        k2 = torch.cat([torch.arange(0, m2//2, device=device), torch.arange(-m2//2, 0, device=device)], 0)  # (m2,)
        k3 = torch.arange(-(m3_pos - 1), m3_pos, device=device)  # (m3_full,)

        K1, K2, K3 = torch.meshgrid(k1, k2, k3, indexing='ij')  # (m1,m2,m3_full)

        x = x_out if iphi is None else iphi(x_out, code)  # (B,N,3)
        # <x, k>
        phase = (x[..., 0:1, None, None]*K1 + x[..., 1:2, None, None]*K2 + x[..., 2:3, None, None]*K3)  # (B,N,m1,m2,m3_full)
        basis = torch.exp(1j * 2 * np.pi * phase).to(device)  # (B,N,m1,m2,m3_full)

        Y = torch.einsum('bcxyz,bnxyz->bcn', full, basis).real  # (B,Cout,N)
        return Y

# ---------------------------
# 3D IPHI (coordinate warp): x -> xi
# ---------------------------
class IPHI3d(nn.Module):
    def __init__(self, width=32, code_dim=None, device='cuda'):
        super().__init__()
        self.width = width
        self.code_dim = code_dim

        in_geom = 6  # (x,y,z, radius, theta, phi)
        self.fc0 = nn.Linear(in_geom, width)

        # NeRF-style multiscale sin/cos (powers of 2)
        self.register_buffer(
            'Bfreq',
            (np.pi * torch.pow(2, torch.arange(0, max(1, width // 6), dtype=torch.float32))).view(1, 1, 1, -1)
        )

        # ---- compute concat dim dynamically ----
        num_freqs = max(1, width // 6)
        base_cat_in = width + 2 * (in_geom * num_freqs)  # fc0(width) + sin + cos
        if code_dim is not None:
            self.fc_code = nn.Linear(code_dim, width)
            cat_in = base_cat_in + width                   # add code embedding
        else:
            self.fc_code = None
            cat_in = base_cat_in

        self.fc1 = nn.Linear(cat_in, 4 * width)           # <-- was nn.Linear(92, ...)
        self.fc2 = nn.Linear(4 * width, 4 * width)
        self.fc3 = nn.Linear(4 * width, 3)

        self.center = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3)

    def forward(self, x, code=None):
        # x: (B,N,3)
        B, N, _ = x.shape
        xc = x - self.center
        radius = torch.linalg.norm(xc, dim=-1, ord=2)                       # (B,N)

        # ---- use proper atan2 without clamping (preserve sign) ----
        theta = torch.atan2(torch.linalg.norm(xc[..., :2], dim=-1), xc[..., 2])  # polar
        phi   = torch.atan2(xc[..., 1], xc[..., 0])                               # azimuth

        feats = torch.stack([x[...,0], x[...,1], x[...,2], radius, theta, phi], dim=-1)  # (B,N,6)

        h0 = self.fc0(feats)  # (B,N,W)

        # NeRF encodings
        d = feats.shape[-1]
        rep = feats.view(B, N, d, 1) * self.Bfreq  # (B,N,d,F)
        s = torch.sin(rep).view(B, N, -1)
        c = torch.cos(rep).view(B, N, -1)

        h = torch.cat([h0, s, c], dim=-1)

        if self.fc_code is not None and code is not None:
            emb = self.fc_code(code).unsqueeze(1).repeat(1, N, 1)  # (B,N,W)
            h = torch.cat([h, emb], dim=-1)

        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        delta = self.fc3(h)  # (B,N,3)
        return x + x * delta



# ---------------------------
# 3D FNO wrapper
# ---------------------------
class FNO3d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels,
                 is_mesh=False, s1=32, s2=32, s3=32):
        super().__init__()
        self.modes1, self.modes2, self.modes3 = modes, modes, modes
        self.width = width
        self.is_mesh = is_mesh
        self.s1, self.s2, self.s3 = s1, s2, s3

        self.fc0 = nn.Linear(in_channels, width)

        self.conv0 = SpectralConv3d(width, width, modes, modes, modes, s1, s2, s3)
        self.conv1 = SpectralConv3d(width, width, modes, modes, modes, s1, s2, s3)
        self.conv2 = SpectralConv3d(width, width, modes, modes, modes, s1, s2, s3)
        self.conv3 = SpectralConv3d(width, width, modes, modes, modes, s1, s2, s3)
        self.conv4 = SpectralConv3d(width, width, modes, modes, modes, s1, s2, s3)

        # 1x1x1 conv "local" paths (like your w1..w3)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)

        # Optional canonical-grid bias (3 coords -> width)
        self.b0 = nn.Conv3d(3, width, 1)
        self.b1 = nn.Conv3d(3, width, 1)
        self.b2 = nn.Conv3d(3, width, 1)
        self.b3 = nn.Conv3d(3, width, 1)

        # Output bias from x_out (coords) via 1D conv over point dimension
        self.b_out = nn.Conv1d(3, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid3d(self, B, device):
        gx = torch.linspace(0, 1, self.s1, device=device).view(1, self.s1, 1, 1, 1).repeat(B, 1, self.s2, self.s3, 1)
        gy = torch.linspace(0, 1, self.s2, device=device).view(1, 1, self.s2, 1, 1).repeat(B, self.s1, 1, self.s3, 1)
        gz = torch.linspace(0, 1, self.s3, device=device).view(1, 1, 1, self.s3, 1).repeat(B, self.s1, self.s2, 1, 1)
        grid = torch.cat([gx, gy, gz], dim=-1)  # (B,s1,s2,s3,3)
        return grid.permute(0, 4, 1, 2, 3)      # (B,3,s1,s2,s3)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        """
        u: (B, Nin, Cin) if point-cloud path, with Cin = func_dim + 3 if you concat coords,
           OR (B,Cin,s1,s2,s3) if uniform-grid path.
        x_in:  (B, Nin, 3)   input coords
        x_out: (B, Nout, 3)  output coords
        """
        B = u.shape[0]
        device = u.device

        # If using point clouds, you must provide x_in/x_out explicitly.
        # We keep the canonical-grid bias path optional:
        grid = self.get_grid3d(B, device)  # (B,3,s1,s2,s3)

        # Lift
        # print(x_in.shape, x_out.shape,)
        if x_in is not None:       # point-cloud path: (B,N,Cin) -> (B,Cin,N)
            u = self.fc0(u)        # (B,N,width)
            u = u.permute(0, 2, 1) # (B,width,N)

            # conv0 on point cloud -> latent canonical grid via learned Fourier projection
            uc = self.conv0(u, x_in=x_in, x_out=None, iphi=iphi, code=code)  # (B,width,s1,s2,s3)
            uc = uc + self.b0(grid)
            uc = F.gelu(uc)

            uc = self.conv1(uc, x_in=None, x_out=None, iphi=None) + self.w1(uc) + self.b1(grid)
            uc = F.gelu(uc)
            uc = self.conv2(uc, x_in=None, x_out=None, iphi=None) + self.w2(uc) + self.b2(grid)
            uc = F.gelu(uc)
            uc = self.conv3(uc, x_in=None, x_out=None, iphi=None) + self.w3(uc) + self.b3(grid)
            uc = F.gelu(uc)

            # Project from grid latent -> output point cloud
            u = self.conv4(uc, x_in=None, x_out=x_out, iphi=iphi, code=code)  # (B,width,Nout)
            u = u + self.b_out(x_out.permute(0, 2, 1))                        # (B,width,Nout)

            u = u.permute(0, 2, 1)   # (B,Nout,width)
            u = F.gelu(self.fc1(u))  # (B,Nout,128)
            u = self.fc2(u)          # (B,Nout,Cout)
            return u

        else:
            # uniform-grid path (vanilla 3D FNO on a tensor grid)
            u = self.fc0(u.permute(0, 2, 3, 4, 1))            # (B,s1,s2,s3,width)
            u = u.permute(0, 4, 1, 2, 3).contiguous()          # (B,width,s1,s2,s3)

            uc = self.conv0(u) + self.b0(grid)
            uc = F.gelu(uc)
            uc = self.conv1(uc) + self.w1(uc) + self.b1(grid); uc = F.gelu(uc)
            uc = self.conv2(uc) + self.w2(uc) + self.b2(grid); uc = F.gelu(uc)
            uc = self.conv3(uc) + self.w3(uc) + self.b3(grid); uc = F.gelu(uc)

            u = self.conv4(uc)                                     # (B,width,s1,s2,s3)
            u = u.permute(0, 2, 3, 4, 1).contiguous()              # (B,s1,s2,s3,width)
            u = F.gelu(self.fc1(u))                                # (B,s1,s2,s3,128)
            u = self.fc2(u)                                        # (B,s1,s2,s3,Cout)
            return u






################################################################
# configs
################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modes', type=int, default=12)
parser.add_argument('--res1d', type=int, default=40)
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
### num sin features
args = parser.parse_args()
print(args)
set_seed(args.seed)
batch_size = 100
learning_rate_fno = 0.001
learning_rate_iphi = 0.0001

epochs = 10000

### load data
fp = './data/diffrec_3d.npz'
data = np.load(fp)
dataset = fp.split('/')[-1].split('.')[0]
x, x_grid, y, y_grid = data["x"].astype(np.float32), data["x_grid"].astype(np.float32), data["y"].astype(np.float32), data["y_grid"].astype(np.float32)


y = y.reshape(1200, -1, 1)
x = x.reshape(1200, -1, 1)
x_grid = np.repeat(x_grid[None], 1200, axis=0)
y_grid = np.repeat(y_grid[None], 1200, axis=0)
ntrain,ntest = 1000,200
x,x_grid,y,y_grid = torch.tensor(x, dtype=torch.float32), torch.tensor(x_grid, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(y_grid, dtype=torch.float32)

print(x.shape, y.shape, x_grid.shape, y_grid.shape)
train_x, train_x_grid, train_y, train_y_grid = x[:ntrain], x_grid[:ntrain], y[:ntrain], y_grid[:ntrain]
test_x, test_x_grid, test_y, test_y_grid = x[-ntest:], x_grid[-ntest:], y[-ntest:], y_grid[-ntest:]
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_x_grid, train_y,train_y_grid), 
                                                                            batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_x_grid, test_y,test_y_grid), 
                                            batch_size=batch_size, shuffle=False) 

################################################################
# training and evaluation
################################################################
model = FNO3d(args.modes, args.width, in_channels=4, out_channels=1, is_mesh=False, s1=args.res1d, s2=args.res1d, s3=args.res1d).cuda()
model_iphi = IPHI3d().cuda()
print(count_params(model), count_params(model_iphi))

optimizer_fno = Adam(model.parameters(), lr=learning_rate_fno, weight_decay=1e-3)
scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max = epochs)
optimizer_iphi = Adam(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=1e-4)
scheduler_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iphi, T_max = epochs)

myloss = LpLoss(size_average=False)
N_sample = 1000
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0

    for x, x_grid, y, y_grid in train_loader:
        # rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda() ### feature thing, y, mesh
        x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad() 
        inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
        out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) ### self, u, code=None, x_in=None, x_out=None, iphi=None

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()

    scheduler_fno.step()
    scheduler_iphi.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, x_grid, y, y_grid in test_loader:
            x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()
            # print(rr.shape, sigma.shape, mesh.shape) ## 20,42 ; 20, 972, 1 ; 20, 972, 2
            # rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
            inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
            out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) ### self, u, code=None, x_in=None, x_out=None, iphi=None
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)
