import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import (
    pow2_int, 
    fp_decompose_positive, 
    next_float, 
    prev_float, 
    smallest_positive_subnormal, 
    build_input_vector
)


class IndicatorGEZ(nn.Module):

    def __init__(self, z, p: int, q: int, device='cpu', dtype=torch.float32,
                 strict_k_range: bool = True , mode='positive'):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.p = int(p)
        self.q = int(q)

        z = torch.as_tensor(z, device=device, dtype=dtype)

        e_min = -(2 ** (q - 1)) + 2
        e_max =  (2 ** (q - 1)) - 1
        e_0   =  (2 ** (q - 2)) - 1

        self.e_min = e_min
        self.e_max = e_max
        self.e_0 = e_0

        a_z, e_z = fp_decompose_positive(z)
        c_z = max(e_min - e_z, 0)

        if e_z >= e_min:
            u0 = pow2_int(-p, device=device, dtype=dtype)
        else:
            u0 = pow2_int(-p + c_z, device=device, dtype=dtype)

        a_is_one = bool((a_z == torch.ones_like(a_z)).item())

        if (not a_is_one) or (e_z == -p + e_min):
            tilde_u0 = torch.zeros((), device=device, dtype=dtype)
            tilde_c = 0
        else:
            tilde_u0 = u0 * pow2_int(-1, device=device, dtype=dtype)
            tilde_c = 1

        # k
        if 0 < e_z <= (-2 - p + e_max):
            k = -p + e_0
        elif e_z <= 0:
            k = 3 - p - e_0 - c_z
        else:

            k = -p + e_0


        beta = (torch.tensor(2.0, device=device, dtype=dtype) - u0) * pow2_int(e_z, device=device, dtype=dtype)
        A    = pow2_int(p - e_z + k, device=device, dtype=dtype)
        B1   = (torch.tensor(2.0, device=device, dtype=dtype) - a_z - tilde_u0) * pow2_int(p + k, device=device, dtype=dtype)
        B2   = (torch.tensor(2.0, device=device, dtype=dtype) - a_z - u0)       * pow2_int(p + k, device=device, dtype=dtype)
        S    = pow2_int(tilde_c - c_z - k, device=device, dtype=dtype)


        self.lin1 = nn.Linear(1, 1, bias=True, dtype=dtype).to(device)
        if mode == "positive":
          w = -1.0
        elif mode == "negative":
          w = 1.0
        self.lin1.weight = nn.Parameter(
            torch.tensor([[w]], device=device, dtype=dtype),
            requires_grad=True
        )
        self.lin1.bias = nn.Parameter(beta.reshape(()), requires_grad=True)


        self.lin2 = nn.Linear(1, 2, bias=True, dtype=dtype).to(device)
        self.lin2.weight = nn.Parameter(
            torch.stack([-A, -A]).reshape(2, 1),
            requires_grad=True
        )
        self.lin2.bias = nn.Parameter(
            torch.stack([B1, B2]),
            requires_grad=True
        )

        self.lin3 = nn.Linear(2, 1, bias=False, dtype=dtype).to(device)
        self.lin3.weight = nn.Parameter(
            torch.stack([S, -S]).reshape(1, 2),
            requires_grad=True
        )

        self.diff = w*A*S

    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        x = x.unsqueeze(-1)

        g1 = F.relu(self.lin1(x))
        h = F.relu(self.lin2(g1))
        y = F.relu(self.lin3(h))
        return y


class IndicatorLEZ(nn.Module):


    def __init__(self, z, p: int, q: int, device='cpu', dtype=torch.float32,
                 strict_k_range: bool = True, mode='positive'):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.p = int(p)
        self.q = int(q)

        z = torch.as_tensor(z, device=device, dtype=dtype)

        e_min = -(2 ** (q - 1)) + 2
        e_max =  (2 ** (q - 1)) - 1
        e_0   =  (2 ** (q - 2)) - 1

        self.e_min = e_min
        self.e_max = e_max
        self.e_0 = e_0

        # Decompose z = a_z * 2^{e_z}, a_z in [1,2)
        a_z, e_z = fp_decompose_positive(z)

        # c_z = max(e_min - e_z, 0)
        c_z = max(e_min - e_z, 0)

        # u0
        if e_z >= e_min:
            u0 = pow2_int(-p, device=device, dtype=dtype)
        else:
            u0 = pow2_int(-p + c_z, device=device, dtype=dtype)

        # Detect a_z == 1 exactly
        a_is_one = bool((a_z == torch.ones_like(a_z)).item())

        # \tilde{u}_0 and \tilde{c}
        if (not a_is_one) or (e_z == -p + e_min):
            tilde_u0 = torch.zeros((), device=device, dtype=dtype)
            tilde_c = 0
        else:
            tilde_u0 = u0 * pow2_int(-1, device=device, dtype=dtype)
            tilde_c = 1

        # k
        if 0 < e_z <= (-2 - p + e_max):
            k = -p + e_0
        elif e_z <= 0:
            k = 3 - p - e_0 - c_z
        else:
            if strict_k_range:
                raise ValueError(
                    f"Your definition does not specify k for e_z={e_z} "
                    f"with p={p}, q={q}. "
                    f"This is the uncovered case e_z > -2-p+e_max = {-2-p+e_max}."
                )
            k = -p + e_0

        self.e_z = e_z
        self.c_z = c_z
        self.k = k
        self.a_z = a_z
        self.u0 = u0
        self.tilde_u0 = tilde_u0
        self.tilde_c = tilde_c

        # Constants appearing in the LEQ construction
        two_to_ez = pow2_int(e_z, device=device, dtype=dtype)
        A = pow2_int(p - e_z + k, device=device, dtype=dtype)
        B1 = (a_z - torch.tensor(1.0, device=device, dtype=dtype) + u0) * pow2_int(p + k, device=device, dtype=dtype)
        B2 = (a_z - torch.tensor(1.0, device=device, dtype=dtype))      * pow2_int(p + k, device=device, dtype=dtype)
        S = pow2_int(-c_z - k, device=device, dtype=dtype)

        # Layer 1:
        # g2(x) = ReLU(x - 2^{e_z})
        self.lin1 = nn.Linear(1, 1, bias=True, dtype=dtype).to(device)
        if mode == 'positive':
          w1 = 1.0
        elif mode == 'negative':
          w1 = -1.0


        self.lin1.weight = nn.Parameter(
            torch.tensor([[w1]], device=device, dtype=dtype),
            requires_grad=True
        )
        self.lin1.bias = nn.Parameter(
            (-two_to_ez).reshape(()),
            requires_grad=True
        )

        # Layer 2:
        # zeta_2,1(x) = (-A * g2(x)) + B1
        # zeta_2,2(x) = (-A * g2(x)) + B2
        self.lin2 = nn.Linear(1, 2, bias=True, dtype=dtype).to(device)
        self.lin2.weight = nn.Parameter(
            torch.stack([-A, -A]).reshape(2, 1),
            requires_grad=True
        )
        self.lin2.bias = nn.Parameter(
            torch.stack([B1, B2]),
            requires_grad=True
        )

        # Layer 3:
        # f2(x) = S * h1(x) - S * h2(x)
        self.lin3 = nn.Linear(2, 1, bias=False, dtype=dtype).to(device)
        self.lin3.weight = nn.Parameter(
            torch.stack([S, -S]).reshape(1, 2),
            requires_grad=True
        )

        self.diff = w1*A*S

    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        x = x.unsqueeze(-1)

        g2 = F.relu(self.lin1(x))
        h = F.relu(self.lin2(g2))
        y = F.relu(self.lin3(h))
        return y



class IndicatorGEZFull(nn.Module):
    def __init__(self, z, p=23, q=8, device='cpu', dtype=torch.float32):
        super().__init__()

        self.device = device
        self.dtype = dtype

        z = torch.as_tensor(z, device=device, dtype=dtype)
        self.z = z

        if z > 0:
            self.mode = "positive"
            self.model = IndicatorGEZ(z, p=p, q=q, device=device, dtype=dtype)

        else:
            self.mode = "negative"
            self.model = IndicatorLEZ(-z, p=p, q=q, device=device, dtype=dtype , mode="negative")

        self.diff = self.model.diff


    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        out = self.model(x)
        return out



class IndicatorLEZFull(nn.Module):
    def __init__(self, z, p=23, q=8, device='cpu', dtype=torch.float32):
        super().__init__()

        self.device = device
        self.dtype = dtype

        z = torch.as_tensor(z, device=device, dtype=dtype)
        self.z = z

        if z > 0:
            self.mode = "positive"
            self.model = IndicatorLEZ(z, p=p, q=q, device=device, dtype=dtype)

        else:
            self.mode = "negative"
            self.model = IndicatorGEZ(-z, p=p, q=q, device=device, dtype=dtype , mode="negative")

        self.diff = self.model.diff

    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        out = self.model(x)
        return out

class IndicatorLEZFull(nn.Module):
    def __init__(self, z, p=23, q=8, device='cpu', dtype=torch.float32):
        super().__init__()

        self.device = device
        self.dtype = dtype

        z = torch.as_tensor(z, device=device, dtype=dtype)
        self.z = z

        if z > 0:
            self.mode = "positive"
            self.model = IndicatorLEZ(z, p=p, q=q, device=device, dtype=dtype)

        else:
            self.mode = "negative"
            self.model = IndicatorGEZ(-z, p=p, q=q, device=device, dtype=dtype , mode="negative")

        self.diff = self.model.diff

    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        out = self.model(x)
        return out

class ExactIndicator(nn.Module):


    def __init__(self,z, p = 23, q=8, e_max=127 , device='cpu', dtype=torch.float32):
        super().__init__()


        self.device = device
        self.dtype = dtype
        self.p = int(p)
        self.q = int(q)
        self.e_max = e_max

        omega = smallest_positive_subnormal(p=p,q=q,device=device,dtype=dtype)


        z = torch.as_tensor(z, device=device, dtype=dtype)
        z_plus = next_float(z )[0]
        z_minus = prev_float(z)[0]

        self.z_value = float(z.item())



        self.left = IndicatorGEZFull(
            z=z,
            p=p,
            q=q,
            device=device,
            dtype=dtype,
        )
        self.right = IndicatorGEZFull(
            z=z_plus,
            p=p,
            q=q,
            device=device,
            dtype=dtype,
        )

        self.lin_out = nn.Linear(2, 1, bias=False, dtype=dtype).to(device)
        self.lin_out.weight = nn.Parameter(
            torch.tensor([[1.0, -1.0]], device=device, dtype=dtype),
            requires_grad=True
        )

        self.diff =  - self.right.diff



    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        x = x.unsqueeze(-1)

        y1 = self.left(x)
        y2 = self.right(x)
        h = torch.cat([y1, y2], dim=-1)
        y = F.relu(self.lin_out(h))
        y = y.squeeze()




        return y

class CancellationLinear(nn.Module): # 2 layer
    def __init__(self, p=23, e_max=127, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.p = p
        self.e_max = e_max
        self.dtype = dtype
        self.device = device

        num_pairs = e_max - p + 1
        in_features = 1 + 2 * num_pairs

        self.linear = nn.Linear(in_features, 1, bias=False, dtype=dtype, device=device)

        self.linear2 =  nn.Linear(
            1, 1, bias=False, dtype=dtype, device=device
        )

        weight = [1.0]
        for _ in range(num_pairs):
            weight.extend([1.0, 1.0])

        weight = torch.tensor([weight], dtype=dtype, device=device)
        weight2 = torch.tensor([2**(p-e_max)], dtype=dtype, device=device)

        self.linear.weight = nn.Parameter(weight, requires_grad=True)
        self.linear2.weight = nn.Parameter(weight2, requires_grad=True)

    def forward(self, x):
        x = x.to(dtype=self.dtype, device=self.device)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))

        return x

class TripleCancellation(nn.Module): # 6 layer
    def __init__(self, p=23, e_max=127, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.p = p
        self.e_max = e_max
        self.dtype = dtype
        self.device = device

        self.c1 = CancellationLinear(p=p, e_max=e_max, dtype=dtype, device=device)
        self.c2 = CancellationLinear(p=p, e_max=e_max, dtype=dtype, device=device)
        self.c3 = CancellationLinear(p=p, e_max=e_max, dtype=dtype, device=device)

    def forward(self, x0):
        x1 = build_input_vector(x0, p=self.p, e_max=self.e_max,
                                dtype=self.dtype, device=self.device)
        y1 = self.c1(x1).squeeze()

        x2 = build_input_vector(y1, p=self.p, e_max=self.e_max,
                                dtype=self.dtype, device=self.device)
        y2 = self.c2(x2).squeeze()

        x3 = build_input_vector(y2, p=self.p, e_max=self.e_max,
                                dtype=self.dtype, device=self.device)
        y3 = self.c3(x3).squeeze()

        return y3

class ZeroGradIndicator(nn.Module): # 11 layer
    def __init__(self, z, value, p=23, e_max=127, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.indicator = ExactIndicator(z=z.item(), device=device, dtype=dtype)    # 4 layer
        self.cancel = TripleCancellation(                                           # 6 layer
            p=p, e_max=e_max, dtype=dtype, device=device
        )

        self.linear = nn.Linear(
            1, 1, bias=False, dtype=dtype, device=device
         )

        weight = torch.tensor(value[None], dtype=dtype, device=device)
        self.linear.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        z = self.indicator(x)     # shape (N,1)           # 4 layer
        z_scalar = z.reshape(-1)[0]
        y = self.cancel(z_scalar)                         # 6 layer
        y = y.unsqueeze(0)
        y = self.linear(y)                                # 1 layer
        return y


class GradIndicator(nn.Module):
    def __init__(self, z, value, p=23, e_max=127, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.device = device
        self.dtype = dtype



        self.indicator = ExactIndicator(z=z.item(), device=device, dtype=dtype)  # 4 layer


        v = (1/self.indicator.diff.item()) * value.item()
        v = torch.tensor(v, dtype=dtype, device=device)
        self.zerograd_indicator = ZeroGradIndicator(z, - v ,  dtype=dtype, device=device).to(device)


        one = torch.tensor(1.0, dtype=dtype, device=device)


        self.linear = nn.Linear(1, 1, bias=False, dtype=dtype, device=device)

        self.linear2 = nn.Linear(
            1, 1, bias=False, dtype=dtype, device=device
         )
        self.linear3 = nn.Linear(2, 1 , bias=False, dtype=dtype, device=device)

        weight = torch.tensor([[1/self.indicator.diff.item()]], dtype=dtype, device=device)
        weight2 = torch.tensor([[value.item()]], dtype=dtype, device=device)
        weight3 = torch.tensor([[1,  1 ]], dtype=dtype, device=device)


        self.linear.weight = nn.Parameter(weight, requires_grad=True)
        self.linear2.weight = nn.Parameter(weight2, requires_grad=True)
        self.linear3.weight = nn.Parameter(weight3, requires_grad=True)



    def forward(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        y = self.indicator(x)     # shape (N,1)
        y = y.unsqueeze(0)
        z = F.relu(self.linear(y))
        z1 = F.relu(self.linear2(z))

        z2 = self.zerograd_indicator(x)
        z2= z2.unsqueeze(0)


        z = torch.cat([z1, z2], dim=-1)
        z = self.linear3(z)


        self.z1 = z1
        self.z2 = z2
        return z