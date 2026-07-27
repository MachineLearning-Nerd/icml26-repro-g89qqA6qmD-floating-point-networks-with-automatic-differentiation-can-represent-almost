import torch

def next_float(x):
    inf_tensor = torch.tensor([float('inf')], device=x.device)
    return torch.nextafter(x, inf_tensor)

def prev_float(x):
    inf_tensor = torch.tensor([float('-inf')], device=x.device)
    return torch.nextafter(x, inf_tensor)

def pow2_int(n: int, *, device, dtype):
    """Return 2^n as a scalar tensor in the target dtype/device."""
    base = torch.ones((), device=device, dtype=dtype)
    exp = torch.tensor(int(n), device=device, dtype=torch.int64)
    return torch.ldexp(base, exp)

def fp_decompose_positive(z: torch.Tensor):
    """
    Decompose z = a_z * 2^{e_z}, with a_z in [1, 2).
    Assumes z > 0.
    """
    if z.ndim != 0:
        raise ValueError("z must be a scalar tensor.")
    if not torch.isfinite(z):
        raise ValueError("z must be finite.")
    if z <= 0:
        raise ValueError("This construction assumes z > 0.")

    m, e = torch.frexp(z)   # z = m * 2^e, m in [0.5, 1)
    a_z = m * 2             # in [1, 2)
    e_z = int(e.item()) - 1
    return a_z, e_z

def smallest_positive_subnormal(p: int, q: int, *, device, dtype):
    """
    Smallest positive subnormal in F_{p,q}:
        omega = 2^{e_min - p + 1}
    """
    e_min = -(2 ** (q - 1)) + 2
    exp = e_min - p + 1
    base = torch.ones((), device=device, dtype=dtype)
    return torch.ldexp(base, torch.tensor(exp, device=device, dtype=torch.int64))

def build_input_vector(x, p=23, e_max=127, dtype=torch.float32, device="cpu"):
    values = [x.reshape(1)]  # keeps grad connection to x
    consts = []
    
    for k in range(p, e_max + 1):
        k_tensor = torch.tensor(k, dtype=torch.int32, device=device)
        a = torch.ldexp(torch.tensor(1.0, dtype=dtype, device=device), k_tensor)
        a_plus = next_float(a)

        consts.append(a_plus.reshape(1))
        consts.append((-a_plus).reshape(1))

    values = values + consts
    return torch.cat(values).unsqueeze(0)