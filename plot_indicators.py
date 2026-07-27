import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import ZeroGradIndicator, GradIndicator
from src.utils import prev_float, next_float

def main():
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    n_trials = 1000

    z = torch.rand(1, dtype=dtype, device=device)
    value = torch.rand(1, dtype=dtype, device=device)
    
    value_indicator = ZeroGradIndicator(z,value,  dtype=dtype, device=device)
    grad_indicator = GradIndicator(z, value, dtype=dtype, device=device)

    xs = torch.cat([
        z,
        prev_float(z),
        next_float(z),
        torch.tensor([0.0, 1.0], dtype=dtype, device=device),
        torch.rand(n_trials, dtype=dtype, device=device),
    ])


    print(f'Verifing value indicator with z={z.item():.4f}:')
    ys, gs = [], []
    trial = len(xs)
    okay = 0
    fail = 0
    for xval in xs:
        x = xval.view(1).detach().clone().requires_grad_(True)
        y = value_indicator(x)
        g = torch.autograd.grad(y, x)[0]
        
        # print(f'x={x.item():.4f} , v={y.item():.4f} , g={g.item():.4f} ' )
        if x != z :
            if y==0 and g == 0:
                # print(' verification pass.')
                okay += 1
            else:
                # print(' verification fail.')
                fail += 1
        if x == z :
            if y==1 and g == 0:
                # print(' verification pass.')
                okay += 1
            else:
                # print(' verification fail.')
                fail += 1                
        ys.append(y.item())
        gs.append(g.item())
    xs_np = xs.detach().cpu().numpy()

    
    print(f'  pass / trials {okay}/{okay+fail}')

    

    # Plot Values
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'Value Indicator with z={z.item():.4f}), value={value.item():.4f})')
    plt.scatter(xs_np, ys, label='value')
    plt.scatter(xs_np, gs, label='gradient')
    plt.legend()
    
    print(f'Verifing gradient indicator with z={z.item():.4f}:')
    ys, gs = [], []
    trial = len(xs)
    okay = 0
    fail = 0
    for xval in xs:
        x = xval.view(1).detach().clone().requires_grad_(True)
        y = grad_indicator(x)
        g = torch.autograd.grad(y, x)[0]
        
        # print(f'x={x.item():.4f} , v={y.item():.4f} , g={g.item():.4f} ' )
        if x != z :
            if y==0 and g == 0:
                # print(' verification pass.')
                okay += 1
            else:
                # print(' verification fail.')
                fail +=1
        if x == z :
            if y==0 and g == 1:
                # print(' verification pass.')
                okay += 1
            else:
                # print(' verification fail.')   
                fail += 1
        ys.append(y.item())
        gs.append(g.item())
    xs_np = xs.detach().cpu().numpy()
    
    print(f'  pass / trials {okay}/{okay+fail}')

    # Plot Gradients
    plt.subplot(1, 2, 2)
    plt.title(f'Gradient Indicator with z={z.item():.4f}), value={value.item():.4f})')
    plt.scatter(xs_np, ys, label='value')
    plt.scatter(xs_np, gs, label='gradient', color='g')
    
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()