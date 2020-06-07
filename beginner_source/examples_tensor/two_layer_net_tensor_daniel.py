# %%
import torch 
dtype = torch.float

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, dtype=dtype)
y = torch.randn(N, D_out, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, dtype=dtype)
w2 = torch.randn(H, D_out, dtype=dtype)
learning_rate = 1e-6

# %% 
for t in range(1):
    #forward
    h = x.mm(w1)
    h_relu = torch.relu(h)
    y_pred = h_relu.mm(w2)
    
    #loss
    loss = (y_pred - y ).pow(2).sum().item()
    if t//100 == 0:
        print(t, loss)  

    #backprop
    grad_y_pred = 2.0 * (y_pred-y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.T.mm(grad_h)

    #update
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    



# %%
# %%
# %%
# %%
# %%
# %%
# %%
