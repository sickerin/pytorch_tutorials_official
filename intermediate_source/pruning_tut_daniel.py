# %%
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

torch.__version__

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Build LeNet


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.reul(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet().to(device=device)

# %%
module = model.conv1
print(list(module.named_parameters()))

# %%
prune.random_unstructured(module, name='weight', amount=0.3)
print(list(module.named_parameters()))

# %%
print(list(module.named_buffers()))
# %%
print(module._forward_pre_hooks)
# %%
# prune the 3 smallest entries in bias by L1 norm
print(list(module.named_parameters()))
prune.l1_unstructured(module, name='bias', amount=3)

# %%
print(list(module.named_buffers()))
print(module.bias)
# %%
print(module._forward_pre_hooks)
# %%
# interative or combined pruning
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)


# %%
print(module._forward_pre_hooks)

# %%
print(model.state_dict().keys())


# %%
prune.remove(module, 'weight')

# %%
print(list(module.named_buffers()))

# %%
