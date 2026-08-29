import torch


def zeropower_via_newtonschulz5(gradient, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.bfloat16() if gradient.is_cuda else gradient.float()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_matrix = x @ x.T
        b_matrix = b * a_matrix + c * a_matrix @ a_matrix
        x = a * x + b_matrix @ x
    if transposed:
        x = x.T
    return x.to(gradient.dtype)


class muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"] != 0:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr * scale)


def build_optimizer(model, train_config):
    seen = set()
    matrix_params = []
    other_params = []
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim == 2 and "embed" not in name:
            matrix_params.append(param)
        else:
            other_params.append(param)

    optimizers = []
    if matrix_params:
        optimizers.append(muon(matrix_params, lr=train_config["muon_learning_rate"], weight_decay=train_config["weight_decay"]))
    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"], betas=(0.9, 0.95)))
    return optimizers

