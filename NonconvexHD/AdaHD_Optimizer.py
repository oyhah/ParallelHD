import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from torch.distributions.exponential import Exponential


class AdaHD(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(1, 0.999), eps=1e-8,
         weight_decay=0.0, mean_duration=0.1, mu=0.5, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon parameter : {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= mean_duration:
            raise ValueError("Invalid mean_duration value: {}".format(mean_duration))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, 
                        mean_duration=mean_duration, mu=mu,
                        maximize=maximize)
        self.s = torch.tensor(0., requires_grad=False)
        self.max_S = torch.tensor(0., requires_grad=False)
        self.mean_duration = mean_duration
        self.mu = mu
        self.lr = lr
        self.recharge = False
        
        super(AdaHD, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        if self.s == 0:
            distr = Exponential(rate=torch.tensor([1/self.mean_duration]))
            duration = distr.sample()
            self.max_S = torch.floor(duration / self.lr)
            if self.max_S == 0:
                self.max_S += 1
            
            mu_sample = torch.rand(1)
            if mu_sample < self.mu:
                self.recharge = True
            else:
                self.recharge = False


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            eps = group['eps']
            lr = group['lr']
            beta1, beta2 = group['betas']
            mean_duration = group['mean_duration']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0., requires_grad=False)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.tensor(0., requires_grad=False)
                    state['max_S'] = torch.tensor(0., requires_grad=False)
                
                # Apply weight decay 
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                if self.s == 0:
                    if self.recharge == True:
                        state['exp_avg'] = torch.randn_like(p, memory_format=torch.preserve_format)
#                         state["exp_avg_sq"] = torch.randn_like(p, memory_format=torch.preserve_format)
                    elif self.recharge == False:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                         state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.add_(grad, alpha=1)
                exp_avg_sq.addcmul_(grad, grad, value=1)

                state['step'] += 1

                step = state['step']

                bias_correction2 = 1 - beta2**step
                
                step_size = lr
                bias_correction2_sqrt = bias_correction2.sqrt()

                denom = (exp_avg_sq.sqrt()).add_(eps)

                # Take step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        self.s += 1
        if self.s == self.max_S:
            self.s = torch.tensor(0., requires_grad=False)

        return loss
    
    
class HD(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.0, mean_duration=0.1, mu=0.5, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= mean_duration:
            raise ValueError("Invalid mean_duration value: {}".format(mean_duration))
        
        defaults = dict(lr=lr, weight_decay=weight_decay, 
                        mean_duration=mean_duration, mu=mu,
                        maximize=maximize)
        
        self.s = torch.tensor(0., requires_grad=False)
        self.max_S = torch.tensor(0., requires_grad=False)
        self.mean_duration = mean_duration
        self.mu = mu
        self.lr = lr
        self.recharge = False
        
        super(HD, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        if self.s == 0:
            distr = Exponential(rate=torch.tensor([1/self.mean_duration]))
            duration = distr.sample()
            self.max_S = torch.floor(duration / self.lr)
            if self.max_S == 0:
                self.max_S += 1
            
            mu_sample = torch.rand(1)
            if mu_sample < self.mu:
                self.recharge = True
            else:
                self.recharge = False


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            mean_duration = group['mean_duration']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0., requires_grad=False)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.tensor(0., requires_grad=False)
                    state['max_S'] = torch.tensor(0., requires_grad=False)
                
                # Apply weight decay 
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                if self.s == 0:
                    if self.recharge == True:
                        state['exp_avg'] = torch.randn_like(p, memory_format=torch.preserve_format)
                    elif self.recharge == False:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                exp_avg = state['exp_avg']

                exp_avg.add_(grad, alpha=1)

                state['step'] += 1
                
                step_size = lr 

                # Take step
                p.data.add_(exp_avg, alpha=-step_size)
        
        self.s += 1
        if self.s == self.max_S:
            self.s = torch.tensor(0., requires_grad=False)

        return loss