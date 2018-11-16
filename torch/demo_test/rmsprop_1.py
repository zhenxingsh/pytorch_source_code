import torch
from .optimizer import Optimizer
class RMSprop(Optimizer):
    def __init__(self,params,lr=1e-2,alpha=0.99,eps=1e-8,weight_decay=0,momentum=0,centered=False):
        defaults=dict(lr=lr,momentum=momentum,alpha=alpha,eps=eps,centered=centered,weight_decay=weight_decay)
        super(RMSprop,self).__init__(params,defaults)
    
    def __setstate__(self,state):
        super(RMSprop,self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum',0)
            group.setdefault('centered',False)
    
    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                if grad.is_sparse:
                    raise RuntimeError
                state=self.state[p]

                if len(state)==0:
                    state['step']=0
                    state['square_avg']=torch.zeros_like(p.data)
                    if group['momentum']>0:
                        state['momentum_buffer']=borch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg']=torch.zeros_like(p.data)

                square_avg=state['square_avg']
                alpha=group['alpha']
                state['step']+=1
                if group['weight_decay']!=0:
                    grad=grad.add(group['weight_decay'],p.data)
                square_avg.mul_(alpha).addcmul_(1-alpha,grad,grad)

                if group['centered']:
                    grad_avg=state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha,grad)
                    avg=square_avg.addcmul(-1,grad_avg,grad_avg).sqrt().add_(group['eps'])
                else:
                    avg=square_avg.sqrt().add_(group['eps'])
                
                if group['momentum']>0:
                    buf=state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad,avg)
                    p.data.add_(-group['lr'],buf)
                else:
                    p.data.addcdiv_(-group['lr'],grad,avg)
        return loss