import torch
from .optimizer import optimizer,required
class SGD(optimizer):
    def __init__(self,params,lr=required,momentum=0,dampening=0,
    weight_decay=0,nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate:{}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value:{}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value:{}".format(weight_decay))
        defaults=dict(lr=lr,momentum=momentum,dampening=dampening,
        )
        if nesterov and (momentum <=0 or dampening!=0):
            raise ValueError("nesterov momentum requires a momentum and zero dampening")
        super(SGD,self).__init__(params,defaults)
    
    def __setstate__(self,state):
        super(SGD,self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov',False)
    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure()
        for group in self.param_groups:
            weight_decay=group['weight_decay']
            momentum=group['momentum']
            dampening=group['dampening']
            nesterov=group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p=p.grad.data
                if weight_decay !=0:
                    d_p.add_(weight_decay,p.data)
                if momentum != 0:
                    param_state=self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf=param_state['momentum_buffer']=torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf=param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1-dampening,d_p)
                    
                if nesterov:
                    d_p=d_p.add(momentum,buf)
                else:
                    d_p=buf
                
                p.data.add_(-group['lr'],d_p)
            return loss