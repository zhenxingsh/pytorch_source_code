import torch
from .optimizer import Optimizer
class Adagrad(Optimizer):
    def __init__(self,param,lr=1e-2,lr_decay=0,weight_decay=0,initial_accumulator_value=0):
        if not 0.0<=lr:
            raise
        if not 0.0 <=lr_decay:
            raise
        if not 0.0 <=initial_accumulator_value:
            raise
        defaults=dict(lr=lr,lr_decay,weight_decay=weight_decay,
        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad,self).__init__(params,defaults)
        for group in self.param_groups:
            for p in group['params']:
                state=self.state[p]
                state['step']=0
                state['sum']=torch.full_like(p.data,initial_accumulator_value)
    def share_memory(self):
        for group in self.param_groups:
            for p in group['param']:
                state=self.state[p]
                state['sum'].share_memory_()
    def step(self,closure=None):
        loss=None
        if closure is not NOne:
            loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                state=self.state[p]
                state['step']+=1
                if group['weight_decay']!=0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad=grad.add(group['weight_decay'],p.data)
                
                clr=group['lr']/(1+(state['step']-1)*group['lr_decay'])

                if grad.is_sparse:
                    grad=grad.coalesce()
                    grad_indices=grad._indices()
                    grad_values=grad._values()
                    size=grad.size()

                    def make_sparse(values):
                        constructor=grad.new
                        if grad_indices.dim()==0 or values.dim()==0:
                            return constructor().resize_as_(grad)
                        return comstructor(grad_indices,values,size)

                    state['sum'].add_(make_parse(grad_values.pow(2)))
                    std=state['sum'].sparse_mask(grad)
                    std_values=std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr,make_sparse(grad_values/std_values))
                else:
                    state['sum'].addcmul_(1,grad,grad)
                    std=state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr,grad,std)
            return loss
