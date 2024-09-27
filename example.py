from mamba_block import CrossBlock as Cross_Block
from cross_mamba_simple import Mamba as Cross_Mamba
from mamba_block import Block as Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from functools import partial
from torch import nn
import torch

class cross_mamba(nn.Module):
    def __init__(self, d_model = 256, d_state = 16):
        super(cross_mamba, self).__init__()
        self.mamba = Cross_Block(
                    d_model,
                    mixer_cls=partial(Cross_Mamba, d_state=d_state, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
        
        self.mamba_bw = Cross_Block(
                    d_model,
                    mixer_cls=partial(Cross_Mamba, d_state=d_state, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
        
    def forward(self, _q, _v):
        for_residual = None
        forward_f, for_residual = self.mamba(_q, _v, for_residual, 
                                             inference_params=None)
        forward_f = (forward_f + for_residual) 
        
        back_residual = None
        backward_q = torch.flip(_q, [1])
        backward_v = torch.flip(_v, [1])
        backward_f, back_residual = self.mamba_bw(backward_q, backward_v, 
                                                  back_residual, inference_params=None)
        backward_f = (backward_f + back_residual) if back_residual is not None else backward_f

        backward_f = torch.flip(backward_f, [1])
        forward_f = forward_f + backward_f
        return forward_f
        
class self_mamba(nn.Module):
    def __init__(self, d_model = 256, d_state = 16):
        super(self_mamba, self).__init__()
        self.mamba = Block(
                    d_model,
                    mixer_cls=partial(Mamba, d_state=d_state, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
        self.mamba_bw = Block(
                    d_model,
                    mixer_cls=partial(Mamba, d_state=d_state, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
        
    def forward(self, x):
        for_residual = None
        forward_f, for_residual = self.mamba(x, for_residual, 
                                             inference_params=None)
        forward_f = (forward_f + for_residual)
        
        back_residual = None
        backward_x = torch.flip(x, [1])
        backward_f, back_residual = self.mamba_bw(backward_x, 
                                                  back_residual, inference_params=None)
        backward_f = (backward_f + back_residual) if back_residual is not None else backward_f

        backward_f = torch.flip(backward_f, [1])
        forward_f = forward_f + backward_f
        return forward_f
    
if __name__ == '__main__':
    CMModel = cross_mamba().cuda()
    _q = torch.randn((4, 100, 256)).cuda()
    _v = torch.randn((4, 100, 256)).cuda()
    out_cm = CMModel(_q, _v)
    print(out_cm.shape)
    
    SMModel = self_mamba().cuda()
    _v = torch.randn((4, 100, 256)).cuda()
    out_sm = SMModel(_v)
    print(out_sm.shape)