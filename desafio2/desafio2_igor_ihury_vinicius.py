import math
import torch
import torch.nn.functional as F

def ativacao(x: torch.Tensor) -> torch.Tensor:
    #LeakyReLU - Mantem o fluxo linear (evita explosões severas) e não zera os negativos
    return F.leaky_relu(x, negative_slope=0.1)

#Calibração via Monte Carlo da LeakyReLU
_g = torch.Generator().manual_seed(42)
_z = torch.randn(2_000_000, generator=_g)
_E_f2 = F.leaky_relu(_z, negative_slope=0.1).pow(2).mean().item()

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor, 
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    
    #Mantemos a aplicação direta da calibração para todas as matrizes
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
        
    #Fator de amortecimento de 0.5 na última camada: 
    #Garante um início suave na entropia cruzada sem matar o gradiente de retorno
    if camada == n_camadas:
        desvio *= 0.5

    W.normal_(0.0, desvio)
    b.zero_()
