# -*- coding: utf-8 -*-
"""
Desafio 3 — O melhor treino para o seu problema
GBC073 — Inteligência Computacional (FACOM/UFU)

Discentes:
Ihury Kewin de Oliveira Costa
Igor Oliveira Souza e Silva
Vinicius Resende Garcia

Problema: a rede profunda do Desafio 2 — MLP com 48 camadas lineares, SEM
normalização, ativação LeakyReLU(0.1) e inicialização calibrada por Monte Carlo
(importadas de ../desafio2/desafio2_igor_ihury_vinicius.py) — treinada em
Fashion-MNIST (784 -> 128 x 46 -> 10).

Uma execução = (otimizador, eta, agenda, semente). Exemplo:
    python treino.py --otimizador muon --lr 0.02 --lr-adamw 1e-3 \
                     --agenda warmup_cosseno --semente 0

Cada execução grava logs/<id>.csv com os sinais (a cada K passos) e acrescenta
uma linha em experimentos.csv.
"""
import argparse
import csv
import gzip
import importlib.util
import math
import os
import struct
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_DADOS = os.path.join(AQUI, "dados")
DIR_LOGS = os.path.join(AQUI, "logs")
CSV_EXPERIMENTOS = os.path.join(AQUI, "experimentos.csv")

# --- Desafio 2: ativação e inicialização do grupo ---------------------------
_spec = importlib.util.spec_from_file_location(
    "desafio2", os.path.join(AQUI, "..", "desafio2", "desafio2_igor_ihury_vinicius.py"))
desafio2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(desafio2)

# --- Orçamento fixo (igual para todas as comparações) -----------------------
N_CAMADAS = 48          # camadas lineares (como no Desafio 2)
LARGURA = 128
LOTE = 128
PASSOS = 3000
K_LOG = 20              # registra os sinais a cada K passos (10 <= K <= 50)
VAL_A_CADA = 100        # perda/acurácia de validação periódica
N_VAL = 5000            # validação separada do treino; teste usado uma vez
SEMENTE_DADOS = 12345   # divisão treino/validação fixa para todas as execuções
LIMIAR_DIVERGENCIA = 100.0

# camadas instrumentadas: primeira, uma do meio e última
CAMADAS = ("camadas.0.weight", f"camadas.{(N_CAMADAS - 1) // 2}.weight", "head.weight")

# hiperparâmetros fixos por otimizador (só eta e agenda são varridos)
WD_PADRAO = {"sgd": 0.0, "sgd_nesterov": 0.0, "adam": 0.0, "adamw": 0.1,
             "nadam": 0.0, "muon": 0.01}
WD_ADAMW_AUX = 0.1      # parte AdamW do Muon (vieses e camada de saída)

URL_FM = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
ARQS_FM = {"xtr": "train-images-idx3-ubyte.gz", "ytr": "train-labels-idx1-ubyte.gz",
           "xte": "t10k-images-idx3-ubyte.gz", "yte": "t10k-labels-idx1-ubyte.gz"}


# =============================================================================
# Dados
# =============================================================================

def _ler_idx(caminho):
    with gzip.open(caminho, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        forma = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).reshape(forma)


def carregar_dados():
    """Fashion-MNIST: 55k treino / 5k validação (fixos) / 10k teste."""
    os.makedirs(DIR_DADOS, exist_ok=True)
    t = {}
    for chave, nome in ARQS_FM.items():
        caminho = os.path.join(DIR_DADOS, nome)
        if not os.path.exists(caminho):
            urllib.request.urlretrieve(URL_FM + nome, caminho)
        t[chave] = _ler_idx(caminho)
    xtr = t["xtr"].reshape(-1, 784).float() / 255.0
    xte = t["xte"].reshape(-1, 784).float() / 255.0
    ytr, yte = t["ytr"].long(), t["yte"].long()

    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(SEMENTE_DADOS))
    iv, it = perm[:N_VAL], perm[N_VAL:]
    xval, yval, xtr, ytr = xtr[iv], ytr[iv], xtr[it], ytr[it]

    mu, sd = xtr.mean(0), xtr.std(0) + 1e-3   # estatísticas só do treino
    norm = lambda x: (x - mu) / sd
    return (norm(xtr), ytr), (norm(xval), yval), (norm(xte), yte)


# =============================================================================
# Modelo (Desafio 2)
# =============================================================================

class RedeProfunda(nn.Module):
    def __init__(self, d_in=784, largura=LARGURA, n_classes=10, n_camadas=N_CAMADAS):
        super().__init__()
        dims = [d_in] + [largura] * (n_camadas - 1)
        self.camadas = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(largura, n_classes)
        for i, lin in enumerate(list(self.camadas) + [self.head], start=1):
            desafio2.inicializar(lin.weight, lin.bias, lin.in_features, lin.out_features,
                                 i, n_camadas)

    def forward(self, x):
        for lin in self.camadas:
            x = desafio2.ativacao(lin(x))
        return self.head(x)


# =============================================================================
# Muon (implementação própria, ortogonalização por Newton–Schulz)
# =============================================================================

def ortogonaliza(G, passos=5):
    """Newton–Schulz (Jordan, 2024): aproxima U V^T de G = U S V^T.
    Aceita uma matriz (m, n) ou um lote de matrizes (b, m, n)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    T = X.size(-2) > X.size(-1)
    if T:
        X = X.mT
    for _ in range(passos):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if T else X


class Muon(torch.optim.Optimizer):
    """Momento (Nesterov) + passo ortogonalizado U V^T, só para matrizes 2-D ocultas.
    Weight decay desacoplado; escala sqrt(max(1, linhas/colunas)) como em Jordan (2024)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0,
                 ns_passos=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_passos=ns_passos))

    @torch.no_grad()
    def step(self):
        for g_ in self.param_groups:
            por_forma = {}
            for p in g_["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - g_["momentum"])
                d = p.grad.lerp(buf, g_["momentum"]) if g_["nesterov"] else buf
                por_forma.setdefault(tuple(p.shape), []).append((p, d))
            # matrizes de mesma forma são ortogonalizadas juntas (lote), só por velocidade
            for (m, n), itens in por_forma.items():
                U = ortogonaliza(torch.stack([d for _, d in itens]), g_["ns_passos"])
                U *= max(1.0, m / n) ** 0.5
                for (p, _), u in zip(itens, U):
                    p.mul_(1 - g_["lr"] * g_["weight_decay"])
                    p.add_(u, alpha=-g_["lr"])


# =============================================================================
# Otimizadores e agendas
# =============================================================================

OTIMIZADORES = ("sgd", "sgd_nesterov", "adam", "adamw", "nadam", "muon")
AGENDAS = ("constante", "cosseno", "warmup_cosseno", "wsd")


def criar_otimizadores(modelo, nome, lr, lr_adamw, wd):
    ps = list(modelo.parameters())
    if nome == "sgd":
        return [torch.optim.SGD(ps, lr=lr, momentum=0.9, weight_decay=wd)]
    if nome == "sgd_nesterov":
        return [torch.optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)]
    if nome == "adam":
        return [torch.optim.Adam(ps, lr=lr, weight_decay=wd)]
    if nome == "adamw":
        return [torch.optim.AdamW(ps, lr=lr, weight_decay=wd)]
    if nome == "nadam":
        return [torch.optim.NAdam(ps, lr=lr, weight_decay=wd)]
    if nome == "muon":
        muon_p = [p for n, p in modelo.named_parameters() if p.ndim == 2 and "head" not in n]
        ids = {id(p) for p in muon_p}
        adamw_p = [p for p in ps if id(p) not in ids]
        return [Muon(muon_p, lr=lr, momentum=0.95, weight_decay=wd),
                torch.optim.AdamW(adamw_p, lr=lr_adamw, weight_decay=WD_ADAMW_AUX)]
    raise ValueError(nome)


def fator_agenda(nome, passos, frac_warmup=0.05, frac_decaimento=0.2):
    """Multiplicador de eta no passo t (LambdaLR)."""
    w = max(1, int(frac_warmup * passos))
    d = max(1, int(frac_decaimento * passos))

    def constante(t):
        return 1.0

    def cosseno(t):
        return 0.5 * (1 + math.cos(math.pi * min(t, passos) / passos))

    def warmup_cosseno(t):
        if t < w:
            return (t + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min(t - w, passos - w) / (passos - w)))

    def wsd(t):  # warmup–stable–decay (Hägele et al., 2024): decaimento linear nos 20% finais
        if t < w:
            return (t + 1) / w
        if t < passos - d:
            return 1.0
        return max(0.0, (passos - t) / d)

    return {"constante": constante, "cosseno": cosseno,
            "warmup_cosseno": warmup_cosseno, "wsd": wsd}[nome]


# =============================================================================
# Treino instrumentado
# =============================================================================

def id_execucao(cfg):
    s = f"{cfg['otimizador']}_lr{cfg['lr']:g}"
    if cfg["otimizador"] == "muon":
        s += f"_aw{cfg.get('lr_adamw', 1e-3):g}"
    s += f"_{cfg['agenda']}_p{cfg.get('passos', PASSOS)}_s{cfg['semente']}"
    return s + ("_teste" if cfg.get("teste") else "")


def comando(cfg):
    c = (f"python treino.py --otimizador {cfg['otimizador']} --lr {cfg['lr']:g} "
         f"--agenda {cfg['agenda']} --semente {cfg['semente']} --passos {cfg['passos']}")
    if cfg["otimizador"] == "muon":
        c += f" --lr-adamw {cfg['lr_adamw']:g}"
    if cfg.get("teste"):
        c += " --teste"
    return c


@torch.no_grad()
def avaliar(modelo, x, y, lote=2500):
    perda, acertos = 0.0, 0
    for i in range(0, len(x), lote):
        out = modelo(x[i:i + lote])
        perda += F.cross_entropy(out, y[i:i + lote], reduction="sum").item()
        acertos += (out.argmax(1) == y[i:i + lote]).sum().item()
    return perda / len(x), acertos / len(x)


def _norma(ts):
    return torch.sqrt(sum((t.float() ** 2).sum() for t in ts)).item()


def treinar(cfg, dados=None, salvar_log=True):
    """Executa uma configuração e devolve a linha de experimentos.csv."""
    cfg = dict(cfg)
    cfg.setdefault("passos", PASSOS)
    cfg.setdefault("lr_adamw", 1e-3)
    cfg.setdefault("wd", WD_PADRAO[cfg["otimizador"]])
    cfg.setdefault("teste", False)
    cfg.setdefault("etapa", "manual")
    T = cfg["passos"]

    (xtr, ytr), (xval, yval), (xte, yte) = dados if dados is not None else carregar_dados()

    torch.manual_seed(cfg["semente"])
    modelo = RedeProfunda()
    opts = criar_otimizadores(modelo, cfg["otimizador"], cfg["lr"], cfg["lr_adamw"], cfg["wd"])
    f = fator_agenda(cfg["agenda"], T)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in opts]   # a MESMA agenda

    params = dict(modelo.named_parameters())
    g_lote = torch.Generator().manual_seed(1000 + cfg["semente"])
    perm, pos = torch.randperm(len(xtr), generator=g_lote), 0

    log, janela = [], []
    divergiu, passo_div = False, ""
    t0 = time.time()
    for t in range(T):
        if pos + LOTE > len(perm):
            perm, pos = torch.randperm(len(xtr), generator=g_lote), 0
        idx = perm[pos:pos + LOTE]
        pos += LOTE

        loss = F.cross_entropy(modelo(xtr[idx]), ytr[idx])
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        janela.append(loss.item())

        if not math.isfinite(loss.item()) or loss.item() > LIMIAR_DIVERGENCIA:
            divergiu, passo_div = True, t
            log.append(dict(passo=t, perda=loss.item(), perda_media=float("nan")))
            break

        registrar = (t % K_LOG == 0) or (t == T - 1)
        if registrar:
            # depois de backward(), antes do clipping e de opt.step()
            linha = dict(passo=t, perda=loss.item(), perda_media=sum(janela) / len(janela),
                         grad=_norma(p.grad for p in modelo.parameters() if p.grad is not None),
                         theta=_norma(modelo.parameters()),
                         lr=opts[0].param_groups[0]["lr"])
            if len(opts) > 1:
                linha["lr_adamw"] = opts[1].param_groups[0]["lr"]
            for n in CAMADAS:
                linha[f"grad/{n}"] = params[n].grad.norm().item()
                linha[f"theta/{n}"] = params[n].detach().norm().item()
            antes = {n: params[n].detach().clone() for n in CAMADAS}
            janela = []

        for o in opts:
            o.step()
        for s in scheds:
            s.step()

        if registrar:
            for n in CAMADAS:
                linha[f"rho/{n}"] = ((params[n].detach() - antes[n]).norm()
                                     / antes[n].norm()).item()
            if t % VAL_A_CADA == 0 or t == T - 1:
                linha["perda_val"], linha["acc_val"] = avaliar(modelo, xval, yval)
            log.append(linha)

    tempo = time.time() - t0
    if divergiu:
        perda_val, acc_val = float("nan"), 0.0
    else:
        perda_val, acc_val = avaliar(modelo, xval, yval)
        if not math.isfinite(perda_val):
            divergiu, passo_div = True, T
    acc_teste = ""
    if cfg["teste"] and not divergiu:
        acc_teste = round(avaliar(modelo, xte, yte)[1], 5)

    rid = id_execucao(cfg)
    if salvar_log:
        os.makedirs(DIR_LOGS, exist_ok=True)
        cols = []
        for l in log:
            cols += [k for k in l if k not in cols]
        with open(os.path.join(DIR_LOGS, rid + ".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(log)

    perda_tr = [l["perda_media"] for l in log[-5:] if "grad" in l]
    return dict(id=rid, etapa=cfg["etapa"], otimizador=cfg["otimizador"], lr=cfg["lr"],
                lr_adamw=cfg["lr_adamw"] if cfg["otimizador"] == "muon" else "",
                agenda=cfg["agenda"], semente=cfg["semente"], passos=T, lote=LOTE,
                wd=cfg["wd"], acc_val=round(acc_val, 5),
                perda_val=round(perda_val, 5) if math.isfinite(perda_val) else "nan",
                perda_treino_final=round(sum(perda_tr) / len(perda_tr), 5) if perda_tr else "nan",
                acc_teste=acc_teste, divergiu=divergiu, passo_divergencia=passo_div,
                tempo_s=round(tempo, 1), comando=comando(cfg))


CAMPOS_CSV = ["id", "etapa", "otimizador", "lr", "lr_adamw", "agenda", "semente", "passos",
              "lote", "wd", "acc_val", "perda_val", "perda_treino_final", "acc_teste",
              "divergiu", "passo_divergencia", "tempo_s", "comando"]


def anexar_csv(linha, caminho=None):
    caminho = caminho or CSV_EXPERIMENTOS
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_CSV)
        if novo:
            w.writeheader()
        w.writerow(linha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--otimizador", choices=OTIMIZADORES, required=True)
    ap.add_argument("--lr", type=float, required=True,
                    help="eta (no Muon: eta das matrizes ocultas)")
    ap.add_argument("--lr-adamw", type=float, default=1e-3,
                    help="só Muon: eta da parte AdamW (vieses e camada de saída)")
    ap.add_argument("--agenda", choices=AGENDAS, default="warmup_cosseno")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--passos", type=int, default=PASSOS)
    ap.add_argument("--wd", type=float, default=None, help="padrão: WD_PADRAO[otimizador]")
    ap.add_argument("--teste", action="store_true", help="avalia no teste (use uma vez)")
    ap.add_argument("--etapa", default="manual")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    cfg = dict(otimizador=a.otimizador, lr=a.lr, lr_adamw=a.lr_adamw, agenda=a.agenda,
               semente=a.semente, passos=a.passos, teste=a.teste, etapa=a.etapa)
    if a.wd is not None:
        cfg["wd"] = a.wd
    linha = treinar(cfg)
    anexar_csv(linha)
    print({k: linha[k] for k in ("id", "acc_val", "perda_val", "acc_teste", "divergiu",
                                 "tempo_s")})


if __name__ == "__main__":
    main()
