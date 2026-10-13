# -*- coding: utf-8 -*-
"""
Gera painel.pdf a partir de experimentos.csv e logs/.

    python painel.py [--instavel <id da execução>]

Páginas:
  1. os sinais alinhados no mesmo eixo de passos (perda, ||g||, ||theta||, eta_t,
     razão de atualização) para a melhor configuração, a baseline, a melhor com
     Muon e uma instável. Quando a melhor já é com Muon, o lugar da "melhor com
     Muon" é ocupado pela melhor SEM Muon, para comparação
  2. sinais por camada (primeira, do meio, última): ||g||, ||theta|| e razão de
     atualização (compara o Muon com o AdamW)
  3. métrica de validação x eta, por otimizador (varredura)
  4. agendas: métrica por otimizador x agenda e eta_t de cada agenda
  5. confirmação com 3 sementes (média ± desvio) contra a baseline
"""
import argparse
import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import treino

AQUI = os.path.dirname(os.path.abspath(__file__))
CORES = {"melhor": "#d1495b", "baseline": "#6b6b6b", "muon": "#d1495b", "sem_muon": "#1b6ca8",
         "instavel": "#edae49"}
ROTULO = {"melhor": "melhor", "baseline": "baseline", "muon": "melhor c/ Muon",
          "sem_muon": "melhor sem Muon", "instavel": "instável"}
INSTAVEL_PADRAO = "muon_lr0.03_aw0.0003_warmup_cosseno_p3000_s0"
CORES_OTIM = {"sgd": "#8c564b", "sgd_nesterov": "#e377c2", "adam": "#7f7f7f",
              "adamw": "#1b6ca8", "nadam": "#2ca02c", "muon": "#d1495b"}
ROTULO_CAMADA = {treino.CAMADAS[0]: "primeira", treino.CAMADAS[1]: "meio",
                 treino.CAMADAS[2]: "última (saída)"}


def ler_experimentos():
    with open(treino.CSV_EXPERIMENTOS) as fh:
        linhas = list(csv.DictReader(fh))
    for r in linhas:
        r["lr"] = float(r["lr"])
        r["acc_val"] = float(r["acc_val"])
        r["divergiu"] = r["divergiu"] == "True"
        r["semente"] = int(r["semente"])
    return linhas


def ler_log(rid):
    with open(os.path.join(treino.DIR_LOGS, rid + ".csv")) as fh:
        linhas = list(csv.DictReader(fh))
    cols = linhas[0].keys() if linhas else []
    return {c: np.array([float(l[c]) if l.get(c) not in ("", None) else np.nan
                         for l in linhas]) for c in cols}


def config(r):
    return (r["otimizador"], r["lr"], r["lr_adamw"], r["agenda"])


def nome_curto(r):
    s = f"{r['otimizador']} η={r['lr']:g}"
    if r["otimizador"] == "muon":
        s += f" (AdamW {float(r['lr_adamw']):g})"
    return s + f", {r['agenda']}"


def agrupar_sementes(linhas):
    g = {}
    for r in linhas:
        if r["etapa"] == "teste":
            continue
        g.setdefault(config(r), {})[r["semente"]] = r
    return g


def escolher(linhas, instavel=None):
    g = agrupar_sementes(linhas)
    com3 = {c: rs for c, rs in g.items() if len(rs) >= 3}
    media = lambda rs: statistics.mean(r["acc_val"] for r in rs.values())
    base = g[("adam", 1e-3, "", "constante")]
    melhor = max(com3, key=lambda c: media(com3[c]))
    muons = [c for c in g if c[0] == "muon"]
    melhor_muon = max(muons, key=lambda c: (len(g[c]) >= 3, media(g[c])))

    if instavel is None:
        # instável: a execução de maior "espiculosidade" de ||g|| (max/mediana) entre as
        # que treinaram ao menos 10 registros; empate favorece as que divergiram
        cand = []
        for r in linhas:
            if r["etapa"] == "teste":
                continue
            lg = ler_log(r["id"])
            gr = lg.get("grad", np.array([]))
            gr = gr[np.isfinite(gr)]
            if len(gr) >= 10:
                cand.append((float(np.max(gr) / np.median(gr)) * (2 if r["divergiu"] else 1),
                             r["id"]))
        instavel = max(cand)[1]
    rinst = next(r for r in linhas if r["id"] == instavel)
    esc = dict(melhor=base_or(g[melhor]), baseline=base_or(base))
    if melhor[0] == "muon":
        # a melhor já é com Muon: no lugar dela, mostra a melhor SEM Muon para comparação
        sem = max((c for c in com3 if c[0] != "muon" and c != ("adam", 1e-3, "", "constante")),
                  key=lambda c: media(com3[c]))
        esc["sem_muon"] = base_or(g[sem])
    else:
        esc["muon"] = base_or(g[melhor_muon])
    esc["instavel"] = rinst
    return esc


def base_or(rs):
    return rs[min(rs)]  # execução da semente 0 representa a configuração nos gráficos


def pagina_sinais(pdf, esc):
    fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    for k, r in esc.items():
        lg = ler_log(r["id"])
        rot = ROTULO[k] + f": {nome_curto(r)}"
        c = CORES[k]
        p = lg["passo"]
        ax[0].plot(p, lg["perda_media"], color=c, lw=1.4, label=rot)
        m = np.isfinite(lg.get("perda_val", np.full_like(p, np.nan)))
        if m.any():
            ax[0].plot(p[m], lg["perda_val"][m], "o--", color=c, ms=3, lw=0.8)
        ax[1].plot(p, lg["grad"], color=c, lw=1.0)
        ax[2].plot(p, lg["theta"], color=c, lw=1.4)
        ax[3].plot(p, lg["lr"], color=c, lw=1.4)
        if "lr_adamw" in lg:
            ax[3].plot(p, lg["lr_adamw"], ":", color=c, lw=1.2)
        ax[4].plot(p, lg[f"rho/{treino.CAMADAS[1]}"], color=c, lw=1.0)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("perda\n(linha: treino, média de K passos;\npontos: validação)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("‖∇θ ℒ‖₂ (antes do clipping)")
    ax[2].set_ylabel("‖θ‖₂")
    ax[3].set_yscale("log")
    ax[3].set_ylabel("η_t\n(pontilhado: parte AdamW do Muon)")
    ax[3].set_ylim(1e-6, 1e-1)
    ax[4].set_yscale("log")
    ax[4].axhline(1e-3, color="k", lw=0.6, ls="--")
    ax[4].set_ylabel("‖Δθ‖/‖θ‖\ncamada do meio")
    ax[4].set_ylim(1e-6, 1e-1)
    ax[4].set_xlabel("passo")
    ax[0].legend(fontsize=8, loc="upper right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Sinais do treino alinhados no mesmo eixo de passos "
                 f"(orçamento: {treino.PASSOS} passos, lote {treino.LOTE})", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pagina_camadas(pdf, esc):
    ks = list(esc)
    fig, ax = plt.subplots(3, len(ks), figsize=(16, 10), sharex=True)
    for j, k in enumerate(ks):
        r = esc[k]
        lg = ler_log(r["id"])
        p = lg["passo"]
        for n, cor in zip(treino.CAMADAS, ("#1b9e77", "#d95f02", "#7570b3")):
            ax[0, j].plot(p, lg[f"grad/{n}"], color=cor, lw=1, label=ROTULO_CAMADA[n])
            ax[1, j].plot(p, lg[f"theta/{n}"], color=cor, lw=1.2)
            ax[2, j].plot(p, lg[f"rho/{n}"], color=cor, lw=1)
        ax[0, j].set_title(f"{ROTULO[k]}\n{nome_curto(r)}", fontsize=9)
        ax[0, j].set_yscale("log")
        ax[2, j].set_yscale("log")
        ax[2, j].axhline(1e-3, color="k", lw=0.6, ls="--")
        ax[2, j].set_ylim(bottom=1e-6)
        ax[2, j].set_xlabel("passo")
        for a in ax[:, j]:
            a.grid(alpha=0.3)
    ax[0, 0].set_ylabel("‖g‖ da camada")
    ax[1, 0].set_ylabel("‖θ‖ da camada")
    ax[2, 0].set_ylabel("razão de atualização\n‖Δθ‖/‖θ‖ (ref. 1e-3)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Sinais por camada (primeira, do meio, última)", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pagina_eta(pdf, linhas):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax in axs:
        for o in treino.OTIMIZADORES:
            rs = sorted((r for r in linhas if r["etapa"] == "varredura"
                         and r["otimizador"] == o), key=lambda r: r["lr"])
            if not rs:
                continue
            ax.plot([r["lr"] for r in rs], [r["acc_val"] for r in rs], "o-",
                    color=CORES_OTIM[o], label=o + (" (η das matrizes)" if o == "muon" else ""))
            for r in rs:
                if r["divergiu"]:
                    ax.plot(r["lr"], r["acc_val"], "x", color=CORES_OTIM[o], ms=12, mew=2)
        ax.set_xscale("log")
        ax.set_xlabel("η (pico da agenda warmup + cosseno)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("acurácia de validação")
    axs[0].set_title("Métrica × η por otimizador (varredura, semente 0; × = divergiu)")
    axs[0].legend()
    axs[1].set_ylim(0.82, 0.895)
    axs[1].set_title("Zoom na região útil (melhores η)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pagina_agendas(pdf, linhas):
    rs = [r for r in linhas if r["semente"] == 0 and r["etapa"] in ("agendas", "varredura")]
    otims = sorted({r["otimizador"] for r in linhas if r["etapa"] == "agendas"})
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[3, 2]))
    larg = 0.8 / len(treino.AGENDAS)
    cores_ag = dict(zip(treino.AGENDAS, ("#999999", "#66a61e", "#1b6ca8", "#e7298a")))
    for i, o in enumerate(otims):
        lr_o = next(r["lr"] for r in linhas if r["etapa"] == "agendas" and r["otimizador"] == o)
        for j, ag in enumerate(treino.AGENDAS):
            r = next((r for r in rs if r["otimizador"] == o and r["agenda"] == ag
                      and r["lr"] == lr_o), None)
            if r is None:
                continue
            ax[0].bar(i + (j - 1.5) * larg, r["acc_val"], larg, color=cores_ag[ag],
                      label=ag if i == 0 else None)
            ax[0].text(i + (j - 1.5) * larg, r["acc_val"] + 0.001, f"{r['acc_val']:.4f}",
                       ha="center", fontsize=7, rotation=90)
    ax[0].set_xticks(range(len(otims)))
    ax[0].set_xticklabels([o for o in otims])
    accs = [r["acc_val"] for r in linhas if r["etapa"] == "agendas"]
    ax[0].set_ylim(min(accs) - 0.01, max(accs) + 0.01)
    ax[0].set_ylabel("acurácia de validação (semente 0)")
    ax[0].set_title("Agendas com o melhor η de cada otimizador")
    ax[0].legend()
    T = treino.PASSOS
    for ag in treino.AGENDAS:
        f = treino.fator_agenda(ag, T)
        ax[1].plot(range(T), [f(t) for t in range(T)], color=cores_ag[ag], label=ag)
    ax[1].set_xlabel("passo")
    ax[1].set_ylabel("η_t / η")
    ax[1].set_title("Formato das agendas")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pagina_confirmacao(pdf, linhas):
    g = agrupar_sementes(linhas)
    com3 = {c: rs for c, rs in g.items() if len(rs) >= 3}
    ordem = sorted(com3, key=lambda c: -statistics.mean(r["acc_val"] for r in com3[c].values()))
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[4, 3]))
    for ax, cs in ((axs[0], ordem), (axs[1], [c for c in ordem
                                             if c != ("adam", 1e-3, "", "constante")])):
        for i, c in enumerate(cs):
            accs = [r["acc_val"] for r in com3[c].values()]
            m, s = statistics.mean(accs), statistics.stdev(accs)
            cor = (CORES["baseline"] if c == ("adam", 1e-3, "", "constante")
                   else CORES_OTIM[c[0]])
            ax.errorbar(i, m, yerr=s, fmt="o", color=cor, capsize=6, ms=8)
            ax.scatter([i] * len(accs), accs, color=cor, alpha=0.4, s=18)
            ax.text(i + 0.08, m, f"{m:.4f} ± {s:.4f}", fontsize=9, va="center")
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([nome_curto(next(iter(com3[c].values()))).replace(", ", "\n")
                            for c in cs], fontsize=8)
        ax.set_xlim(-0.4, len(cs) - 0.2)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("acurácia de validação (3 sementes)")
    axs[0].set_title("Confirmação: média ± desvio-padrão entre sementes (pontos = sementes)")
    axs[1].set_title("Zoom nas 3 melhores combinações")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instavel", default=INSTAVEL_PADRAO,
                    help="id da execução instável; 'auto' escolhe a de maior max/mediana de ||g||")
    a = ap.parse_args()
    linhas = ler_experimentos()
    esc = escolher(linhas, None if a.instavel == "auto" else a.instavel)
    for k, r in esc.items():
        print(f"{k:9s} {r['id']}  acc_val={r['acc_val']}  divergiu={r['divergiu']}")
    with PdfPages(os.path.join(AQUI, "painel.pdf")) as pdf:
        pagina_sinais(pdf, esc)
        pagina_camadas(pdf, esc)
        pagina_eta(pdf, linhas)
        pagina_agendas(pdf, linhas)
        pagina_confirmacao(pdf, linhas)


if __name__ == "__main__":
    main()
