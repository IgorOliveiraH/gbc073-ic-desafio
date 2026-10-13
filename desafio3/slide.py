# -*- coding: utf-8 -*-
"""
Gera slide.pdf (um slide, 3 minutos): receita vencedora, ganho ± desvio sobre a
baseline e o diagnóstico lido nos sinais que a justifica.

    python slide.py
"""
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import painel
import treino

AQUI = os.path.dirname(os.path.abspath(__file__))

DIAGNOSTICO = "(preencher)"


def main():
    linhas = painel.ler_experimentos()
    g = painel.agrupar_sementes(linhas)
    com3 = {c: rs for c, rs in g.items() if len(rs) >= 3}
    media = lambda c: statistics.mean(r["acc_val"] for r in com3[c].values())
    desvio = lambda c: statistics.stdev(r["acc_val"] for r in com3[c].values())
    base = ("adam", 1e-3, "", "constante")
    venc = max(com3, key=media)
    teste = {}
    for r in linhas:
        if r["etapa"] == "teste":
            teste.setdefault(painel.config(r), []).append(float(r["acc_teste"]))

    rv = com3[venc][0]
    ganho = [com3[venc][s]["acc_val"] - com3[base][s]["acc_val"] for s in (0, 1, 2)]

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.04, 0.93, "Desafio 3 — o melhor treino para a rede de 48 camadas do Desafio 2",
             fontsize=20, weight="bold")
    fig.text(0.04, 0.885, "Fashion-MNIST • MLP 784→128×46→10, LeakyReLU + init Monte Carlo, "
             f"sem normalização • {treino.PASSOS} passos, lote {treino.LOTE} para todos",
             fontsize=11, color="#555555")

    receita = f"Receita vencedora: {painel.nome_curto(rv)}"
    fig.text(0.04, 0.80, receita, fontsize=15, weight="bold", color="#1b6ca8")
    tv = teste.get(venc, [])
    tb = teste.get(base, [])
    txt = (f"Validação (3 sementes):  {media(venc):.4f} ± {desvio(venc):.4f}"
           f"   vs   baseline Adam η=1e-3 constante:  {media(base):.4f} ± {desvio(base):.4f}\n"
           f"Ganho pareado por semente: {statistics.mean(ganho):+.4f} ± "
           f"{statistics.stdev(ganho):.4f}")
    if tv and tb:
        txt += (f"\nTeste (usado uma vez):  {statistics.mean(tv):.4f} ± "
                f"{statistics.stdev(tv):.4f}   vs   {statistics.mean(tb):.4f} ± "
                f"{statistics.stdev(tb):.4f}")
    fig.text(0.04, 0.665, txt, fontsize=12, family="DejaVu Sans", linespacing=1.6)

    fig.text(0.04, 0.60, "Diagnóstico lido nos sinais", fontsize=14, weight="bold")
    fig.text(0.04, 0.575, DIAGNOSTICO, fontsize=11.5, va="top", linespacing=1.5, wrap=True)

    # mini-painel: mesmo eixo de passos
    lv = painel.ler_log(rv["id"])
    lb = painel.ler_log(com3[base][0]["id"])
    eixos = [fig.add_axes([0.60, 0.53 - 0.155 * i, 0.37, 0.14]) for i in range(3)]
    for lg, cor, rot in ((lb, painel.CORES["baseline"], "baseline"),
                         (lv, painel.CORES["melhor"], "vencedora")):
        p = lg["passo"]
        m = np.isfinite(lg["acc_val"])
        eixos[0].plot(p[m], lg["acc_val"][m], "o-", ms=2.5, color=cor, label=rot)
        eixos[1].plot(p, lg["grad"], color=cor, lw=0.9)
        eixos[2].plot(p, lg["lr"], color=cor, lw=1.4)
    eixos[0].set_ylabel("acc val", fontsize=9)
    eixos[1].set_ylabel("‖g‖", fontsize=9)
    eixos[1].set_yscale("log")
    eixos[2].set_ylabel("η_t", fontsize=9)
    eixos[2].set_yscale("log")
    eixos[2].set_xlabel("passo (semente 0)", fontsize=9)
    eixos[0].legend(fontsize=8, loc="lower right")
    for a in eixos:
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
    for a in eixos[:-1]:
        a.set_xticklabels([])

    fig.savefig(os.path.join(AQUI, "slide.pdf"))
    print(receita, f"ganho {statistics.mean(ganho):+.4f} ± {statistics.stdev(ganho):.4f}")


if __name__ == "__main__":
    main()
