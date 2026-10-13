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

DIAGNOSTICO = (
    "• Baseline (Adam, η=1e-3 constante): razão de atualização ≈2×10⁻³ na 1ª e na última\n"
    "  camada (acima da referência 10⁻³), picos de ‖g‖ (até 100) seguidos de saltos na perda\n"
    "  (validação cai a 25% no passo 1200 e a 10% no 2800) e ‖θ‖ da 1ª camada crescendo sem\n"
    "  estabilizar (15,9→21,4): passo grande demais e sem decaimento — o resultado final\n"
    "  depende de onde o treino parou (0,55 a 0,76 entre sementes).\n"
    "• η ajustada + cosseno: η_t→0 no fim e a perda \"assenta\" no vale; o desvio entre\n"
    "  sementes cai de 0,117 para 0,003.\n"
    "• Por que Muon: o passo ortogonalizado faz a camada do meio mudar ~6,6×10⁻⁴ por passo,\n"
    "  mesmo com ‖g‖ 16× menor que o da 1ª camada; no AdamW ela anda 3×10⁻⁴ (metade).\n"
    "  Perda de treino 0,19 vs 0,29 → +0,45 p.p. de validação, vencendo nas 3 sementes.\n"
    "• Limite: com η=0,03 o Muon diverge (‖θ‖ 109→200, ‖g‖×200, razão ~10⁻²)."
)


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
    alt = ("adamw", 3e-4, "", "wsd")
    if alt in com3:
        d = [com3[venc][s]["acc_val"] - com3[alt][s]["acc_val"] for s in (0, 1, 2)]
        txt += (f"\nMelhor sem Muon (AdamW η=3e-4 + WSD): {media(alt):.4f} ± {desvio(alt):.4f} — "
                f"Muon vence nas 3 sementes, {statistics.mean(d):+.4f} ± {statistics.stdev(d):.4f}"
                " (margem ≈ desvio)")
    fig.text(0.04, 0.635, txt, fontsize=11.5, family="DejaVu Sans", linespacing=1.6)

    fig.text(0.04, 0.575, "Diagnóstico lido nos sinais", fontsize=14, weight="bold")
    fig.text(0.04, 0.55, DIAGNOSTICO, fontsize=10.2, va="top", linespacing=1.5, wrap=True)

    # mini-painel: mesmo eixo de passos
    lv = painel.ler_log(rv["id"])
    lb = painel.ler_log(com3[base][0]["id"])
    eixos = [fig.add_axes([0.665, 0.50 - 0.15 * i, 0.31, 0.13]) for i in range(3)]
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
    eixos[2].set_ylim(1e-6, 1e-2)
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
