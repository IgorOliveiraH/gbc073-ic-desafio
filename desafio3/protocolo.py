# -*- coding: utf-8 -*-
"""
Executa o protocolo completo do Desafio 3 e preenche experimentos.csv / logs/.

    python protocolo.py [--workers 4]

Etapas (orçamento fixo: PASSOS passos, lote LOTE, em TODAS as execuções):
  baseline   receita do grupo (Adam, eta=1e-3, constante), 3 sementes
  varredura  >= 5 etas em grade log por otimizador, warmup+cosseno, semente 0;
             se o melhor cair na borda, estende a grade. Muon varre o eta das
             matrizes com a parte AdamW no melhor eta do AdamW.
  agendas    2 melhores otimizadores + Muon, melhor eta de cada, 4 agendas, semente 0
  confirmacao 3 melhores combinações, sementes 0, 1, 2
  teste      vencedora e baseline avaliadas no teste (única vez em que o teste é usado)

Execuções já presentes em experimentos.csv (mesmo id) são reaproveitadas, então o
script pode ser interrompido e retomado.
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import statistics

import torch

import treino

GRADES = {
    "sgd":          [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "sgd_nesterov": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "adam":         [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "adamw":        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "nadam":        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "muon":         [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}
BASELINE = dict(otimizador="adam", lr=1e-3, agenda="constante")
SEMENTES = (0, 1, 2)
MAX_EXTENSOES = 3

_DADOS = None


def _init_worker():
    global _DADOS
    torch.set_num_threads(1)
    _DADOS = treino.carregar_dados()


def _rodar(cfg):
    return treino.treinar(cfg, _DADOS)


def ler_csv():
    if not os.path.exists(treino.CSV_EXPERIMENTOS):
        return {}
    with open(treino.CSV_EXPERIMENTOS) as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


class Executor:
    def __init__(self, workers):
        self.pool = mp.Pool(workers, initializer=_init_worker)
        self.feitos = ler_csv()

    def rodar(self, cfgs):
        """Roda (em paralelo) as configurações ainda não executadas; devolve as linhas."""
        pend, vistos = [], set()
        for c in cfgs:
            c.setdefault("passos", treino.PASSOS)
            rid = treino.id_execucao(c)
            if rid not in self.feitos and rid not in vistos:
                pend.append(c)
                vistos.add(rid)
        for linha in self.pool.imap_unordered(_rodar, pend):
            treino.anexar_csv(linha)
            self.feitos[linha["id"]] = {k: str(v) for k, v in linha.items()}
            print(f"  [{linha['etapa']}] {linha['id']}: acc_val={linha['acc_val']} "
                  f"divergiu={linha['divergiu']} ({linha['tempo_s']}s)", flush=True)
        return [self.feitos[treino.id_execucao(c)] for c in cfgs]


def chave(r):
    """Ordenação: maior acurácia de validação; desempate pela menor perda de validação."""
    pv = float(r["perda_val"]) if r["perda_val"] not in ("", "nan") else math.inf
    return (-float(r["acc_val"]), pv)


def vizinho(lr, subir):
    m = round(lr / 10 ** math.floor(math.log10(lr) + 1e-9))
    if subir:
        return lr * (3 if m == 1 else 10 / 3)
    return lr / (10 / 3 if m == 1 else 3)


def varrer(ex, grades, extras=None):
    """Varredura em paralelo para vários otimizadores; estende a grade (em rodadas,
    também em paralelo) enquanto o melhor eta de algum otimizador cair na borda."""
    extras = extras or {}
    grades = {o: sorted(g) for o, g in grades.items()}

    def cfgs(o):
        return [dict(otimizador=o, lr=lr, agenda="warmup_cosseno", semente=0,
                     etapa="varredura", **extras.get(o, {})) for lr in grades[o]]

    def melhor(o):
        return min(zip(grades[o], ex.rodar(cfgs(o))), key=lambda gl: chave(gl[1]))

    ex.rodar([c for o in grades for c in cfgs(o)])
    for _ in range(MAX_EXTENSOES):
        novos = []
        for o, g in grades.items():
            lr = melhor(o)[0]
            if lr == g[0]:
                novo = float(f"{vizinho(g[0], False):.3g}")
            elif lr == g[-1]:
                novo = float(f"{vizinho(g[-1], True):.3g}")
            else:
                continue
            print(f"  {o}: melhor eta na borda ({lr:g}) -> estendendo com {novo:g}")
            grades[o] = sorted(g + [novo])
            novos += [c for c in cfgs(o) if c["lr"] == novo]
        if not novos:
            break
        ex.rodar(novos)
    return {o: melhor(o) for o in grades}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ex = Executor(a.workers)

    print("== baseline ==")
    ex.rodar([dict(BASELINE, semente=s, etapa="baseline") for s in SEMENTES])

    print("== varredura de eta (warmup + cosseno) ==")
    melhores = varrer(ex, {o: GRADES[o] for o in GRADES if o != "muon"})
    lr_aw = melhores["adamw"][0]
    melhores.update(varrer(ex, {"muon": GRADES["muon"]}, {"muon": dict(lr_adamw=lr_aw)}))
    for o, (lr, r) in melhores.items():
        print(f"  melhor {o}: eta={lr:g} acc_val={r['acc_val']}")

    print("== agendas ==")
    ranking = sorted(melhores, key=lambda o: chave(melhores[o][1]))
    escolhidos = ranking[:2] + ([] if "muon" in ranking[:2] else ["muon"])
    print("  otimizadores:", escolhidos)
    combos = []
    for otim in escolhidos:
        extra = dict(lr_adamw=lr_aw) if otim == "muon" else {}
        for ag in treino.AGENDAS:
            combos.append(dict(otimizador=otim, lr=melhores[otim][0], agenda=ag, semente=0,
                               etapa="agendas", **extra))
    linhas = ex.rodar(combos)

    print("== confirmação (3 sementes) ==")
    top3 = [c for c, _ in sorted(zip(combos, linhas), key=lambda cl: chave(cl[1]))[:3]]
    finais = top3 + [dict(BASELINE, passos=treino.PASSOS)]
    ex.rodar([dict(c, semente=s, etapa="confirmacao") for c in finais for s in SEMENTES])
    resumo = {}
    for c in finais:
        ls = ex.rodar([dict(c, semente=s) for s in SEMENTES])
        accs = [float(l["acc_val"]) for l in ls]
        nome = treino.id_execucao(dict(c, semente="*"))
        resumo[nome] = (statistics.mean(accs), statistics.stdev(accs), c)
        print(f"  {nome}: acc_val = {resumo[nome][0]:.4f} ± {resumo[nome][1]:.4f}")

    print("== teste (vencedora e baseline, uma única vez) ==")
    venc = max(resumo, key=lambda k: resumo[k][0])
    base = treino.id_execucao(dict(BASELINE, semente="*"))
    alvos = [resumo[n][2] for n in dict.fromkeys((venc, base))]
    ex.rodar([dict(c, semente=s, teste=True, etapa="teste") for c in alvos for s in SEMENTES])
    for c in alvos:
        ls = ex.rodar([dict(c, semente=s, teste=True) for s in SEMENTES])
        print(f"  {treino.id_execucao(dict(c, semente='*'))}: "
              f"acc_teste = {[l['acc_teste'] for l in ls]}")


if __name__ == "__main__":
    main()
