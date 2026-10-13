# Desafio 3 — O melhor treino para o seu problema

**Problema:** a rede profunda do Desafio 2: MLP com **48 camadas lineares, sem normalização**, ativação
LeakyReLU(0.1) e inicialização calibrada por Monte Carlo. A ativação e a inicialização são importadas
diretamente de `../desafio2/desafio2_igor_ihury_vinicius.py`. A rede é treinada em **Fashion-MNIST**
(784 → 128 × 46 → 10).

Divisão fixa para todas as execuções: 55 000 treino / 5 000 validação / 10 000 teste. O teste é
usado uma única vez, no fim. Orçamento fixo para **todas** as execuções: **3000 passos, lote 128**
(≈ 7 épocas).

## Receita vencedora

**Muon (η = 3·10⁻³ nas matrizes ocultas; AdamW η = 3·10⁻⁴ em vieses e na saída) + agenda cosseno**

| Configuração | validação (3 sementes) | teste (3 sementes) |
|---|---|---|
| **Muon η=3e-3 + cosseno** | **0,8850 ± 0,0029** | **0,8791 ± 0,0004** |
| Muon η=3e-3 + warmup+cosseno | 0,8845 ± 0,0021 | — |
| AdamW η=3e-4 + WSD (melhor sem Muon) | 0,8805 ± 0,0013 | — |
| Baseline: Adam η=1e-3, constante | 0,6275 ± 0,1167 | 0,6273 ± 0,1102 |

- **Ganho sobre a baseline** (pareado por semente): +0,2575 ± 0,1137 na validação. Ele é muito maior
  que a variação entre sementes de qualquer das duas configurações.
- **Muon × AdamW**: o Muon vence nas 3 sementes, com +0,0045 ± 0,0042. A margem é consistente, mas
  tem o tamanho do desvio; não é uma vitória folgada.
- **Cosseno × warmup+cosseno** no Muon: +0,0005 ± 0,0016, empate técnico.

## Diagnóstico lido nos sinais (ver `painel.pdf`)

1. **Baseline: passo grande demais e sem decaimento.**
   - A razão de atualização fica em ≈ 2·10⁻³ na primeira e na última camada, acima da
     referência de 10⁻³.
   - Há picos de ‖g‖ (até ~100) seguidos de saltos na perda. Na semente 0, a acurácia de validação
     cai para 25% no passo 1200 e para 10% no passo 2800.
   - A ‖θ‖ da primeira camada cresce sem estabilizar (15,9 → 21,4).
   - Com η constante o treino nunca "assenta": a métrica final depende de onde o treino parou
     (0,55 a 0,76 entre sementes).
2. **η ajustada + agenda que decai.** Com cosseno, η_t → 0 no fim, a perda cai no vale e o desvio
   entre sementes despenca de 0,117 para 0,003. Para o Muon, a agenda constante perde 3,8 p.p.
   (0,850 contra 0,888).
3. **Por que o Muon ganha.** O passo ortogonalizado dá o mesmo tamanho a todas as direções
   singulares:
   - A camada do meio muda ~6,6·10⁻⁴ por passo, mesmo com ‖g‖ ~16× menor que o da primeira camada.
   - No AdamW+WSD, essa mesma camada anda 3,1·10⁻⁴, metade disso.
   - Resultado: a rede profunda ajusta melhor o treino (perda 0,19 contra 0,29) e ganha
     +0,45 p.p. de acurácia de validação.
4. **Instável: Muon com η = 0,03.** ‖θ‖ sobe de 109 para 200 sem estabilizar, ‖g‖ cresce ~200× e a
   razão de atualização fica em ~10⁻², 10× a referência, até divergir no passo 820. É o teto de η
   do Muon nesta rede.
5. **Gradiente sumindo: SGD+Nesterov com η = 0,1.** Esta execução não diverge: ‖g‖ ≈ 0 desde cedo,
   a razão de atualização fica em ~10⁻⁷ e a perda fica presa em ln 10 = 2,30. A rede "morreu" (sinal
   colapsado), não explodiu.

### Ressalvas honestas

- **Perda de validação:** o Muon tem acurácia maior, mas perda de validação maior que a do AdamW+WSD
  (≈ 0,40 contra 0,35). Ele ajusta mais o treino e fica mais confiante. Se a métrica de seleção fosse
  a perda de validação, AdamW+WSD seria o vencedor.
- **Custo:** o orçamento é igual em passos, não em tempo. No CPU, cada execução do Muon leva ~160 s,
  contra ~42 s do AdamW, por causa das 5 iterações de Newton–Schulz em 47 matrizes por passo.

## Protocolo executado (`protocolo.py`)

1. **Baseline**: Adam η=1e-3 constante, 3 sementes. O Desafio 2 definia só a ativação e a
   inicialização; esta é a receita "padrão" que usaríamos sem ajuste.
2. **Varredura de η**: 6 otimizadores, ≥ 5 valores em grade log (×√10), warmup+cosseno, semente 0.
   - Otimizadores: SGD com momento 0,9, SGD+Nesterov, Adam, AdamW, NAdam e Muon.
   - Quando o melhor η caiu na borda, a grade foi estendida (SGD e SGD+Nesterov: estendidos até 1e-3).
   - Muon: η das matrizes em 1e-3…1e-1, com a parte AdamW no melhor η do AdamW (3e-4).

   | otimizador | melhor η | acc_val |
   |---|---|---|
   | Muon | 3e-3 | 0,8866 |
   | AdamW (wd 0,1) | 3e-4 | 0,8778 |
   | NAdam | 3e-4 | 0,8768 |
   | Adam | 3e-4 | 0,8758 |
   | SGD+Nesterov | 3e-3 | 0,8606 |
   | SGD | 3e-3 | 0,8594 |

3. **Agendas** (constante, cosseno, warmup+cosseno, WSD): aplicadas aos 2 melhores otimizadores
   (Muon e AdamW), com o melhor η de cada, semente 0.
4. **Confirmação**: as 3 melhores combinações com sementes 0, 1 e 2.
5. **Teste**: a vencedora e a baseline, 3 sementes cada. É a única vez em que o teste é usado.

**Todas as 53 execuções estão em `experimentos.csv`, incluindo as 13 que divergiram.**
"Divergiu" significa perda não finita ou maior que 100. A coluna `comando` reproduz a linha
exatamente. As 6 execuções de teste são reexecuções das de confirmação/baseline e reproduziram bit a
bit a acurácia e a perda de validação.

## Instrumentação (`treino.py`)

A cada K = 20 passos o treino registra, globalmente e para a primeira camada (`camadas.0`), a do
meio (`camadas.23`) e a última (`head`):

- perda do lote e média dos K passos;
- ‖∇θ ℒ‖₂, antes de qualquer clipping (o clipping não é usado);
- ‖θ‖₂;
- razão de atualização ‖θₜ₊₁ − θₜ‖/‖θₜ‖;
- η_t (e o η_t da parte AdamW, no Muon).

Perda e acurácia de validação são registradas a cada 100 passos. Os logs ficam em `logs/<id>.csv`.

**Muon**: implementação própria (`Muon` em `treino.py`):

- momento de Nesterov (0,95) e passo `ortogonaliza(M)` com 5 iterações de Newton–Schulz;
- escala √max(1, linhas/colunas) e weight decay desacoplado de 0,01;
- vale para todas as matrizes 2-D das camadas ocultas (inclusive a de entrada 784→128);
- vieses e camada de saída ficam no AdamW (wd 0,1), com a **mesma** agenda nos dois otimizadores.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `treino.py` | uma execução configurável: `--otimizador --lr --agenda --semente [--lr-adamw] [--teste]` |
| `protocolo.py` | roda o protocolo inteiro em paralelo; é retomável, pois reaproveita ids já no CSV |
| `experimentos.csv` | uma linha por execução: configuração, métrica de validação, divergiu?, tempo e comando |
| `logs/` | sinais de todas as execuções |
| `painel.py` → `painel.pdf` | sinais alinhados (melhor, baseline, melhor sem Muon, instável), sinais por camada, métrica × η, agendas e confirmação |
| `slide.py` → `slide.pdf` | o slide de 3 minutos |

## Reproduzir

```bash
pip install torch matplotlib numpy
cd desafio3
python treino.py --otimizador muon --lr 0.003 --lr-adamw 0.0003 --agenda cosseno --semente 0
python protocolo.py --workers 4     # tudo (~55 min de CPU no total; ~15 min com 4 processos)
python painel.py && python slide.py
```

O Fashion-MNIST é baixado automaticamente para `dados/`, que fica fora do git. Usamos CPU, 1 thread
por processo; os resultados são determinísticos por semente.
