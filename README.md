# Desafio 1: Feature Map Adaptativo para o Perceptron
**Disciplina:** GBC073 — Inteligência Computacional  
**Docente:** Dr. Marcelo Keese Albertini  

## Equipe
* Igor Oliveira Souza e Silva
* Ihury Kewin de Oliveira Costa
* Vinicius Resende Garcia

## Sobre o Projeto
Este repositório contém a submissão do Desafio 1, cujo objetivo é criar um mapeamento de características para permitir que um classificador linear (Perceptron) resolva fronteiras não-lineares complexas. O nosso modelo superou o *baseline* e a referência, alcançando um **Escore S = 101.1**.

## Melhorias Implementadas
1. **Prevenção do Bug do Float32:** Projeções elevadas para `float64` para garantir determinismo absoluto no PyTorch.
2. **Dinamicidade de Dimensões:** Alocação inteligente (budget = 64 - 2d) baseada na entrada original.
3. **Orthogonal Random Features (ORF):** Decomposição QR combinada com distribuição Chi para eliminar redundância direcional.
4. **Remoção de Fase Aleatória:** Uso da Identidade de Euler com pares de Seno e Cosseno para reduzir o erro de Monte Carlo.
5. **Multi-escala de Frequências:** Divisão dos pesos em 4 bandas (0.2 a 2.5×) para capturar micro e macro-topologias.

## Arquivos
* `submissao.py`: Código-fonte com a classe de mapeamento.
* `Apresentacao_Desafio1.pdf`: Slides detalhando a evolução matemática e arquitetural do modelo.
