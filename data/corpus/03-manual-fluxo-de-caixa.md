---
doc_id: man-fluxo-caixa
title: Manual de Fluxo de Caixa do Franqueado
doc_type: manual
period: 2025-09
version: "3.1"
---

# Manual de Fluxo de Caixa do Franqueado

## 1. Regime de caixa versus regime de competência

O painel de Fluxo de Caixa da plataforma Finanças Aurora opera em **regime de
caixa**: cada lançamento aparece na data em que o dinheiro entra ou sai da
conta. O painel de DRE Gerencial opera em **regime de competência**: a receita
aparece na data da venda, independentemente de quando o recebível liquida.

A consequência prática é que uma venda de crédito à vista feita em 28/09
aparece na receita de setembro no DRE e apenas em outubro no Fluxo de Caixa.
Divergência entre os dois painéis não é erro: é diferença de regime.

## 2. Estrutura de categorias

### 2.1 Entradas

| Categoria | Código | Origem |
| --- | --- | --- |
| Recebíveis de cartão liquidados | ENT-01 | Adquirentes |
| Vendas em dinheiro e Pix | ENT-02 | PDV |
| Antecipação de recebíveis | ENT-03 | Adquirentes |
| Aporte de sócio | ENT-04 | Sócios |
| Empréstimo e financiamento | ENT-05 | Bancos |
| Devolução de tributos | ENT-06 | Receita Federal e Sefaz |
| Outras entradas | ENT-99 | Diversas |

### 2.2 Saídas

| Categoria | Código | Vencimento típico |
| --- | --- | --- |
| Compra de mercadoria da Aurora | SAI-01 | 28 dias da nota |
| Royalties | SAI-02 | Dia 10 do mês seguinte |
| Fundo de marketing | SAI-03 | Dia 10 do mês seguinte |
| Aluguel do ponto comercial | SAI-04 | Dia 5 |
| Folha de pagamento | SAI-05 | Dia 5 |
| Encargos e tributos sobre folha | SAI-06 | Dia 20 |
| Tributos sobre venda | SAI-07 | Dia 20 |
| Aluguel de terminais POS | SAI-08 | Dia 15 |
| Energia, água e condomínio | SAI-09 | Variável |
| Despesas administrativas | SAI-10 | Variável |
| Investimento em reforma da loja | SAI-11 | Conforme contrato |
| Outras saídas | SAI-99 | Variável |

## 3. Capital de giro mínimo recomendado

A Aurora recomenda que a unidade mantenha capital de giro equivalente a **45
dias de custo fixo mensal**. Para uma loja de padrão P2 (área entre 60 e 90
metros quadrados), o custo fixo médio apurado no terceiro trimestre de 2025 foi
de R$ 74.200,00 por mês, o que resulta em capital de giro mínimo recomendado de
R$ 111.300,00.

A unidade que operar por dois meses consecutivos com capital de giro abaixo de
25 dias de custo fixo é classificada como **atenção financeira** e passa a
receber acompanhamento quinzenal do consultor de campo.

## 4. Projeção de fluxo de caixa

A projeção padrão do painel cobre 90 dias corridos e é montada com base em:

- agenda de recebíveis confirmada pelas adquirentes (dado firme);
- vendas em dinheiro e Pix projetadas pela média móvel de 8 semanas;
- contas a pagar já lançadas (dado firme);
- royalties e fundo de marketing projetados sobre a venda projetada;
- sazonalidade da praça, aplicada por um índice regional divulgado
  trimestralmente pela Controladoria.

O erro médio absoluto da projeção de 30 dias medido em 2025 foi de 6,8%. Para a
projeção de 90 dias, o erro médio absoluto foi de 14,2%.

## 5. Alertas automáticos

| Alerta | Gatilho | Destinatário |
| --- | --- | --- |
| ALT-CX-01 | Saldo projetado negativo em até 15 dias | Franqueado e consultor |
| ALT-CX-02 | Saldo projetado negativo em 16 a 45 dias | Franqueado |
| ALT-CX-03 | Capital de giro abaixo de 25 dias de custo fixo | Franqueado e consultor |
| ALT-CX-04 | Antecipação acima de 40% da agenda em um mês | Franqueado e Controladoria |
| ALT-CX-05 | Royalties em atraso há mais de 5 dias | Franqueado e Financeiro Aurora |

O alerta ALT-CX-04 existe porque antecipação recorrente acima de 40% da agenda
é o indicador que melhor antecipa inadimplência de royalties na base histórica
da rede: das unidades que ultrapassaram esse patamar por três meses seguidos,
62% ficaram inadimplentes nos seis meses subsequentes.
