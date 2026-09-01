---
doc_id: tab-mdr-bandeiras
title: Tabela de MDR por Bandeira e Adquirente
doc_type: tabela
period: 2025-09
version: "2025.09"
---

# Tabela de MDR por Bandeira e Adquirente

Vigência: 01/09/2025 a 30/11/2025. Renegociação trimestral conduzida pela
Tesouraria Corporativa. MDR (Merchant Discount Rate) é a taxa retida pela
adquirente sobre o valor bruto da transação.

## Adquirente Alfa — código ADQ-4471

Adquirente principal da rede, responsável por 71% do volume transacionado.

| Bandeira | Débito à vista | Crédito à vista | Parcelado 2 a 6x | Parcelado 7 a 12x |
| --- | --- | --- | --- | --- |
| Visa | 1,29% | 2,49% | 2,89% | 3,19% |
| Mastercard | 1,29% | 2,49% | 2,89% | 3,19% |
| Elo | 1,39% | 2,89% | 3,29% | 3,59% |
| American Express | — | 3,45% | 3,85% | 4,15% |
| Hipercard | 1,49% | 3,09% | 3,49% | 3,79% |

## Adquirente Beta — código ADQ-8802

Adquirente secundária, usada como contingência e nas praças onde a Alfa não
tem cobertura de terminal.

| Bandeira | Débito à vista | Crédito à vista | Parcelado 2 a 6x | Parcelado 7 a 12x |
| --- | --- | --- | --- | --- |
| Visa | 1,45% | 2,79% | 3,15% | 3,49% |
| Mastercard | 1,45% | 2,79% | 3,15% | 3,49% |
| Elo | 1,55% | 3,15% | 3,55% | 3,89% |
| American Express | — | 3,69% | 4,05% | 4,39% |
| Hipercard | 1,65% | 3,35% | 3,75% | 4,05% |

## Taxas acessórias

| Item | ADQ-4471 (Alfa) | ADQ-8802 (Beta) |
| --- | --- | --- |
| Aluguel mensal por terminal POS | R$ 39,90 | R$ 54,90 |
| Antecipação de recebíveis (mês) | 1,49% | 1,72% |
| Tarifa por chargeback contestado | R$ 12,00 | R$ 18,50 |
| Tarifa de conciliação por arquivo EDI | isenta | R$ 89,00/mês |
| Multa por cancelamento antecipado de contrato | 3 aluguéis | 6 aluguéis |

## Regra de escolha de adquirente

O roteamento de transação segue, nesta ordem:

1. Se a praça tem cobertura Alfa e o terminal está saudável, usar ADQ-4471.
2. Se a transação é American Express acima de R$ 3.000,00, usar ADQ-4471
   independentemente da praça, por causa do limite de garantia da Beta.
3. Em falha de comunicação do terminal Alfa por mais de 90 segundos, cair
   automaticamente para ADQ-8802.
4. Pix na maquininha é sempre roteado pela Alfa, que não cobra MDR sobre a
   modalidade até 31/12/2025.

## Observação sobre o cálculo do valor líquido

O valor líquido creditado é calculado sobre o valor bruto da transação, e não
sobre o valor da parcela. Em uma venda parcelada, o MDR é retido integralmente
na primeira parcela liquidada. Exemplo: venda de R$ 1.200,00 em 4x Visa
(parcelado 2 a 6x, MDR 2,89%) tem retenção total de R$ 34,68, debitada na
liquidação da primeira parcela, restando parcelas de R$ 300,00 sem retenção
adicional.
