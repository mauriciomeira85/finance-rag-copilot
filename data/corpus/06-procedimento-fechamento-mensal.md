---
doc_id: proc-fechamento-mensal
title: Procedimento de Fechamento Contábil e Gerencial Mensal
doc_type: procedimento
period: 2025-09
version: "5.3"
---

# Procedimento de Fechamento Contábil e Gerencial Mensal

## 1. Calendário

O fechamento gerencial encerra no **5º dia útil** do mês seguinte à
competência. O fechamento contábil encerra no **10º dia útil**. Após o
fechamento gerencial, lançamentos na competência encerrada exigem autorização
do Gerente de Controladoria e ficam registrados na trilha de auditoria com
justificativa obrigatória.

| Dia útil | Atividade | Responsável |
| --- | --- | --- |
| D1 | Encerramento do movimento de PDV e bloqueio de venda retroativa | TI |
| D1 | Importação final dos arquivos EDI das adquirentes | Tesouraria |
| D2 | Conciliação bancária de todas as contas | Tesouraria |
| D2 | Conciliação de cartões consolidada | Controladoria |
| D3 | Apuração de estoque e custo da mercadoria vendida | Suprimentos |
| D3 | Provisões de folha, encargos e férias | RH e Contabilidade |
| D4 | Cálculo de royalties e fundo de marketing por unidade | Controladoria |
| D4 | Revisão de despesas por centro de custo | Controladoria |
| D5 | Publicação do DRE Gerencial e do Fluxo de Caixa realizado | Controladoria |
| D8 | Apuração de tributos e obrigações acessórias | Fiscal |
| D10 | Encerramento contábil e travamento do período | Contabilidade |

## 2. Checklist obrigatório do fechamento gerencial

O fechamento não é publicado enquanto os oito itens abaixo não estiverem
marcados como concluídos no painel de controle:

1. Nenhuma divergência de conciliação de cartões acima de R$ 5.000,00 sem
   classificação.
2. Todas as contas bancárias com saldo contábil igual ao saldo do extrato.
3. Estoque físico apurado com divergência total menor ou igual a 1,5% do valor
   do estoque contábil.
4. Provisão de férias e décimo terceiro atualizada com a folha do mês.
5. Royalties e fundo de marketing calculados para 100% das unidades ativas.
6. Nenhum lançamento em conta transitória com saldo residual acima de
   R$ 1.000,00.
7. Todas as notas de compra de mercadoria do mês escrituradas.
8. Aprovação eletrônica do Gerente de Controladoria registrada.

## 3. Contas transitórias

Contas transitórias existem para acolher lançamentos cuja classificação
definitiva depende de informação ainda não disponível no momento do registro.
São de uso restrito e devem ser zeradas no fechamento.

| Conta | Uso permitido | Prazo máximo de permanência |
| --- | --- | --- |
| 1.9.01 Recebíveis em conciliação | Crédito recebido sem identificação da venda | 5 dias úteis |
| 1.9.02 Adiantamento a fornecedor | Pagamento antes da nota fiscal | 30 dias corridos |
| 2.9.01 Valores a classificar | Débito em conta sem documento suporte | 5 dias úteis |
| 2.9.02 Provisão de chargeback | Contestação em análise | até decisão final |

Saldo em conta transitória além do prazo máximo gera apontamento automático no
relatório de qualidade contábil e é reportado ao Comitê de Auditoria.

## 4. Reabertura de período

A reabertura de um período já travado é excepcional e admitida apenas em três
hipóteses:

- erro material identificado pela auditoria independente;
- determinação de autoridade fiscal ou judicial;
- reclassificação exigida por mudança de norma contábil.

Reabertura por conveniência operacional não é permitida. Ajuste de valor
imaterial identificado após o travamento é registrado na competência corrente,
com nota explicativa no DRE Gerencial.

## 5. Indicadores de qualidade do fechamento

| Indicador | Meta | Realizado em setembro de 2025 |
| --- | --- | --- |
| Fechamento gerencial publicado até D5 | 100% | 100% |
| Divergências de conciliação sem classificação em D2 | 0 | 3 |
| Saldo residual em contas transitórias | R$ 0 | R$ 4.180,00 |
| Divergência de estoque | <= 1,5% | 0,9% |
| Reaberturas de período no mês | 0 | 0 |
| Lançamentos retroativos autorizados | <= 5 | 2 |
