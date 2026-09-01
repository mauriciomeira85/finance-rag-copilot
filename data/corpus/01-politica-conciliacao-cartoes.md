---
doc_id: pol-conciliacao-cartoes
title: Política de Conciliação de Cartões
doc_type: politica
period: 2025-09
version: "4.2"
---

# Política de Conciliação de Cartões

## 1. Objetivo e abrangência

Esta política define o processo de conciliação entre as vendas registradas no
PDV das lojas da Rede Aurora Cosméticos, os arquivos de venda enviados pelas
adquirentes e os créditos efetivamente liquidados em conta corrente. Aplica-se
a todas as 1.412 lojas da rede, sejam próprias ou franqueadas, e a todas as
modalidades de pagamento com cartão.

A responsabilidade primária pela conciliação diária é do franqueado. A
Controladoria da rede executa a conciliação consolidada mensal e é a instância
que homologa divergências acima do limite de tolerância.

## 2. Modalidades cobertas

| Modalidade | Código interno | Prazo de liquidação | Antecipável |
| --- | --- | --- | --- |
| Débito à vista | DEB-AV | D+1 útil | Não |
| Crédito à vista | CRE-AV | D+30 corridos | Sim |
| Crédito parcelado sem juros (2 a 6x) | CRE-PS | D+30 por parcela | Sim |
| Crédito parcelado com juros (7 a 12x) | CRE-PC | D+30 por parcela | Sim |
| Voucher e cartão-presente Aurora | VCH-AU | D+2 úteis | Não |
| Carteira digital (Pix na maquininha) | PIX-MQ | D+0 | Não |

## 3. Limite de tolerância para divergências

Uma divergência entre o valor esperado e o valor liquidado é considerada
**imaterial** e pode ser baixada automaticamente pelo sistema quando atende
simultaneamente aos dois critérios abaixo:

- valor absoluto da divergência menor ou igual a R$ 50,00; e
- valor relativo da divergência menor ou igual a 0,5% do total liquidado no dia.

Divergências que ultrapassem qualquer um dos dois critérios entram na fila de
tratamento manual e devem ser classificadas em até 5 dias úteis. Divergências
acima de R$ 5.000,00 exigem aprovação do Gerente de Controladoria e abertura de
chamado formal junto à adquirente.

## 4. Classificação de divergências

| Código | Causa | Responsável pelo tratamento | Prazo |
| --- | --- | --- | --- |
| DIV-01 | Venda no PDV sem arquivo da adquirente | Franqueado | 5 dias úteis |
| DIV-02 | Arquivo da adquirente sem venda no PDV | Franqueado | 5 dias úteis |
| DIV-03 | Diferença de MDR aplicado | Controladoria | 10 dias úteis |
| DIV-04 | Chargeback | Jurídico e Controladoria | 30 dias corridos |
| DIV-05 | Antecipação não refletida no extrato | Tesouraria | 3 dias úteis |
| DIV-06 | Cancelamento parcial não processado | Franqueado | 5 dias úteis |
| DIV-07 | Divergência de data de liquidação | Tesouraria | 3 dias úteis |

## 5. Fluxo operacional diário

1. Às 06h00, o robô de integração baixa os arquivos EDI das adquirentes
   referentes a D-1.
2. Às 07h00, o motor de conciliação cruza três fontes: movimento do PDV,
   arquivo de vendas da adquirente e extrato bancário (OFM).
3. Às 08h00, o painel de conciliação é liberado ao franqueado com o resultado
   classificado por código de divergência.
4. O franqueado tem até as 18h00 do mesmo dia para tratar as divergências de
   sua responsabilidade.
5. Divergências não tratadas em 5 dias úteis são escalonadas automaticamente
   para o consultor de campo da região.

## 6. Chargeback

O chargeback é debitado da agenda futura de recebíveis do franqueado no
momento em que a adquirente comunica a contestação, ainda que a defesa esteja
em andamento. Se a defesa for aceita, o valor é restituído na agenda em até 2
dias úteis após a comunicação da adquirente.

O índice de chargeback aceitável é de até 0,35% do faturamento com cartão de
crédito no mês. Acima disso, a loja entra em monitoramento e a Aurora pode
exigir a adoção de antifraude adicional no e-commerce da unidade.

## 7. Antecipação de recebíveis

A antecipação é uma decisão do franqueado e não depende de autorização da
Aurora, desde que a unidade esteja adimplente com royalties e com o fundo de
marketing. A taxa de antecipação vigente é de 1,49% ao mês, calculada de forma
pro rata die sobre o prazo remanescente até a liquidação original.

Recebíveis já dados em garantia de operação de crédito não podem ser
antecipados. A trava de domicílio bancário, quando existente, deve ser
respeitada pelo sistema de conciliação, que classifica o crédito como
liquidado apenas quando o valor entra na conta gravada.
