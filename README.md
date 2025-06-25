# Predição de Churn em Telecomunicações com Redes Neurais

## Descrição
Projeto de Machine Learning para predição de cancelamento (churn) de clientes em empresas de telecomunicações usando redes neurais. Desenvolvido como trabalho da disciplina de Inteligência Artificial e Aprendizado de Máquina - UNISINOS.

## Por que esse dataset é interessante para o projeto final? 
A retenção de clientes é um tema comum dentro do setor de telecomunicações, e o Brasil tem mais de 10 mil provedores de Internet no Brasil. O uso de Redes neurais são justificadas porque o churn é um problema complexo e não-linear que requer capacidade avançada de reconhecimento de padrões.
No trabalho apresentado, depende da interação entre 21 variáveis (tempo de contrato + tipo de serviço + valor + comportamento, etc). Redes neurais capturam essas interações ocultas que algoritmos lineares muitas vezes não conseguem detectar.

## Qual problema real ele representa? 
	Representa um problema enfrentando por todas as empresas de telecomunicações, visto a gama de prestadores que estão competindo e poder prever e prevenir o churn pode trazer economia no caixa da empresa, tanto no quesito aquisição de clientes, quanto na prevenção da perda. 

## Dataset
- **Fonte**: Telco Customer Churn (Kaggle)
- **Registros**: 7.043 clientes
- **Features**: 21 variáveis (demográficas, serviços, contratuais, financeiras)
- **Target**: Churn (Yes/No) - 26.5% de churn

## Arquitetura da Rede Neural
- **Tipo**: Multilayer Perceptron (MLP)
- **Camadas**: 4 camadas Dense (64 → 32 → 16 → 1)
- **Ativação**: ReLU nas camadas ocultas, Sigmoid na saída
- **Regularização**: Dropout (0.3, 0.2, 0.1)
- **Otimizador**: Adam (lr=0.001)
- **Parâmetros**: 4.673 parâmetros treináveis

## Pipeline de ML
1. **Análise Exploratória**: Características de clientes que cancelaram
2. **Pré-processamento**: One-hot encoding, normalização StandardScaler
3. **Divisão**: 70% treino, 15% validação, 15% teste
5. **Avaliação**: Métricas completas e comparação com baselines


## Principais Insights
1. **Tempo na base (tenure)** é o fator mais importante para predição
2. **Valor mensal (MonthlyCharges)** tem alto impacto nas decisões
3. **Contratos de dois anos** reduzem significativamente o churn
4. **Clientes novos (0-12 meses)** têm maior risco de cancelamento
5. **Falta de serviços atrelados** correlaciona com maior churn

## Interpretabilidade
Implementados dois métodos de análise:
- **Análise de pesos**: Conexões da primeira camada
- **Perturbação de features**: Impacto real nas predições (mais confiável)

## Estrutura do Código
- Pipeline completo de ML orientado a objetos
- Função de predição
- Sistema de salvamento/carregamento de modelos
- Análise comparativa com visualizações

## Conclusão
O projeto conseguiu implementar uma rede neural para predição de Churn, respondendo às questões principais sobre comportamento dos clientes. A análise revelou que clientes novos (0-12 meses), com contratos mensais e sem serviços adicionais têm maior propensão ao cancelamento.

Embora a rede neural tenha apresentado boa performance, a Regressão Logística se mostrou ligeiramente superior. Isso demonstra que para datasets tabulares menores como este, algoritmos mais simples podem ser mais eficazes que redes neurais complexas. Escolha do algoritmo deve sempre considerar a natureza dos dados.


## Requisitos
- Python 3.8+
- TensorFlow/Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## Execução
Execute o notebook sequencialmente. O código inclui tratamento de erros e logs detalhados para debugging.
