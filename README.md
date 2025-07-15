# TP3 (não entregue anteriormente) 

Neste trabalho foram testado três modelos, o original KNN 8 que foi visto como o melhor modelo do TP2, o random forest (sem alteração), e o random forest ajustado com hiperparâmetros. A métrica usada na classificação, acurácia, precisão, recall, f1-score e principalmente AUC, nos mostram o desempenho de cada modelo, todos foram avaliados no final para saber qual teve o melhor desempenho.

A tabela final obtida foi essa:

| Modelo                | Acurácia | Precisão | Recall  | F1-score | F1 (CV Média) |
| --------------------- | -------- | -------- | ------- | -------- | ------------- |
| KNN                   | 0.95614  | 0.95890  | 0.97222 | 0.96551  | 0.97409       |
| Random Forest         | 0.95614  | 0.95890  | 0.97222 | 0.96551  | 0.96341       |
| Random Forest (hiper) | 0.95614  | 0.95890  | 0.97222 | 0.96551  | 0.96844       |

De acordo com os resultados podemos começar a interpretar.

A acurácia dos modelos mostram resultados similares, então o modelo em si não é importante nesse aspecto. Já em precisão e recall vemos dados idênticos, ambos esses dados servem para averiguar as taxas de acerto de um modelo, comparando o resultado e o previsto, sendo o f1 a harmonia dos dois resultados de precisão e recall, que também se mantém estático.

Agora no resultado final vemos uma maior diferença, também chamadop de AUC, vemos que o KNN se destaca com uma precisão maior do que modelos mais complexos que ele, além disso também vemos que o Random Forest que teve seus parâmetros re-treinados por hiperparâmetros, teve um aumento em seu score. Com isso o que podemos aprender?

Que o modelo escolhido ainda é o KNN, apesar de ser mais simples é visível que esse modelo de dados, é mais simples de previsões, todos as 3 formas de treinar o ml chegaram em resultados extremamente parecidos, e nas poucas diferenças que tiveram o KNN se destacou, portanto é o escolhido para essa situação.

Uma forma mais visível de suas diferenças é no gráfico de ROC que o KNN começa a performar melhor com uma menor seleção de dados, enquanto os outros precisam de mais falsos positivos para performar bem.

# AT

1.A análise do problema

1.B a classificação supervisionada é a forma de solucionar este problema visto que a base de dados de indicadores de diabetes possui features que relacionam diretamente à um resultado verificável, ou seja, cada atributo como fumante, atividade física, e outros parâmetros são relacionados diretamente com o fator diabetes, possuir ou não, e como terceiro caso, pré diabetes, então à partir de cada entrada o modelo aprendera a correlação com a saída procurada.

1.C Possuímos na base de dados um número muito baixo de indicadores de colesterol que sejam diferentes de 1, que pode tornar difícil encontrar a correlação desejada, assim como, na própria categoria de prediabetes um número extremamente baixo, provavelmente que fará o modelo não procurar por sinais confiáveis de pré diabetes.

1.D Pandas, biblioteca essencial para organização e criação de conjuntos de dados de forma tabular.
Numpy, biblioteca de operações com dados que se tratam em formato de array.
Matplotlib, criação de gráficos, utilizados em várias partes do projeto que necessitam de partes visuais para compreensão dos dados.
Scikit-learn, incluem diversos métodos para o treinamento de ml, como exemplo nesse programa temos, accuracy_score, precision_score, recall_score, f1_score, entre outros, essencial para o desenvolvimento da inteligência e capacidade de classificação do modelo.

2.A Considerando o objetivo do projeto, de realizar identificações precoces utilizando sinais de diabetes, nos precisamos de como variável-alvo a 'Diabetes_012' que ira guiar as features, que precisam ser alinhadas com o cidadão médio e não causem erro por utilizarem features com pouca relação à diabetes 012, portanto foi utilizado uma árvore de decisão que fizesse uma análise das features com maior impacto na tabela e estas foram utilizadas para os treinamentos seguintes, mas nesse caso à tendência são features gerais e com maior apelo geral para que uma pessoa média possa ser classificada de acordo com sinais de diabete ou não. 

2.b A separação dos dados dividiu entre dados de treino e dados de teste e a estretificação foi aplicada na variável y, o alvo do modelo, para que quantidades justas sejam distribuidas entre os diferentes cortes para treino e teste, assim o modelo não fica enviesado ou até mesmo com menor acurácia.

2.c A normalização é uma parte essencial para o desenvolvimento já que muitos dados numéricos utilizam diferentes escalas, e com isso uma comunicação e uma interligação menos funcional quando o objetivo é encontrar os padrões, portanto features como bmi que funciona entre 18 e 29 normalmente, é altamente diferente de renda ou idade.

2.d Diferentes valores de K foram testados (de 1 a 20), observando-se como a acurácia variava. É necessário entender para escolher o melhor valor, que valores baixos tendem à se acomodar aos dados de treino, e valores altos tendem à não pegar à nuância dos dados para ser capaz de avaliar apropriadamente os sinais dos dados que nesse caso, indicam sinais de diabetes. Portanto considerando, a mediana dos valores K e também se preocupando com a acuracidade descoberta, o melhor valor K para se seguir é K10, com acuracidade 8.609 e não sendo alto ou baixo, o equilíbrio.

3.a A validação cruzada aplicada sobre o valor K anterior assegura a acuraciadade do modelo dividindo os dados em dobras diferentes e utilizando entre as dobras partes de treino e algumas de teste, realizando uma estimativa mais confiável do valor anteriormente conseguido. Nesse caso escolhi o valor de 5 que é recomendado por não ser tão baixo ou muito alto.

3.b Foi aplicado regressão logística, que é um método de estatística utilizado para classificar a correlação entre uma variável e o seu resultado, é feito a estimativa da probabilidade de certo evento ocorrer utilizando estes parâmetros, após aplicar a regressão logística foi utilizado novamente a validação cruzada, que dessa vez retornou uma maior acuracidade, mantendo o mesmo valor de dobras e objeto de avaliação.

3.c O gráfico mostra uma maior acuracidade no modelo de Regressão Logística, apesar de levemente um desvio maior, e o KNN com uma acuracidade cerca de alguns décimos abaixo, e um desvio considerávelmente menor, apesar disso, a Regressão Logística ainda é o modelo com melhores resultados.

3.d Como dito anteriormente o modelo de Regressão logística apresentou maior acurácia por mais que agora comparando com o KNN, apresentou também um desvio maior, os indícios de overfitting ou underfitting são inconclusivos, considerando os testes feitos em folds, não apresentaram dificuldade em se manter estáveis em diferentes dados, e também sobre o underfitting, ambos modelos possuem acuracidade acima de 85% que não é um sinal de impossibilidade em estudar o modelo, portanto não há sinais de ambas condições

4.a As métricas presente na curva ROC são as seguintes, Precisão, que calcula quantos dos selecionados positivo realmente entravam nessa categoria, Recall, que calcula quantos dados que eram realmente positivos o modelo conseguiu classificar corretamente, f1-score, a junção das duas medidas previamente calculadas, e AUC, a capacidade entre separação de classes. Dito isso, os resultados desse modelo mostra altos níveis de precisão e recall para quando o paciente não tem a condição, quando ela é real a precisão e recall cai drasticamente. 

Já a curva ROC, utiliza o AUC e analisa a frequência de que os verdadeiros positivos são descobertos, e de acordo com essa curva, mesmo com uma taxa baixa de 0.2 falsos positivos, a curva se mantém acima de um modelo impreciso, com certa de 63% de acuracidade.

4.b Em um contesto de triagem de saúde pública, pacientes que não tiverem a condição estão com maiores changes de sair com uma predição acurada dos dados, e com uma pequena chance de falso positivo, em contrapartida, o alto índice de falsos negativos é alarmante, ouve um erro em focar apenas em diagnósticos de falsos e os diagnósticos de verdadeiros estão faltando. O que nesse contexto nos mostra um problema considerando o risco de diagnosticar um falso negativo ser superior à falso positivos.

4.c A busca por hiperparâmetros ou GridSearchCV como o nome da biblioteca no sklearn. Para a busca é necessário focar em cobrir diferentes áreas para que seja realizado uma otimização e melhora principalmente nos casos de falsos negativos. L1 e L2 como atributos de penalidade ajudam o algoritmo à explorar diferentes estratégias de regularização, a intensidade da regularização é controlada por um bom intervalo e bem amplo, foram utilizados 2 solver para maior cobertura dos dados.

4.d Sobre os resultados, após focar melhor na resolução de casos reais de diabetes, o modelo apresentou um aumento que partiu de 12% anteriormente para 76%, na medida de recall, apesar da menor acurárica na categoria precisão, o modelo começou a classificar mais casos de diabetes 1 como realmente 1, causando o balanceamento de precision e recall aumentar em 24% porcento. Tudo isso sem tornar o modelo obsoleto, considerando que o índice AUC se manteve inclusive teve um aumento de 1%.

Com o índice AUC se mantendo estável ou com um pequeno acréscimo, a curva ROC se mantém a mesma, e indica um modelo preciso com alta precisão em média. 
