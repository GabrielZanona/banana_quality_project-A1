Mais detalhes na documentação do projeto, está junto ao projeto no git Documento Projeto Bananálise.docx

Rodar o projeto : 

instalando os requirements :

Crie um ambiente virtual:

1. python -m venv venv

2. venv\Scripts\activate     

3. pip install -r requirements.txt

4. python app.py


ou instalando as dependências direto:

1. pip install flask pandas matplotlib seaborn scikit-learn joblib

2. python app.py


Funcionalidades

Upload de Arquivo CSV:

Faça o upload de um arquivo CSV contendo os dados para análise.
Resumo dos Dados:
Exibe uma tabela com as primeiras 10 linhas e colunas do dataset.

Visualizações Gráficas:
Gráficos como barras, pizza, dispersão, boxplot e heatmap para explorar os dados.
Previsão:

Pontuação de Qualidade: Previsão da qualidade com base nas características.

Categoria de Qualidade: Classifica a qualidade em categorias (e.g., alta, média, baixa).

Re-treinamento:
Envie novos datasets para atualizar os modelos e gráficos.
