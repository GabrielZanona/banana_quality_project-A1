import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import render_template, request, jsonify
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib


def init_routes(app):
    @app.route("/")
    def home():
        return render_template("upload.html", title="Upload de Dados")

    @app.route("/analysis", methods=["POST"])
    def analysis():
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            return render_template("upload.html", title="Upload de Dados", message="Por favor, faça o upload de um arquivo CSV válido.")

        df = pd.read_csv(file)
        image_dir = os.path.join("static", "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        generated_charts = generate_graphs(df, image_dir)
        regression_model, features = train_regression_model(df)
        classification_model, category_features = train_classification_model(df)

        table_html = df.head(10).iloc[:, :10].to_html(classes="table table-striped", index=False)

        return render_template(
            "analysis.html",
            title="Análise de Dados",
            tables=[table_html],
            features=features,
            category_features=category_features,
            charts=generated_charts
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json() 
        model_data = load_model("model.joblib")
        model = model_data["model"]
        features = model_data["features"]
        

        input_data = [data[feature] for feature in features]
        prediction = model.predict([input_data])
        
        return jsonify({"prediction": prediction[0]})

    @app.route("/predict_category", methods=["POST"])
    def predict_category():
        data = request.get_json()  
        model_data = load_model("classification_model.joblib")
        model = model_data["model"]
        features = model_data["features"]
        

        input_data = [data[feature] for feature in features]
        prediction = model.predict([input_data])
        
        return jsonify({"prediction": prediction[0]})

    @app.route("/retrain", methods=["POST"])
    def retrain_models():
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            return jsonify({"status": "error", "message": "Nenhum arquivo CSV válido fornecido."}), 400

        df = pd.read_csv(file)
        image_dir = os.path.join("static", "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        generated_charts = generate_graphs(df, image_dir)
        regression_model, features = train_regression_model(df)
        classification_model, category_features = train_classification_model(df)

        table_html = df.head(10).iloc[:, :10].to_html(classes="table table-striped", index=False)

        return jsonify({
            "status": "success",
            "message": "Modelos re-treinados e gráficos atualizados com sucesso!",
            "charts": generated_charts,
            "table_html": table_html
        })


def train_regression_model(df):
    features = ['sugar_content_brix', 'firmness_kgf', 'length_cm']
    target = 'quality_score'
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    save_model(model, features)
    return model, features

def train_classification_model(df):
    features = ['sugar_content_brix', 'firmness_kgf', 'ripeness_index']
    target = 'quality_category'
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    save_model(model, features, model_name="classification_model.joblib")
    return model, features

def save_model(model, features, model_name="model.joblib"):
    model_data = {"model": model, "features": features}
    joblib.dump(model_data, model_name)

def load_model(model_name="model.joblib"):
    return joblib.load(model_name)

def generate_graphs(df, image_dir):
    charts = {
        'bar_chart': create_bar_chart(df, image_dir),
        'pie_chart': create_pie_chart(df, image_dir),
        'scatter_plot': create_scatter_plot(df, image_dir),
        'boxplot': create_boxplot(df, image_dir),
        'heatmap': create_heatmap(df, image_dir)
    }
    return charts

def create_bar_chart(df, image_dir):
    plt.figure(figsize=(8, 5))
    df['variety'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribuição por Variedade')
    plt.xlabel('Variedade')
    plt.ylabel('Contagem')
    plt.tight_layout()
    filename = "bar_chart.png"
    plt.savefig(os.path.join(image_dir, filename))
    plt.close()
    return filename

def create_pie_chart(df, image_dir):
    plt.figure(figsize=(6, 6))
    df['ripeness_category'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Proporção de Categorias de Maturação')
    plt.tight_layout()
    filename = "pie_chart.png"
    plt.savefig(os.path.join(image_dir, filename))
    plt.close()
    return filename

def create_scatter_plot(df, image_dir):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['quality_score'], df['sugar_content_brix'], alpha=0.6, edgecolors='w')
    plt.title('Pontuação de Qualidade vs. Teor de Açúcar')
    plt.xlabel('Pontuação de Qualidade')
    plt.ylabel('Teor de Açúcar')
    plt.tight_layout()
    filename = "scatter_plot.png"
    plt.savefig(os.path.join(image_dir, filename))
    plt.close()
    return filename

def create_boxplot(df, image_dir):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='quality_category', y='firmness_kgf')
    plt.title('Distribuição da Firmeza por Categoria de Qualidade')
    plt.xlabel('Categoria de Qualidade')
    plt.ylabel('Firmeza (kgf)')
    plt.tight_layout()
    filename = "boxplot.png"
    plt.savefig(os.path.join(image_dir, filename))
    plt.close()
    return filename

def create_heatmap(df, image_dir):
    plt.figure(figsize=(10, 8))
    heatmap_data = pd.pivot_table(df, values='ripeness_index', index='quality_category', aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
    plt.title('Média do Índice de Maturação por Categoria de Qualidade')
    plt.tight_layout()
    filename = "heatmap.png"
    plt.savefig(os.path.join(image_dir, filename))
    plt.close()
    return filename
