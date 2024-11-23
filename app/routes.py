import os
from flask import render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

def init_routes(app):
    @app.route("/")
    def home():
        return render_template("upload.html", title="Upload de Dados")

    @app.route("/analysis", methods=["GET", "POST"])
    def analysis():
        if request.method == "POST":
            file = request.files.get("file")
            if not file or not file.filename.endswith(".csv"):
                return render_template("upload.html", title="Upload de Dados", message="Por favor, faça o upload de um arquivo CSV válido.")

            df = pd.read_csv(file)
            features = ['sugar_content_brix', 'firmness_kgf', 'length_cm']
            category_features = ['sugar_content_brix', 'firmness_kgf', 'ripeness_index']

            image_dir = os.path.join("static", "images")
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            generated_charts = generate_graphs(df, image_dir)

            return render_template("analysis.html", 
                                   tables=[df.describe().to_html()], 
                                   features=features, 
                                   category_features=category_features, 
                                   charts=generated_charts)

        return render_template("upload.html", title="Upload de Dados")

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json
        model = load_model()
        input_data = [data[feature] for feature in model["features"]]
        prediction = model["model"].predict([input_data])
        return jsonify({"prediction": prediction.tolist()})

    @app.route("/predict_category", methods=["POST"])
    def predict_category():
        data = request.json
        model = load_model(model_name="classification_model.joblib")
        input_data = [data[feature] for feature in model["features"]]
        prediction = model["model"].predict([input_data])
        return jsonify({"prediction": prediction.tolist()})

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

        return jsonify({"status": "success", 
                        "message": "Modelos re-treinados e gráficos atualizados com sucesso!", 
                        "charts": generated_charts})

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
    charts = {}
    charts['bar_chart'] = create_basic_bar_chart(df, image_dir)
    charts['pie_chart'] = create_pie_chart(df, image_dir)
    charts['heatmap'] = create_advanced_heatmap(df, image_dir)
    charts['scatter_plot'] = create_scatter_plot(df, image_dir)
    charts['boxplot'] = create_boxplot(df, image_dir)
    return charts

def create_basic_bar_chart(df, image_dir):
    plt.figure(figsize=(8, 5))
    if 'variety' in df.columns:
        df['variety'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribuição por Variedade')
        plt.xlabel('Variedade')
        plt.ylabel('Contagem')
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = 'bar_chart.png'
        plt.savefig(os.path.join(image_dir, filename))
        plt.close()
        return filename
    return ''

def create_pie_chart(df, image_dir):
    plt.figure(figsize=(6, 6))
    if 'ripeness_category' in df.columns:
        df['ripeness_category'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Proporção de Categorias de Maturação')
        plt.ylabel('')
        plt.tight_layout()
        filename = 'pie_chart.png'
        plt.savefig(os.path.join(image_dir, filename))
        plt.close()
        return filename
    return ''

def create_advanced_heatmap(df, image_dir):
    plt.figure(figsize=(10, 8))
    if 'quality_category' in df.columns and 'ripeness_index' in df.columns:
        df_grouped = df.groupby('quality_category')['ripeness_index'].mean().reset_index()
        heatmap_data = pd.pivot_table(df_grouped, values='ripeness_index', index=['quality_category'])
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
        plt.title('Média do Índice de Maturação por Categoria de Qualidade')
        plt.xlabel('Índice de Maturação')
        plt.ylabel('')
        plt.tight_layout()
        filename = 'heatmap.png'
        plt.savefig(os.path.join(image_dir, filename))
        plt.close()
        return filename
    return ''

def create_scatter_plot(df, image_dir):
    plt.figure(figsize=(8, 5))
    if 'quality_score' in df.columns and 'sugar_content_brix' in df.columns:
        plt.scatter(df['quality_score'], df['sugar_content_brix'], alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.title('Pontuação de Qualidade vs. Teor de Açúcar')
        plt.xlabel('Pontuação de Qualidade')
        plt.ylabel('Teor de Açúcar (Brix)')
        plt.tight_layout()
        filename = 'scatter_plot.png'
        plt.savefig(os.path.join(image_dir, filename))
        plt.close()
        return filename
    return ''

def create_boxplot(df, image_dir):
    plt.figure(figsize=(10, 6))
    if 'firmness_kgf' in df.columns and 'quality_category' in df.columns:
        sns.boxplot(data=df, x='quality_category', y='firmness_kgf')
        plt.title('Distribuição da Firmeza por Categoria de Qualidade')
        plt.xlabel('Categoria de Qualidade')
        plt.ylabel('Firmeza (kgf)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = 'boxplot.png'
        plt.savefig(os.path.join(image_dir, filename))
        plt.close()
        return filename
    return ''
