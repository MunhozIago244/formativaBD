import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math

dados_produtos = pd.read_csv('C:\\Users\\microc\\Desktop\\code\\bd e ia\\formativa\\formativaBD\\dados_produtos.csv')

sns.set(style="darkgrid")
palette = sns.color_palette("rocket")

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(x='product_name', y='rating', data=dados_produtos, ax=ax, palette=palette)

ax.set_title('Relatório de Avaliação por Produto', fontsize=16)
ax.set_xlabel('Produto', fontsize=12)
ax.set_ylabel('Notas de Avaliação', fontsize=12)

plt.xticks(rotation=45, ha="right")

plt.show()

print("Informações do Conjunto de Dados:")
dados_produtos.info()

print("\nEstatísticas do Conjunto de Dados:")
print(dados_produtos.describe())

X = dados_produtos.drop('purchased', axis=1)
y = dados_produtos['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tamanho do conjunto de treinamento:", X_train.shape[0])
print("Tamanho do conjunto de teste:", X_test.shape[0])

X_train = pd.get_dummies(X_train, columns=['product_name'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['product_name'], drop_first=True)

unique_products = dados_produtos['product_name'].unique()

product_models = {}

n_cols = 3
n_rows = math.ceil(len(unique_products) / n_cols)

for page in range(math.ceil(len(unique_products) / (n_cols * n_rows))):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    products_on_page = unique_products[page * n_cols * n_rows: (page + 1) * n_cols * n_rows]
    
    for i, product in enumerate(products_on_page):
        if f'product_name_{product}' not in X_train.columns:
            print(f"Não há amostras suficientes para treinar um modelo para o produto '{product}'. Pulando este produto.")
            continue
        
        X_train_product = X_train[X_train[f'product_name_{product}'] == 1]
        y_train_product = y_train[X_train[f'product_name_{product}'] == 1]
        
        X_test_product = X_test[X_test[f'product_name_{product}'] == 1]
        y_test_product = y_test[X_test[f'product_name_{product}'] == 1]
        
        if X_train_product.shape[0] < 1:
            print(f"Não há amostras suficientes para treinar um modelo para o produto '{product}'. Pulando este produto.")
            continue
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train_product, y_train_product)
        
        y_pred_product = clf.predict(X_test_product)
        
        accuracy = accuracy_score(y_test_product, y_pred_product)
        precision = precision_score(y_test_product, y_pred_product)
        recall = recall_score(y_test_product, y_pred_product)
        f1 = f1_score(y_test_product, y_pred_product)
        confusion = confusion_matrix(y_test_product, y_pred_product)
        
        product_models[product] = {
            'model': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion
        }

        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        plot_tree(clf, filled=True, feature_names=X_train.columns.tolist(), class_names=['0', '1'], rounded=True, ax=ax)
        ax.set_title(f"Árvore de Decisão para o Produto '{product}'")
        
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')

    for i in range(len(products_on_page), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

dados_agrupados = dados_produtos.groupby('product_name').agg({'rating': 'mean', 'rating_count': 'sum'}).reset_index()

dados_agrupados['metrica'] = dados_agrupados['rating'] * dados_agrupados['rating_count']

correlation_matrix = dados_agrupados[['rating', 'rating_count']].corr()

correlation_values = correlation_matrix.iloc[0, 1]

metrica_correlacao = dados_agrupados['rating_count'] * (1 + correlation_values) * dados_agrupados['rating']

dados_agrupados = dados_agrupados.sort_values(by='metrica', ascending=False)

plt.figure(figsize=(10, 6))
plt.plot(range(len(dados_agrupados)), dados_agrupados['metrica'], marker='o', linestyle='-')
plt.xlabel('Produtos')
plt.ylabel('Métrica Personalizada')
plt.title('Curva de Métrica Personalizada por Produto')
plt.xticks(range(len(dados_agrupados)), dados_agrupados['product_name'], rotation=90)
plt.grid(True)

plt.show()
