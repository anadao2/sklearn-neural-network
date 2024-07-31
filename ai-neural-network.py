from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

cancer = load_breast_cancer()
x,y = cancer["data"], cancer["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=500)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

cm=confusion_matrix(y_test, predictions)

# Visualizar matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Visualizar curva de aprendizagem
plt.plot(mlp.loss_curve_)
plt.title('Curva de Aprendizagem')
plt.xlabel('Iterações')
plt.ylabel('Perda')
plt.grid(True)
plt.show()

# Obter probabilidades de previsão
y_prob = mlp.predict_proba(x_test)[:,1]

# Calcular fpr, tpr e thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calcular a AUC
roc_auc = auc(fpr, tpr)

# Plotar curva ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

print(classification_report(y_test, predictions))






