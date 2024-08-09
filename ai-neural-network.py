from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Classe para centralizar os gráficos
class Plotter:
    @staticmethod
    def plot_confusion_matrix(cm, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.show()

    @staticmethod
    def plot_learning_curve(loss_curve):
        plt.plot(loss_curve)
        plt.title('Curva de Aprendizagem')
        plt.xlabel('Iterações')
        plt.ylabel('Perda')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Curva ROC (área = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.show()

# Classe para o modelo de rede neural
class BreastCancerModel:
    def __init__(self):
        self.data = load_breast_cancer()
        self.x_train, self.x_test, self.y_train, self.y_test = self._prepare_data()
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=500)
        
    def _prepare_data(self):
        x, y = self.data["data"], self.data["target"]
        return train_test_split(x, y)

    def scale_data(self):
        self.scaler.fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def get_confusion_matrix(self, predictions):
        return confusion_matrix(self.y_test, predictions)

    def get_roc_data(self, predictions):
        y_prob = self.model.predict_proba(self.x_test)[:,1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def print_classification_report(self, predictions):
        print(classification_report(self.y_test, predictions))

# Função principal
def main():
    # Inicializa o modelo
    model = BreastCancerModel()
    model.scale_data()
    model.train()

    # Faz previsões
    predictions = model.predict()

    # Matriz de confusão
    cm = model.get_confusion_matrix(predictions)
    Plotter.plot_confusion_matrix(cm, model.data.target_names)

    # Curva de aprendizado
    Plotter.plot_learning_curve(model.model.loss_curve_)

    # Curva ROC
    fpr, tpr, roc_auc = model.get_roc_data(predictions)
    Plotter.plot_roc_curve(fpr, tpr, roc_auc)

    # Relatório de classificação
    model.print_classification_report(predictions)

if __name__ == "__main__":
    main()
