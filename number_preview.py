import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class PricePredictor:
    def __init__(self, n_estimators=100):
        """
        Inicializa a classe PricePredictor com um modelo RandomForestRegressor.
        
        :param n_estimators: Número de árvores na floresta aleatória.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def train(self, historical_prices, future_prices):
        """
        Treina o modelo de previsão de preços com dados históricos.
        
        :param historical_prices: Um array numpy com os preços históricos.
        :param future_prices: Um array numpy com os preços futuros correspondentes.
        """
        self.model.fit(historical_prices.reshape(-1, 1), future_prices)

    def predict(self, next_price):
        """
        Prediz o preço para o próximo período com base no modelo treinado.
        
        :param next_price: O preço atual ou futuro a ser previsto.
        :return: O preço previsto para o próximo período.
        """
        predicted_price = self.model.predict(np.array([[next_price]]))
        return predicted_price[0]

    def plot_prediction(self, historical_prices, future_prices, next_price, predicted_price):
        """
        Plota os preços históricos, futuros e previstos em um gráfico.
        
        :param historical_prices: Um array numpy com os preços históricos.
        :param future_prices: Um array numpy com os preços futuros correspondentes.
        :param next_price: O preço atual ou futuro a ser previsto.
        :param predicted_price: O preço previsto pelo modelo.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(historical_prices, future_prices, 'bo-', label='Historical Prices')
        plt.plot([historical_prices[-1], next_price], [future_prices[-1], predicted_price], 'ro--', label='Predicted Price')
        plt.xlabel('Historical Prices')
        plt.ylabel('Future Prices')
        plt.title('Price Prediction using RandomForestRegressor')
        plt.legend()
        plt.grid(True)
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Dados históricos simulados
    historical_prices = np.array([125924, 122098, 123907, 127652, 131205])
    future_prices = np.array([127350, 122627, 125401, 128123, 131590])
    
    # Inicializando o previsor de preços
    predictor = PricePredictor(n_estimators=100)
    
    # Treinando o modelo com os dados históricos
    predictor.train(historical_prices, future_prices)
    
    # Prevendo o próximo preço
    next_price = 131590
    predicted_price = predictor.predict(next_price)
    
    # Plotando os resultados
    predictor.plot_prediction(historical_prices, future_prices, next_price, predicted_price)
