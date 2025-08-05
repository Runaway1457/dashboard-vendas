import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesPredictor:
    """
    Modelo de Machine Learning para previsão de vendas
    Utiliza Random Forest para prever vendas futuras baseado em dados históricos
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepara features para o modelo de ML"""
        logger.info("Preparando features para o modelo...")
        
        features = pd.DataFrame()
        
        # Features temporais
        features['mes'] = df['data_venda'].dt.month
        features['ano'] = df['data_venda'].dt.year
        features['dia_semana'] = df['data_venda'].dt.dayofweek
        features['trimestre'] = df['data_venda'].dt.quarter
        
        # Features de lag (vendas anteriores)
        features['vendas_mes_anterior'] = df.groupby('mes_ano')['valor_venda'].sum().shift(1)
        features['media_movel_3m'] = df.groupby('mes_ano')['valor_venda'].sum().rolling(3).mean()
        features['tendencia'] = df.groupby('mes_ano')['valor_venda'].sum().pct_change()
        
        # Features sazonais
        features['is_black_friday'] = ((df['data_venda'].dt.month == 11) & 
                                      (df['data_venda'].dt.day >= 20)).astype(int)
        features['is_natal'] = (df['data_venda'].dt.month == 12).astype(int)
        features['is_carnaval'] = ((df['data_venda'].dt.month == 2) | 
                                  (df['data_venda'].dt.month == 3)).astype(int)
        
        logger.info(f"Features criadas: {features.shape[1]} variáveis")
        return features
    
    def train_model(self, X_train, y_train):
        """Treina o modelo de previsão"""
        logger.info("Iniciando treinamento do modelo...")
        
        # Remover valores NaN
        mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]
        
        # Treinar modelo
        self.model.fit(X_train_clean, y_train_clean)
        self.is_trained = True
        
        # Avaliar performance
        train_pred = self.model.predict(X_train_clean)
        mae = mean_absolute_error(y_train_clean, train_pred)
        r2 = r2_score(y_train_clean, train_pred)
        
        logger.info(f"Modelo treinado - MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return {
            'mae': mae,
            'r2_score': r2,
            'feature_importance': dict(zip(X_train.columns, self.model.feature_importances_))
        }
    
    def predict(self, X_test):
        """Faz previsões com o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado antes de fazer previsões")
        
        logger.info(f"Fazendo previsões para {X_test.shape[0]} registros...")
        predictions = self.model.predict(X_test)
        
        return predictions
    
    def get_feature_importance(self):
        """Retorna importância das features"""
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
        
        return self.model.feature_importances_

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    logger.info("Sales Prediction Model - Demonstração")
    
    # Criar dados simulados
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    n_samples = len(dates)
    
    sample_data = pd.DataFrame({
        'data_venda': dates,
        'valor_venda': np.random.uniform(1000, 5000, n_samples) * (1 + 0.1 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)),
        'mes_ano': dates.to_period('M')
    })
    
    # Instanciar e demonstrar modelo
    predictor = SalesPredictor()
    
    print("✅ Modelo de previsão de vendas inicializado com sucesso!")
    print("📊 Pronto para treinamento com dados reais de vendas")
    print("🚀 Execute com dados reais para obter previsões precisas")