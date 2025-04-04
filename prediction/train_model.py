import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Caminho para o CSV
DATA_PATH = '../data/liga_portugal_24_25.csv'

# Carregar dados
df = pd.read_csv(DATA_PATH)

# Converter resultado em número (0 = fora, 1 = empate, 2 = casa)
result_map = {'A': 0, 'D': 1, 'H': 2}
df['FTR'] = df['FTR'].map(result_map)

# Limpar valores nulos
df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'], inplace=True)

# Lista de features estatísticas (apenas 12 features, sem as diferenças ou outras)
stat_features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY']
df[stat_features] = df[stat_features].fillna(0)

# Função para calcular médias por equipa e tipo de jogo
def get_team_stats(df):
    stats_home = df.groupby('HomeTeam')[['HS', 'HST', 'HC', 'HY']].mean()
    stats_home.columns = [f'{col}_home_avg' for col in stats_home.columns]

    stats_away = df.groupby('AwayTeam')[['AS', 'AST', 'AC', 'AY']].mean()
    stats_away.columns = [f'{col}_away_avg' for col in stats_away.columns]

    return stats_home, stats_away

home_stats, away_stats = get_team_stats(df)

# Construir dataset para treino
rows = []
for _, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    result = row['FTR']

    if home in home_stats.index and away in away_stats.index:
        home_values = home_stats.loc[home].values
        away_values = away_stats.loc[away].values
        combined = np.concatenate([home_values, away_values])  # Somente as estatísticas essenciais
        rows.append((home, away, combined, result))

# Separar features e labels
X = np.array([r[2] for r in rows])
y = np.array([r[3] for r in rows])

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar desempenho
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar modelo e scaler
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/futebol_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

# Guardar médias das equipas
home_stats.to_csv('../models/home_stats.csv')
away_stats.to_csv('../models/away_stats.csv')

print("✅ Modelo treinado e guardado com sucesso.")
