import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Carregar modelo e scaler
model = joblib.load('../models/futebol_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Carregar estatísticas médias das equipas
home_stats = pd.read_csv('../models/home_stats.csv', index_col=0)
away_stats = pd.read_csv('../models/away_stats.csv', index_col=0)


def prever_jogo(home, away):
    if home not in home_stats.index or away not in away_stats.index:
        raise ValueError("Uma das equipas não foi encontrada nas estatísticas.")

    home_values = home_stats.loc[home].values  # ex: 6 valores
    away_values = away_stats.loc[away].values  # ex: 6 valores
    differences = home_values - away_values  # 6 valores
    input_vector = np.concatenate([home_values, away_values, differences])  # total: 18

    # Normalizar os dados da mesma forma que foi feito no treino
    scaler = joblib.load('../models/scaler.pkl')
    input_vector_scaled = scaler.transform([input_vector])

    # Prever as probabilidades
    model = joblib.load('../models/futebol_model.pkl')
    probabilities = model.predict_proba(input_vector_scaled)[0]

    return probabilities

def prever_lista_de_jogos(lista_de_jogos):
    print("\n🔮 Previsões para os jogos da próxima jornada:\n")
    for home, away in lista_de_jogos:
        try:
            probabilidades = prever_jogo(home, away)
            print(f"{home} vs {away}")
            print(f"🏠 Vitória da casa: {probabilidades[2]*100:.1f}%")
            print(f"🤝 Empate: {probabilidades[1]*100:.1f}%")
            print(f"🚗 Vitória visitante: {probabilidades[0]*100:.1f}%\n")
        except ValueError as e:
            print(f"Erro com o jogo {home} vs {away}: {e}\n")

if __name__ == '__main__':
    modo = input("Escolher modo (1: jogo individual, 2: próxima jornada): ")

    if modo == '1':
        home = input("Nome da equipa da casa: ")
        away = input("Nome da equipa visitante: ")

        try:
            previsao = prever_jogo(home, away)
            print(f"\n🔮 Probabilidades do jogo {home} vs {away}:")
            print(f"🏠 Vitória da casa ({home}): {previsao[2]*100:.1f}%")
            print(f"🤝 Empate: {previsao[1]*100:.1f}%")
            print(f"🚗 Vitória visitante ({away}): {previsao[0]*100:.1f}%")
        except ValueError as e:
            print(f"Erro: {e}")

    elif modo == '2':
        jogos_jornada = [
            ("Arouca", "Famalicao"),
            ("Guimaraes", "Santa Clara"),
            ("Gil Vicente", "Moreirense"),
            ("Nacional", "Estrela"),
            ("Porto", "Benfica"),
            ("Farense", "Casa Pia"),
            ("Rio Ave", "Boavista"),
            ("Sp Lisbon", "Sp Braga"),
        ]
        prever_lista_de_jogos(jogos_jornada)

    else:
        print("Opção inválida.")
