# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from data_processing import load_data
from database import get_db


def train_random_forest():
    # Conecte-se ao banco de dados e carregue os dados
    db = next(get_db())
    data = load_data(db)

    # Printar os dados
    print(data)

    # Definir as features e o target (atraso da sprint)
    X = data.drop(columns=['cod_sprint'])  # Features
    y = (data['timespent'] > data['time_estimate']).astype(int)  # Target (1 = atrasado, 0 = no prazo)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar o modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Salvar o modelo treinado
    joblib.dump(model, 'random_forest_model.pkl')


if __name__ == "__main__":
    train_random_forest()
