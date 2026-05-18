import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Iniciando o treinamento do modelo Titanic...")
df = pd.read_csv("dataset.csv")

# Pré-processamento mínimo
# Apagar colunas desnecessárias para o modelo (conforme pedido)
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], errors='ignore')

# Preencher idades vazias com a mediana
df['Age'] = df['Age'].fillna(df['Age'].median())

# Converter Sexo para numérico
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

cols = ['Age', 'Sex', 'Pclass', 'SibSp', 'Fare']
X = df[cols]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Acurácia: {acc:.2f}")

joblib.dump(model, "modelo.pkl")
print("Modelo salvo como modelo.pkl")
