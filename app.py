import streamlit as st
import joblib
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Previsão de Sobrevivência - Titanic", page_icon="🚢")

st.title("🚢 Previsão de Sobrevivência - Titanic")
st.write("Insira os dados do passageiro para prever se ele sobreviveria ao naufrágio do Titanic.")

st.divider()
st.subheader("📊 Análise Exploratória de Dados")
st.write("Abaixo podemos ver os dados utilizados no treinamento (Amostra):")
df_view = pd.read_csv("dataset.csv")
st.dataframe(df_view[['Name', 'Sex', 'Age', 'Pclass', 'Survived']].head(5))

st.write("**Taxa de Sobrevivência por Classe do Navio**")
sobrevivencia_classe = df_view.groupby('Pclass')['Survived'].mean() * 100
st.bar_chart(sobrevivencia_classe)

st.divider()
st.subheader("🔮 Fazer uma Previsão")

# Carregar modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Erro ao carregar o modelo. Certifique-se de executar o train.py antes.")
    st.stop()

# Formulário de entrada
col1, col2 = st.columns(2)

with col1:
    idade = st.number_input("Idade", min_value=0.0, max_value=120.0, value=30.0)
    sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    classe = st.selectbox("Classe do Navio (Pclass)", [1, 2, 3])

with col2:
    irmaos_conjuges = st.number_input("Número de irmãos/cônjuges (SibSp)", min_value=0, max_value=10, value=0)
    tarifa = st.number_input("Tarifa (Fare)", min_value=0.0, value=32.0)

if st.button("Prever Sobrevivência"):
    # Converter sexo para o formato numérico esperado pelo modelo (male: 0, female: 1)
    sexo_num = 1 if sexo == "Feminino" else 0
    
    dados = pd.DataFrame([[idade, sexo_num, classe, irmaos_conjuges, tarifa]], 
                         columns=['Age', 'Sex', 'Pclass', 'SibSp', 'Fare'])
    
    predicao = model.predict(dados)[0]
    
    if predicao == 1:
        st.success("🎉 O passageiro SOBREVIVERIA!")
    else:
        st.error("💔 O passageiro NÃO sobreviveria.")
