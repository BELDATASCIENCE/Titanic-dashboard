
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Configure the Streamlit page
st.set_page_config(
    page_title='Titanic Survival Dashboard',
    page_icon='🚢',
    layout='wide'
)

# 2. Function to load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_final.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: 'model_final.joblib' no encontrado. Asegúrese de que el modelo esté entrenado y guardado.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

# 3. Function to recreate and fit the StandardScaler
@st.cache_resource
def fit_scaler():
    try:
        df_original_for_scaler = pd.read_csv('titanic_data.csv')
        
        # Apply the same preprocessing steps as during training
        df_original_for_scaler['Embarked'] = df_original_for_scaler['Embarked'].fillna(df_original_for_scaler['Embarked'].mode()[0])
        df_original_for_scaler['Age'] = df_original_for_scaler['Age'].fillna(df_original_for_scaler['Age'].median())
        
        q_low_scaler = df_original_for_scaler['Fare'].quantile(0.01)
        q_high_scaler = df_original_for_scaler['Fare'].quantile(0.99)
        df_original_for_scaler = df_original_for_scaler[(df_original_for_scaler['Fare'] >= q_low_scaler) & (df_original_for_scaler['Fare'] <= q_high_scaler)]

        df_original_for_scaler['FamilySize'] = df_original_for_scaler['SibSp'] + df_original_for_scaler['Parch'] + 1
        df_original_for_scaler['Sex'] = df_original_for_scaler['Sex'].map({'male':0,'female':1})
        df_original_for_scaler['Embarked'] = df_original_for_scaler['Embarked'].map({'S':0,'C':1,'Q':2})
        
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
        
        scaler = StandardScaler()
        scaler.fit(df_original_for_scaler[features])
        return scaler
    except FileNotFoundError:
        st.error("Error: 'titanic_data.csv' no encontrado. Asegúrese de que el conjunto de datos original esté disponible.")
        st.stop()
    except Exception as e:
        st.error(f"Error al preparar el escalador: {e}")
        st.stop()

# 4. Function to load Inferencia.csv
@st.cache_data
def load_inference_data():
    try:
        df_inferencia = pd.read_csv('Inferencia.csv')
        return df_inferencia
    except FileNotFoundError:
        st.warning("Advertencia: 'Inferencia.csv' no encontrado. Algunas características del panel pueden estar limitadas.")
        return pd.DataFrame() # Return empty DataFrame if not found
    except Exception as e:
        st.error(f"Error al cargar datos de inferencia: {e}")
        st.stop()

# Function to categorize age
def categorize_age(age):
    if age <= 12:
        return 'Niño (0-12)'
    elif 13 <= age <= 17:
        return 'Adolescente (13-17)'
    elif 18 <= age <= 35:
        return 'Joven (18-35)'
    elif 36 <= age <= 60:
        return 'Adulto (36-60)'
    else:
        return 'Adulto Mayor (60+)'

# Main application logic
st.title('🚢 Titanic: Panel de predicción de supervivencia')

# Load model, scaler, and inference data
model = load_model()
scaler = fit_scaler()
df_inferencia = load_inference_data()

# Features used for prediction (must match training order)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']

if not df_inferencia.empty:
    # Sidebar for filters
    st.sidebar.header('Filtros para datos predichos')

    # Gender filter
    gender_options = {'Masculino': 0, 'Femenino': 1}
    selected_genders = st.sidebar.multiselect(
        'Seleccionar Género(s)',
        list(gender_options.keys()),
        default=list(gender_options.keys())
    )
    selected_gender_values = [gender_options[g] for g in selected_genders]

    # Pclass filter
    pclass_options = {1: '1ra Clase', 2: '2da Clase', 3: '3ra Clase'}
    selected_pclasses = st.sidebar.multiselect(
        'Seleccionar Clase de Pasajero',
        list(pclass_options.keys()),
        default=list(pclass_options.keys()),
        format_func=lambda x: pclass_options[x]
    )

    # Age Range filter
    min_age, max_age = int(df_inferencia['Age'].min()), int(df_inferencia['Age'].max())
    age_range = st.sidebar.slider(
        'Seleccionar Rango de Edad',
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    # Apply filters
    filtered_df = df_inferencia[
        (df_inferencia['Sex'].isin(selected_gender_values)) &
        (df_inferencia['Pclass'].isin(selected_pclasses)) &
        (df_inferencia['Age'] >= age_range[0]) &
        (df_inferencia['Age'] <= age_range[1])
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    st.subheader('📄Muestra de datos predichos')
    st.dataframe(filtered_df.head(5))

    st.markdown("<h3 style='text-align: center; color: #007bff;'>👥 Análisis de la tasa de supervivencia por Grupo de Edad y Género</h3>", unsafe_allow_html=True)

    if not filtered_df.empty:
        filtered_df['AgeGroup'] = filtered_df['Age'].apply(categorize_age)
        gender_map_reverse = {0: 'Masculino', 1: 'Femenino'}
        filtered_df['Sex_Label'] = filtered_df['Sex'].map(gender_map_reverse)
        
        # Convert 'Survival_Probability' from string 'xx.yy%' to float
        filtered_df['Survival_Probability_Numeric'] = filtered_df['Survival_Probability'].str.replace('%', '', regex=False).astype(float)

        # Calculate survival rates by AgeGroup and Sex_Label using Survival_Probability_Numeric
        survival_pivot = pd.pivot_table(filtered_df,
                                        values='Survival_Probability_Numeric',
                                        index='AgeGroup',
                                        columns='Sex_Label',
                                        aggfunc='mean',
                                        margins=True,
                                        margins_name='Total') # Add row and column totals
        
        # Format as percentage with one decimal
        survival_pivot = (survival_pivot).round(1).astype(str) + '%'
        survival_pivot = survival_pivot.replace('nan%', 'N/A') # Handle NaN values for combinations with no data
        
        st.dataframe(survival_pivot)
    else:
        st.info("No hay datos filtrados para mostrar la tasa de supervivencia por grupo de edad y género.")


    # Visualizations
    st.subheader('📊Visualizaciones de Predicción de Supervivencia')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de predicciones por género
        gender_map_reverse = {0: 'Masculino', 1: 'Femenino'}
        filtered_df['Sex_Label'] = filtered_df['Sex'].map(gender_map_reverse)
        fig_gender = px.histogram(filtered_df, x='Survived_Prediction', color='Sex_Label',
                                  title='Supervivencia Prevista por Género',
                                  labels={'Survived_Prediction': 'Supervivencia Prevista (0=No, 1=Sí)', 'Sex_Label': 'Género'},
                                  category_orders={'Survived_Prediction': [0, 1]},
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  barmode='group')
        st.plotly_chart(fig_gender, use_container_width=True)

        # Gráfico de barras de la tasa de supervivencia por Puerto de Embarque
        embarked_map = {0: 'Southampton (S)', 1: 'Cherbourg (C)', 2: 'Queenstown (Q)'}
        filtered_df['Embarked_Label'] = filtered_df['Embarked'].map(embarked_map)
        df_embarked_survival = filtered_df.groupby('Embarked_Label')['Survived_Prediction'].mean().reset_index()
        df_embarked_survival['Survival_Rate'] = (df_embarked_survival['Survived_Prediction'] * 100).round(1)
        
        if not df_embarked_survival.empty:
            fig_embarked = px.bar(df_embarked_survival, x='Embarked_Label', y='Survival_Rate', 
                                  title='Tasa de Supervivencia Prevista por Puerto de Embarque',
                                  labels={'Embarked_Label': 'Puerto de Embarque', 'Survival_Rate': 'Tasa de Supervivencia (%)'},
                                  color='Embarked_Label',
                                  color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_embarked, use_container_width=True)
        else:
            st.info("No se predijeron supervivientes para ningún puerto de embarque en los datos filtrados.")

    with col2:
        # Gráfico de línea de las predicciones (Survival_Probability)
        # Ensure 'Survival_Probability_Numeric' is available from the combined table logic above
        # If not, recreate it here:
        if 'Survival_Probability_Numeric' not in filtered_df.columns:
            filtered_df['Survival_Probability_Numeric'] = filtered_df['Survival_Probability'].str.replace('%', '', regex=False).astype(float)
        
        fig_line_proba = px.line(filtered_df.sort_values('Age'), x='Age', y='Survival_Probability_Numeric',
                                 color='Sex_Label', # Differentiated by gender for better insight
                                 title='Probabilidad de Supervivencia Prevista por Edad',
                                 labels={'Age': 'Edad', 'Survival_Probability_Numeric': 'Probabilidad de Supervivencia (%)', 'Sex_Label': 'Género'},
                                 color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_line_proba, use_container_width=True)
        
        # Gráfico de barras de la tasa de supervivencia por Clase (Pclass)
        pclass_map = {1: '1ra Clase', 2: '2da Clase', 3: '3ra Clase'}
        filtered_df['Pclass_Label'] = filtered_df['Pclass'].map(pclass_map)
        df_pclass_survival = filtered_df.groupby('Pclass_Label')['Survived_Prediction'].mean().reset_index()
        df_pclass_survival['Survival_Rate'] = (df_pclass_survival['Survived_Prediction'] * 100).round(1)

        if not df_pclass_survival.empty:
            fig_pclass = px.bar(df_pclass_survival, x='Pclass_Label', y='Survival_Rate',
                                  title='Tasa de Supervivencia Prevista por Clase',
                                  labels={'Pclass_Label': 'Clase de Pasajero', 'Survival_Rate': 'Tasa de Supervivencia (%)'},
                                  color='Pclass_Label',
                                  color_discrete_sequence=px.colors.qualitative.Pastel1)
            st.plotly_chart(fig_pclass, use_container_width=True)
        else:
            st.info("No hay datos filtrados para mostrar la tasa de supervivencia por clase.")

else:
    st.info("No se cargaron datos de inferencia. Asegúrese de que 'Inferencia.csv' esté disponible o generado.")


# Model Performance Metrics (hardcoded from training output)
st.subheader('📈Métricas de rendimiento del modelo')
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric(label="Accuracy", value="81.4%")
with metrics_col2:
    st.metric(label="Precision", value="71.0%")
with metrics_col3:
    st.metric(label="Recall", value="79.0%")
with metrics_col4:
    st.metric(label="AUC-score", value="80.8%")

st.markdown("--- Jardar: Predicciones Manuales ---")

st.subheader('🖐️ Predicción Manual')

# Manual prediction input widgets
with st.form("prediction_form"):
    col_form1, col_form2, col_form3 = st.columns(3)
    with col_form1:
        pclass = st.selectbox('Pclass', options=[1, 2, 3], index=2, help='Ticket class (1=1st, 2=2nd, 3=3rd)')
        sex_input = st.radio('Sex', options=['Male', 'Female'], index=0, help='Gender')
        sibsp = st.slider('SibSp', min_value=0, max_value=8, value=0, help='Number of siblings/spouses aboard')
    with col_form2:
        age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0, step=1.0, help='Age in years')
        parch = st.slider('Parch', min_value=0, max_value=6, value=0, help='Number of parents/children aboard')
        fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=30.0, step=0.1, help='Passenger fare')
    with col_form3:
        embarked_input = st.selectbox('Embarked', options=['S', 'C', 'Q'], index=0, help='Port of Embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)')
        family_size = st.number_input('Family Size', min_value=1, max_value=11, value=1, step=1, help='Number of family members aboard (SibSp + Parch + 1)')

    predict_button = st.form_submit_button('Predecir Supervivencia')

    if predict_button:
        # Map categorical inputs to numerical values
        sex_mapped = 0 if sex_input == 'Male' else 1
        embarked_mapped = {'S': 0, 'C': 1, 'Q': 2}[embarked_input]
        
        # Create a DataFrame for the single prediction input
        input_df = pd.DataFrame([[pclass, sex_mapped, age, sibsp, parch, fare, embarked_mapped, family_size]],
                                columns=features)
        
        # Scale the input data using the fitted scaler
        input_scaled = scaler.transform(input_df)
        
        # Make prediction and probability prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0, 1] # Probability of survival (class 1)
        
        st.write(f"## Estado de Supervivencia Previsto: {'Sobrevivió' if prediction == 1 else 'No Sobrevivió'}")
        st.write(f"## Probabilidad de Supervivencia: {prediction_proba:.2%}")
