import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------
# Cargar y agregar datos diarios
# -------------------------------
@st.cache_data
def load_and_aggregate(file):
    df = pd.read_csv(file)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    daily = df.groupby('pickup_date').agg(
        ingresos_totales=('total_amount', 'sum'),
        num_viajes=('VendorID', 'count')
    ).reset_index()
    daily['pickup_date'] = pd.to_datetime(daily['pickup_date'])
    daily = daily.sort_values('pickup_date')
    return df, daily

st.set_page_config(layout="wide")
st.title("🚕 Análisis de viajes Yellow Taxi - NYC")
st.markdown("Pronóstico simple (media móvil) y anomalías (IQR)")

uploaded_file = st.file_uploader("Sube el archivo yellow_tripdata_sample.csv", type="csv")
if uploaded_file is not None:
    df, daily = load_and_aggregate(uploaded_file)

    # -------------------------------
    # Pronóstico: media móvil de 3 días extendida 7 días
    # -------------------------------
    daily['ma3'] = daily['ingresos_totales'].rolling(3, min_periods=1).mean()
    last_ma = daily['ma3'].iloc[-1]
    last_date = daily['pickup_date'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    forecast_values = [last_ma] * 7
    forecast_df = pd.DataFrame({
        'ds': forecast_dates,
        'yhat': forecast_values,
        'yhat_lower': [last_ma * 0.9] * 7,   # banda simple ±10%
        'yhat_upper': [last_ma * 1.1] * 7
    })

    # -------------------------------
    # Detección de anomalías mediante IQR
    # -------------------------------
    sensibilidad = st.slider("Sensibilidad (factor IQR)", 1.0, 3.0, 1.5, 0.1,
                             help="Más bajo = más anomalías. 1.5 es estándar.")
    Q1 = daily['ingresos_totales'].quantile(0.25)
    Q3 = daily['ingresos_totales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - sensibilidad * IQR
    upper_bound = Q3 + sensibilidad * IQR
    daily['anomaly'] = (daily['ingresos_totales'] < lower_bound) | (daily['ingresos_totales'] > upper_bound)
    anomalies = daily[daily['anomaly']]

    # -------------------------------
    # Visualización con Altair
    # -------------------------------
    base = alt.Chart(daily).mark_line(point=True, color='steelblue').encode(
        x='pickup_date:T',
        y='ingresos_totales:Q',
        tooltip=['pickup_date:T', 'ingresos_totales:Q', 'num_viajes:Q']
    ).properties(title='Ingresos diarios totales', width=700, height=400)

    # Puntos anomalía
    anomaly_points = alt.Chart(anomalies).mark_point(color='red', size=100).encode(
        x='pickup_date:T',
        y='ingresos_totales:Q',
        tooltip=['pickup_date:T', 'ingresos_totales:Q', 'num_viajes:Q']
    )

    # Pronóstico: línea y banda
    forecast_line = alt.Chart(forecast_df).mark_line(color='orange', strokeDash=[5, 3]).encode(
        x='ds:T',
        y='yhat:Q'
    )
    band = alt.Chart(forecast_df).mark_area(opacity=0.2, color='orange').encode(
        x='ds:T',
        y='yhat_lower:Q',
        y2='yhat_upper:Q'
    )

    chart = (base + anomaly_points + band + forecast_line).interactive()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.metric("Días analizados", len(daily))
        st.metric("Anomalías detectadas", len(anomalies))
        st.caption("🔴 Puntos rojos = anomalías | 🟠 Línea discontinua = pronóstico (media móvil)")

    # -------------------------------
    # Explicación de causa (interactivo mediante selectbox)
    # -------------------------------
    if len(anomalies) > 0:
        st.subheader("🔍 Explicación de una anomalía")
        selected_date = st.selectbox("Selecciona un día anómalo:", anomalies['pickup_date'].dt.date.astype(str))
        if selected_date:
            sel_date = pd.to_datetime(selected_date).date()
            day_data = df[df['pickup_date'] == sel_date]
            if not day_data.empty:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Ingresos totales", f"${daily[daily['pickup_date'].dt.date == sel_date]['ingresos_totales'].values[0]:,.2f}")
                with col_b:
                    st.metric("Número de viajes", day_data.shape[0])

                # Desglose de pagos
                st.markdown("**Distribución por tipo de pago**")
                payment_counts = day_data['payment_type'].value_counts().reset_index()
                payment_counts.columns = ['payment_type', 'count']
                payment_map = {1: 'Tarjeta', 2: 'Efectivo', 3: 'Sin cargo', 4: 'Disputa'}
                payment_counts['tipo'] = payment_counts['payment_type'].map(payment_map)
                bar = alt.Chart(payment_counts).mark_bar().encode(
                    x='tipo:N',
                    y='count:Q',
                    color='tipo:N'
                ).properties(height=300)
                st.altair_chart(bar, use_container_width=True)

                # Montos negativos
                negativos = day_data[day_data['total_amount'] < 0]
                if len(negativos) > 0:
                    st.warning(f"⚠️ {len(negativos)} viajes con total_amount negativo. Posible error de datos.")
                    st.dataframe(negativos[['trip_distance', 'fare_amount', 'total_amount']])
                else:
                    st.success("Sin montos negativos.")
    else:
        st.info("No se detectaron anomalías con la sensibilidad actual. Reduce el factor IQR.")

    with st.expander("📋 Ver todas las anomalías"):
        st.dataframe(anomalies[['pickup_date', 'ingresos_totales', 'num_viajes']])

else:
    st.info("Sube el archivo CSV para comenzar.")
