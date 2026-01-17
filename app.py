import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title='Análise de Consumo de Cerveja', layout='wide')
st.title('Análise de Consumo de Cerveja - SP (2015)')

DATA_URL = 'Consumo_cerveja.csv'
DATE_COLUMN = 'Data'
HOLIDAYS_2015 = [
    '01/01/2015', '25/01/2015', '03/04/2015', '05/04/2015',
    '21/04/2015', '01/05/2015', '04/06/2015', '09/07/2015',
    '07/09/2015', '12/10/2015', '02/11/2015', '15/11/2015',
    '20/11/2015', '25/12/2015'
]


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors='coerce')
    return data


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    expected_cols = [
        'Temperatura Media (C)',
        'Temperatura Minima (C)',
        'Temperatura Maxima (C)',
        'Precipitacao (mm)',
        'Consumo de cerveja (litros)'
    ]

    for col in expected_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df = df[df[DATE_COLUMN].dt.year == 2015]

    df['Month'] = df[DATE_COLUMN].dt.month
    try:
        df['Weekday'] = df[DATE_COLUMN].dt.day_name(locale='pt_BR')
    except Exception:
        df['Weekday'] = df[DATE_COLUMN].dt.day_name()
    df['is_weekend'] = df[DATE_COLUMN].dt.dayofweek >= 5

    holidays_dt = pd.to_datetime(HOLIDAYS_2015, dayfirst=True)
    df['is_holiday'] = df[DATE_COLUMN].isin(holidays_dt)
    df['is_holiday_eve'] = df[DATE_COLUMN].isin(holidays_dt - pd.Timedelta(days=1))

    if 'Final de Semana' in df.columns:
        df['is_weekend'] = df['Final de Semana'].astype(bool)

    df = df.dropna(subset=[DATE_COLUMN, 'Consumo de cerveja (litros)'])
    df = df.drop_duplicates()

    return df


raw_df = load_data()
df = clean_data(raw_df)

st.sidebar.header('Opções de visualização')
show_raw = st.sidebar.checkbox('Mostrar dados brutos')
if show_raw:
    st.subheader('Dados brutos (limpos)')
    st.dataframe(df)

st.info(f"Registros após limpeza: {len(df)} | Datas: {df[DATE_COLUMN].min().date()} a {df[DATE_COLUMN].max().date()}")

st.subheader('Estatísticas descritivas')
numeric_cols = [
    'Consumo de cerveja (litros)',
    'Temperatura Media (C)',
    'Temperatura Minima (C)',
    'Temperatura Maxima (C)',
    'Precipitacao (mm)'
]
stats_df = df[numeric_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats_df.columns = ['Média', 'Desvio Padrão', 'Mín', 'Q1', 'Mediana', 'Q3', 'Máx']
st.dataframe(stats_df)

st.subheader('Matriz de correlação (Pearson)')
corr = df[numeric_cols].corr()
st.dataframe(corr.style.background_gradient(cmap='viridis'))

st.markdown('---')

st.subheader('Consumo mensal de cerveja')
monthly_consumption = df.groupby('Month')['Consumo de cerveja (litros)'].sum().reset_index()
month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
monthly_consumption['Mês'] = monthly_consumption['Month'].apply(lambda m: month_names[m-1])
monthly_chart = (
    alt.Chart(monthly_consumption)
    .mark_bar()
    .encode(
        x=alt.X('Mês:N', sort=month_names, title='Mês'),
        y=alt.Y('Consumo de cerveja (litros):Q', title='Total consumido (litros)'),
        color='Mês:N'
    )
)
st.altair_chart(monthly_chart, width='stretch')

st.markdown('---')

st.subheader('Relação entre Temperatura Média e Consumo de Cerveja')
temp_consumo_chart = (
    alt.Chart(df)
    .mark_circle(size=80, opacity=0.6)
    .encode(
        x=alt.X('Temperatura Media (C)', title='Temperatura Média (°C)', scale=alt.Scale(zero=False)),
        y=alt.Y('Consumo de cerveja (litros)', title='Consumo de cerveja (litros)', scale=alt.Scale(zero=False)),
        color=alt.Color('is_weekend:N', title='Fim de semana'),
        tooltip=[DATE_COLUMN, 'Temperatura Media (C)', 'Consumo de cerveja (litros)', 'is_weekend']
    )
)
reg_line = temp_consumo_chart.transform_regression(
    'Temperatura Media (C)', 'Consumo de cerveja (litros)'
).mark_line(color='red')
st.altair_chart(temp_consumo_chart + reg_line, width='stretch')

st.markdown('---')

st.subheader('Impacto da chuva no consumo em finais de semana')
weekend_df = df[df['is_weekend']].copy()
weekend_df['Choveu'] = weekend_df['Precipitacao (mm)'] > 0
rain_group = weekend_df.groupby('Choveu')['Consumo de cerveja (litros)'].mean().reset_index()
rain_group['Choveu'] = rain_group['Choveu'].map({True: 'Com chuva', False: 'Sem chuva'})
rain_chart = (
    alt.Chart(rain_group)
    .mark_bar()
    .encode(
        x=alt.X('Choveu:N', title='Condição'),
        y=alt.Y('Consumo de cerveja (litros):Q', title='Consumo médio (litros)'),
        color='Choveu:N'
    )
)
st.altair_chart(rain_chart, width='stretch')

st.markdown('---')

st.subheader('Média de Consumo de Cerveja por Tipo de Dia')
weekend_df = df[df['is_weekend']].copy()
holiday_df = df[df['is_holiday']].copy()
eve_holiday_df = df[df['is_holiday_eve']].copy()

normal_weekday_consumption = df[
    (df['is_weekend'] == False) &
    (df['is_holiday'] == False) &
    (df['is_holiday_eve'] == False)
]['Consumo de cerveja (litros)'].mean()

eve_holiday_consumption = df[
    (df['is_holiday_eve'] == True)
]['Consumo de cerveja (litros)'].mean()

holiday_consumption = df[
    (df['is_holiday'] == True)
]['Consumo de cerveja (litros)'].mean()

weekend_consumption = df[
    (df['is_weekend'] == True)
]['Consumo de cerveja (litros)'].mean()

consumption_data = {
    'Tipo de Dia': [
        'Dia de Semana (Normal)',
        'Véspera de Feriado',
        'Feriado',
        'Final de Semana'
    ],
    'Média de Consumo (litros)': [
        normal_weekday_consumption,
        eve_holiday_consumption,
        holiday_consumption,
        weekend_consumption
    ]
}

df_consumption_summary = pd.DataFrame(consumption_data)

consumption_chart_df = pd.DataFrame({
    df_consumption_summary['Tipo de Dia'].iloc[0]: [df_consumption_summary['Média de Consumo (litros)'].iloc[0], 0, 0, 0],
    df_consumption_summary['Tipo de Dia'].iloc[1]: [0, df_consumption_summary['Média de Consumo (litros)'].iloc[1], 0, 0],
    df_consumption_summary['Tipo de Dia'].iloc[2]: [0, 0, df_consumption_summary['Média de Consumo (litros)'].iloc[2], 0],
    df_consumption_summary['Tipo de Dia'].iloc[3]: [0, 0, 0, df_consumption_summary['Média de Consumo (litros)'].iloc[3]]
}, index=[df_consumption_summary['Tipo de Dia'].iloc[0], df_consumption_summary['Tipo de Dia'].iloc[1],
          df_consumption_summary['Tipo de Dia'].iloc[2], df_consumption_summary['Tipo de Dia'].iloc[3]])
st.bar_chart(consumption_chart_df)

st.markdown('---')

st.subheader('Consumo de Cerveja vs. Precipitação em Finais de Semana')
weekends_scatter_df = df[['Precipitacao (mm)', 'Consumo de cerveja (litros)']].copy()
weekends_scatter_df = weekends_scatter_df.rename(columns={
    'Precipitacao (mm)': 'Precipitação (mm)',
    'Consumo de cerveja (litros)': 'Consumo (litros)'
})
st.scatter_chart(weekends_scatter_df, x='Precipitação (mm)', y='Consumo (litros)')

st.markdown('---')

st.subheader('Distribuição das variáveis principais')
hist_cols = st.multiselect('Escolha colunas para histogramas', numeric_cols, default=['Consumo de cerveja (litros)'])
for col in hist_cols:
    hist_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=20), title=col),
            y=alt.Y('count()', title='Frequência')
        )
        .properties(height=200)
    )
    st.altair_chart(hist_chart, width='stretch')

st.markdown('---')

st.subheader('Média de Consumo de Cerveja por Tipo de Dia Aprofundado')

is_official_special_day = df['is_holiday'] | df['is_holiday_eve']
prev_day_is_official_special = is_official_special_day.shift(1).fillna(False)
next_day_is_official_special = is_official_special_day.shift(-1).fillna(False)

is_adjacent_weekend = (df['is_weekend'] == True) & \
                      (prev_day_is_official_special | next_day_is_official_special)

is_feriado_estendido_mask = is_official_special_day | is_adjacent_weekend
is_fim_de_semana_normal_mask = (df['is_weekend'] == True) & (~is_feriado_estendido_mask)

df['Tipo_de_Dia_Aprofundado'] = 'Dia de Semana Normal'

df.loc[is_fim_de_semana_normal_mask, 'Tipo_de_Dia_Aprofundado'] = 'Fim de Semana Normal'
df.loc[is_feriado_estendido_mask, 'Tipo_de_Dia_Aprofundado'] = 'Feriado Estendido'

average_consumption_by_day_type_advanced = df.groupby('Tipo_de_Dia_Aprofundado')['Consumo de cerveja (litros)'].mean().reset_index()
average_consumption_by_day_type_advanced = average_consumption_by_day_type_advanced.sort_values(by='Consumo de cerveja (litros)', ascending=False)

num_categories = len(average_consumption_by_day_type_advanced)
advanced_chart_data = {}
for i, row in enumerate(average_consumption_by_day_type_advanced.iterrows()):
    categoria = row[1]['Tipo_de_Dia_Aprofundado']
    valor = row[1]['Consumo de cerveja (litros)']
    valores = [0] * num_categories
    valores[i] = valor
    advanced_chart_data[categoria] = valores

advanced_chart_df = pd.DataFrame(advanced_chart_data,
                                 index=[row[1]['Tipo_de_Dia_Aprofundado'] for row in average_consumption_by_day_type_advanced.iterrows()])
st.bar_chart(advanced_chart_df)

st.markdown('---')
