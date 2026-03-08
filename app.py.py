# app.py

import streamlit as st
import sqlite3
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

DB_PATH = "weather.db"

st.set_page_config(
    page_title="WeatherInsight: Метеорологический анализ",
    page_icon="🌦️",
    layout="wide"
)

# Заголовок приложения
st.title("🌦️ WeatherInsight: Интерактивный анализ метеоданных")
st.markdown("---")

# Загрузка данных с кэшированием
@st.cache_data
def load_data():
    """Загрузка данных из SQLite базы данных"""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Используем pandas для более удобной работы с датами
        df = pd.read_sql_query("SELECT * FROM weather ORDER BY date", conn)
        conn.close()
        
        # Преобразование даты
        df['date'] = pd.to_datetime(df['date'])
        
        # Создание производных столбцов
        df = create_derived_columns(df)
        
        return df
    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")
        return None

def create_derived_columns(df):
    """Создание производных столбцов для анализа"""
    # Категории температуры
    df['temp_category'] = pd.cut(
        df['avg_temp'],
        bins=[-float('inf'), 0, 15, 25, float('inf')],
        labels=['❄️ Холодно', '🌱 Умеренно', '☀️ Тепло', '🔥 Жарко']
    )
    
    # Категории осадков
    df['precip_category'] = pd.cut(
        df['total_precip'],
        bins=[-0.1, 0.1, 5, 20, float('inf')],
        labels=['💧 Без осадков', '🌧️ Небольшие', '🌊 Сильные', '💨 Очень сильные']
    )
    
    # Категории ветра
    df['wind_category'] = pd.cut(
        df['avg_wind'],
        bins=[-0.1, 2, 5, 8, float('inf')],
        labels=['🍃 Штиль', '🌬️ Слабый', '💨 Умеренный', '🌪️ Сильный']
    )
    
    # Индекс комфортности (0-10)
    # Комфортность зависит от температуры, ветра и дождя
    temp_norm = (df['avg_temp'] - df['avg_temp'].min()) / (df['avg_temp'].max() - df['avg_temp'].min() + 0.01)
    wind_norm = 1 - (df['avg_wind'] - df['avg_wind'].min()) / (df['avg_wind'].max() - df['avg_wind'].min() + 0.01)
    rain_penalty = df['is_rainy'] * 0.3
    
    df['comfort_index'] = (
        temp_norm * 0.5 +  # Температура важнее всего
        wind_norm * 0.3 +  # Ветер
        (1 - rain_penalty) * 0.2  # Отсутствие дождя
    ) * 10
    
    df['comfort_category'] = pd.cut(
        df['comfort_index'],
        bins=[0, 3, 5, 7, 10],
        labels=['😫 Дискомфортно', '😐 Нормально', '🙂 Комфортно', '😍 Идеально']
    )
    
    # Месяц и год для агрегации
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['month_name'] = df['date'].dt.strftime('%B')
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    
    return df

# Загрузка данных
with st.spinner('Загрузка данных...'):
    df = load_data()

if df is None:
    st.stop()

# Боковая панель с фильтрами
with st.sidebar:
    st.header("🔍 Фильтры")
    
    # Выбор города
    cities = sorted(df['city'].unique())
    selected_cities = st.multiselect(
        "Выберите города",
        cities,
        default=[cities[0]] if cities else []
    )
    
    if not selected_cities:
        st.warning("⚠️ Выберите хотя бы один город")
        st.stop()
    
    # Фильтр по дате
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    st.subheader("📅 Период анализа")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Начало",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "Конец",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    if start_date > end_date:
        st.error("❌ Начальная дата не может быть позже конечной")
        st.stop()
    
    # Дополнительные фильтры
    st.subheader("⚙️ Дополнительно")
    show_rainy_only = st.checkbox("Только дождливые дни")
    
    # Фильтрация данных
    filtered_df = df[
        (df['city'].isin(selected_cities)) &
        (df['date'].dt.date >= start_date) &
        (df['date'].dt.date <= end_date)
    ]
    
    if show_rainy_only:
        filtered_df = filtered_df[filtered_df['is_rainy'] == 1]
    
    st.info(f"📊 Найдено записей: {len(filtered_df)}")

# Основная область
if not filtered_df.empty:
    # Вкладки для организации контента
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Данные",
        "📊 Разведочный анализ",
        "📈 Временные ряды",
        "🔮 Прогнозирование",
        "🌍 Сравнение городов"
    ])
    
    # Вкладка 1: Просмотр данных
    with tab1:
        st.header("Просмотр данных")
        
        # Настройки отображения
        col1, col2 = st.columns(2)
        with col1:
            n_rows = st.slider(
                "Количество строк",
                min_value=5,
                max_value=min(100, len(filtered_df)),
                value=min(10, len(filtered_df))
            )
        with col2:
            show_all = st.checkbox("Показать все")
        
        # Отображение таблицы
        display_df = filtered_df[[
            'date', 'city', 'avg_temp', 'total_precip', 
            'avg_wind', 'is_rainy', 'temp_category',
            'precip_category', 'comfort_category'
        ]].copy()
        
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        if show_all:
            st.dataframe(display_df, use_container_width=True, height=500)
        else:
            st.dataframe(display_df.head(n_rows), use_container_width=True)
        
        # Статистика по выбранным данным
        st.subheader("📊 Статистика по выборке")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Средняя температура", f"{filtered_df['avg_temp'].mean():.1f}°C")
        with col2:
            st.metric("Всего осадков", f"{filtered_df['total_precip'].sum():.1f} мм")
        with col3:
            st.metric("Ср. скорость ветра", f"{filtered_df['avg_wind'].mean():.1f} м/с")
        with col4:
            rainy_pct = (filtered_df['is_rainy'].sum() / len(filtered_df) * 100)
            st.metric("Дождливых дней", f"{rainy_pct:.1f}%")
    
    # Вкладка 2: Разведочный анализ
    with tab2:
        st.header("Разведочный анализ данных")
        
        # Выбор метрики для анализа
        metric = st.selectbox(
            "Выберите метрику",
            ['avg_temp', 'total_precip', 'avg_wind'],
            format_func=lambda x: {
                'avg_temp': '🌡️ Температура',
                'total_precip': '💧 Осадки',
                'avg_wind': '💨 Ветер'
            }[x]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Гистограмма
            fig_hist = px.histogram(
                filtered_df,
                x=metric,
                nbins=30,
                title=f'Распределение {metric}',
                labels={metric: metric.replace('_', ' ').title()},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot по городам
            fig_box = px.box(
                filtered_df,
                x='city',
                y=metric,
                title=f'Сравнение {metric} по городам',
                labels={metric: metric.replace('_', ' ').title(), 'city': 'Город'},
                color='city'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Матрица корреляций
        st.subheader("Корреляционная матрица")
        numeric_cols = ['avg_temp', 'total_precip', 'avg_wind', 'is_rainy']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Корреляция между метеопараметрами',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Категориальный анализ
        st.subheader("Анализ по категориям")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Распределение по категориям температуры
            temp_cat_counts = filtered_df['temp_category'].value_counts().sort_index()
            fig_temp_cat = px.pie(
                values=temp_cat_counts.values,
                names=temp_cat_counts.index,
                title='Распределение по категориям температуры'
            )
            st.plotly_chart(fig_temp_cat, use_container_width=True)
        
        with col2:
            # Распределение по категориям комфортности
            comfort_counts = filtered_df['comfort_category'].value_counts().sort_index()
            fig_comfort = px.bar(
                x=comfort_counts.index,
                y=comfort_counts.values,
                title='Распределение по уровню комфортности',
                labels={'x': 'Категория', 'y': 'Количество дней'},
                color=comfort_counts.index
            )
            st.plotly_chart(fig_comfort, use_container_width=True)
    
    # Вкладка 3: Временные ряды
    with tab3:
        st.header("Анализ временных рядов")
        
        # Выбор метрики для временного ряда
        time_metric = st.selectbox(
            "Выберите показатель",
            ['avg_temp', 'total_precip', 'avg_wind'],
            format_func=lambda x: {
                'avg_temp': '🌡️ Температура',
                'total_precip': '💧 Осадки',
                'avg_wind': '💨 Ветер'
            }[x],
            key='time_metric'
        )
        
        # Линейный график для всех выбранных городов
        fig_time = px.line(
            filtered_df,
            x='date',
            y=time_metric,
            color='city',
            title=f'Динамика {time_metric}',
            labels={time_metric: time_metric.replace('_', ' ').title(), 'date': 'Дата'},
            line_shape='linear'
        )
        
        # Добавление скользящего среднего
        if len(selected_cities) == 1:
            # Для одного города показываем скользящее среднее
            city_data = filtered_df[filtered_df['city'] == selected_cities[0]].copy()
            city_data = city_data.sort_values('date')
            city_data['rolling_mean'] = city_data[time_metric].rolling(window=7, min_periods=1).mean()
            
            fig_time.add_scatter(
                x=city_data['date'],
                y=city_data['rolling_mean'],
                mode='lines',
                name=f'Скользящее среднее (7 дней)',
                line=dict(color='red', width=2, dash='dash')
            )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Сезонная декомпозиция (простая)
        st.subheader("Сезонный анализ")
        
        # Агрегация по месяцам
        monthly_avg = filtered_df.groupby(['city', 'month'])[time_metric].mean().reset_index()
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
        
        fig_monthly = px.line(
            monthly_avg,
            x='month_name',
            y=time_metric,
            color='city',
            title=f'Среднемесячная {time_metric}',
            labels={time_metric: time_metric.replace('_', ' ').title(), 'month_name': 'Месяц'},
            markers=True
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Вкладка 4: Прогнозирование
    with tab4:
        st.header("Прогнозирование на основе скользящего среднего")
        
        if len(selected_cities) == 1:
            city_for_forecast = selected_cities[0]
            forecast_data = filtered_df[filtered_df['city'] == city_for_forecast].copy()
            forecast_data = forecast_data.sort_values('date')
            
            # Параметры прогноза
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Размер окна для скользящего среднего", 3, 30, 7)
            with col2:
                forecast_days = st.slider("Дней для прогноза", 1, 30, 7)
            
            # Расчет скользящего среднего
            forecast_data['ma'] = forecast_data['avg_temp'].rolling(window=window_size, min_periods=1).mean()
            
            # Простой прогноз: последнее значение скользящего среднего
            last_ma = forecast_data['ma'].iloc[-1]
            last_date = forecast_data['date'].iloc[-1]
            
            # Создание прогнозных дат
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            forecast_values = [last_ma] * forecast_days
            
            # Создание графика
            fig_forecast = go.Figure()
            
            # Исторические данные
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['avg_temp'],
                mode='lines',
                name='Исторические данные',
                line=dict(color='blue')
            ))
            
            # Скользящее среднее
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['ma'],
                mode='lines',
                name=f'Скользящее среднее (окно={window_size})',
                line=dict(color='green', dash='dash')
            ))
            
            # Прогноз
            fig_forecast.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Прогноз',
                line=dict(color='red', dash='dot'),
                marker=dict(size=8)
            ))
            
            fig_forecast.update_layout(
                title=f'Прогноз температуры для {city_for_forecast}',
                xaxis_title='Дата',
                yaxis_title='Температура (°C)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Метрики прогноза
            st.subheader("📊 Прогнозные значения")
            forecast_df = pd.DataFrame({
                'Дата': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                'Прогноз температуры (°C)': [f"{v:.1f}" for v in forecast_values]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("ℹ️ Для прогнозирования выберите один город")
    
    # Вкладка 5: Сравнение городов
    with tab5:
        st.header("Сравнительный анализ городов")
        
        if len(selected_cities) > 1:
            # Агрегация по городам
            city_stats = filtered_df.groupby('city').agg({
                'avg_temp': ['mean', 'min', 'max', 'std'],
                'total_precip': ['sum', 'mean'],
                'avg_wind': 'mean',
                'is_rainy': 'mean'
            }).round(2)
            
            city_stats.columns = ['Ср. темп', 'Мин темп', 'Макс темп', 'Стд темп',
                                  'Всего осадков', 'Ср. осадки', 'Ср. ветер', 'Доля дождей']
            
            city_stats['Доля дождей'] = (city_stats['Доля дождей'] * 100).round(1)
            
            st.dataframe(city_stats, use_container_width=True)
            
            # Сравнительные графики
            col1, col2 = st.columns(2)
            
            with col1:
                # Средняя температура по городам
                avg_temp_by_city = filtered_df.groupby('city')['avg_temp'].mean().reset_index()
                fig_temp_comp = px.bar(
                    avg_temp_by_city,
                    x='city',
                    y='avg_temp',
                    title='Средняя температура по городам',
                    labels={'avg_temp': 'Температура (°C)', 'city': 'Город'},
                    color='city'
                )
                st.plotly_chart(fig_temp_comp, use_container_width=True)
            
            with col2:
                # Общее количество осадков
                total_precip_by_city = filtered_df.groupby('city')['total_precip'].sum().reset_index()
                fig_precip_comp = px.bar(
                    total_precip_by_city,
                    x='city',
                    y='total_precip',
                    title='Общее количество осадков',
                    labels={'total_precip': 'Осадки (мм)', 'city': 'Город'},
                    color='city'
                )
                st.plotly_chart(fig_precip_comp, use_container_width=True)
            
            # Тепловая карта средних температур по месяцам и городам
            st.subheader("Среднемесячная температура по городам")
            
            pivot_temp = filtered_df.pivot_table(
                values='avg_temp',
                index='month_name',
                columns='city',
                aggfunc='mean',
                fill_value=0
            )
            
            # Сортировка по месяцам
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            pivot_temp = pivot_temp.reindex(month_order)
            
            fig_heatmap = px.imshow(
                pivot_temp,
                text_auto='.1f',
                aspect='auto',
                title='Среднемесячная температура (°C)',
                labels=dict(x='Город', y='Месяц', color='Температура'),
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        else:
            st.info("ℹ️ Выберите несколько городов для сравнения")

else:
    st.warning("⚠️ Нет данных для отображения с выбранными фильтрами")

# Подвал
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🌦️ WeatherInsight v2.0 | Интерактивный анализ метеорологических данных
    </div>
    """,
    unsafe_allow_html=True
)