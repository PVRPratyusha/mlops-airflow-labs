"""
Complete Weather System: 20 Cities + Visualizations + Email Alerts
Combines Everything: Lab 1 + Lab 2 + User City Selection

Features:
- Collects data from 20 US cities
- Creates 6 professional visualizations
- User selects their city from dropdown
- Personalized email alerts for selected city
- Smart alert thresholds
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import requests
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# USER CONFIGURATION - CHANGE THIS TO YOUR CITY AND EMAIL!
# ============================================================================

USER_CONFIG = {
    'selected_city': 'Phoenix',  # Change to any city from CITIES dict below
    'email': 'pillalamarri.v@northeastern.edu',  # Change to your email
    'alert_preferences': {
        'temp_hot_threshold': 85,  # °F - alert when temp above this
        'temp_cold_threshold': 32,  # °F - alert when temp below this
        'precipitation_threshold': 0.1,  # inches
        'wind_threshold': 20,  # mph
        'humidity_threshold': 90,  # %
    }
}

# All available cities - user picks one above
CITIES = {
    # East Coast
    'Boston': (42.36, -71.06),
    'New York': (40.71, -74.01),
    'Philadelphia': (39.95, -75.17),
    'Miami': (25.76, -80.19),
    'Atlanta': (33.75, -84.39),

    # Midwest
    'Chicago': (41.88, -87.63),
    'Detroit': (42.33, -83.05),
    'Minneapolis': (44.98, -93.27),
    'St. Louis': (38.63, -90.20),

    # South
    'Houston': (29.76, -95.37),
    'Dallas': (32.78, -96.80),
    'Austin': (30.27, -97.74),
    'New Orleans': (29.95, -90.07),

    # West Coast
    'Los Angeles': (34.05, -118.24),
    'San Francisco': (37.77, -122.42),
    'San Diego': (32.72, -117.16),
    'Portland': (45.52, -122.68),
    'Seattle': (47.61, -122.33),

    # Mountain/Southwest
    'Denver': (39.74, -104.99),
    'Phoenix': (33.45, -112.07),
}

# ============================================================================

# Define paths
DATA_DIR = Path('/opt/airflow/dags/data')
MODEL_DIR = Path('/opt/airflow/dags/model')
VIZ_DIR = Path('/opt/airflow/dags/visualizations')
ALERT_DIR = Path('/opt/airflow/dags/alerts')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': True,
    'email': [USER_CONFIG['email']],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def fetch_all_cities_data(**context):
    """
    Task 1: Fetch weather data from all 20 cities
    """
    print("=" * 70)
    print("TASK 1: Fetching Weather Data from 20 US Cities")
    print("=" * 70)

    all_data = []
    successful = 0

    for city_name, (lat, lon) in CITIES.items():
        print(f"  Fetching {city_name}...", end=" ")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation',
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,pressure_msl,cloud_cover',
            'temperature_unit': 'fahrenheit',
            'past_days': 7,
            'forecast_days': 0,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()

            # Store current conditions for user's city
            if city_name == USER_CONFIG['selected_city']:
                current = weather_data['current']
                context['task_instance'].xcom_push(key='user_city_current', value={
                    'city': city_name,
                    'temperature': current['temperature_2m'],
                    'humidity': current['relative_humidity_2m'],
                    'wind_speed': current['wind_speed_10m'],
                    'precipitation': current.get('precipitation', 0),
                })

            # Store historical data
            hourly = weather_data['hourly']
            for i in range(len(hourly['time'])):
                all_data.append({
                    'city': city_name,
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': hourly['time'][i],
                    'temperature': hourly['temperature_2m'][i],
                    'humidity': hourly['relative_humidity_2m'][i],
                    'wind_speed': hourly['wind_speed_10m'][i],
                    'precipitation': hourly['precipitation'][i],
                    'pressure': hourly['pressure_msl'][i],
                    'cloud_cover': hourly['cloud_cover'][i],
                })

            successful += 1
            print(f"✓ ({len(hourly['time'])} records)")

        except Exception as e:
            print(f"✗ Failed: {e}")
            continue

    df = pd.DataFrame(all_data)

    print(f"\n{'=' * 70}")
    print("DATA COLLECTION SUMMARY")
    print("=" * 70)
    print(f"  Total samples: {len(df):,}")
    print(f"  Cities collected: {successful}/{len(CITIES)}")
    print(f"  Your city: {USER_CONFIG['selected_city']}")
    print(f"  Temperature range: {df['temperature'].min():.1f}°F - {df['temperature'].max():.1f}°F")

    # Save data
    DATA_DIR.mkdir(exist_ok=True)
    raw_path = DATA_DIR / 'weather_all_cities.csv'
    df.to_csv(raw_path, index=False)

    print(f"\n✓ Saved to: {raw_path}")
    return str(raw_path)


def preprocess_and_create_visualizations(**context):
    """
    Task 2: Preprocess data and create 6 visualizations
    """
    print("=" * 70)
    print("TASK 2: Preprocessing & Creating Visualizations")
    print("=" * 70)

    raw_path = DATA_DIR / 'weather_all_cities.csv'
    df = pd.read_csv(raw_path)

    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)

    df = df.dropna(subset=['temperature'])
    numeric_cols = ['humidity', 'wind_speed', 'precipitation', 'pressure', 'cloud_cover']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    print(f"Processed shape: {df.shape}")

    # Create visualizations
    VIZ_DIR.mkdir(exist_ok=True)

    # Viz 1: Temperature by City with highlight
    print("\nCreating Viz 1: Temperature by City...")
    plt.figure(figsize=(14, 8))
    city_temp = df.groupby('city')['temperature'].mean().sort_values(ascending=False)
    colors = ['#ff6b6b' if city == USER_CONFIG['selected_city'] else '#4ecdc4'
              for city in city_temp.index]
    plt.barh(city_temp.index, city_temp.values, color=colors)
    plt.xlabel('Average Temperature (°F)', fontsize=12, fontweight='bold')
    plt.ylabel('City', fontsize=12, fontweight='bold')
    plt.title('Average Temperature Across 20 US Cities\n(Your City Highlighted in Red)',
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    for i, v in enumerate(city_temp.values):
        plt.text(v + 0.5, i, f'{v:.1f}°F', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '1_temperature_by_city.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Viz 2: Correlation Heatmap
    print("Creating Viz 2: Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr_features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure', 'cloud_cover']
    corr_matrix = df[corr_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Weather Features Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Viz 3: Your City Temperature Trend
    print(f"Creating Viz 3: {USER_CONFIG['selected_city']} Temperature Trend...")
    user_city_data = df[df['city'] == USER_CONFIG['selected_city']].copy()
    user_city_data = user_city_data.sort_values('timestamp')

    plt.figure(figsize=(14, 6))
    plt.plot(user_city_data['timestamp'], user_city_data['temperature'],
             color='steelblue', linewidth=2, label='Temperature')
    plt.axhline(y=USER_CONFIG['alert_preferences']['temp_hot_threshold'],
                color='red', linestyle='--', linewidth=2, label='Hot Alert Threshold')
    plt.axhline(y=USER_CONFIG['alert_preferences']['temp_cold_threshold'],
                color='blue', linestyle='--', linewidth=2, label='Cold Alert Threshold')
    plt.fill_between(user_city_data['timestamp'], user_city_data['temperature'],
                     alpha=0.3, color='steelblue')
    plt.xlabel('Date & Time', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (°F)', fontsize=12, fontweight='bold')
    plt.title(f'Temperature Trend: {USER_CONFIG["selected_city"]} (Past 7 Days)',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '3_your_city_temperature_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Viz 4: Temperature Distribution
    print("Creating Viz 4: Temperature Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['temperature'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['temperature'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {df["temperature"].mean():.1f}°F')
    axes[0].set_xlabel('Temperature (°F)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Temperature Distribution (All Cities)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(df['temperature'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[1].set_ylabel('Temperature (°F)', fontsize=12)
    axes[1].set_title('Temperature Box Plot', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '4_temperature_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Viz 5: Weather Conditions by Hour
    print("Creating Viz 5: Weather by Hour...")
    hourly_stats = df.groupby('hour').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(hourly_stats['hour'], hourly_stats['temperature'],
                 marker='o', color='orangered', linewidth=2, markersize=6)
    axes[0].set_ylabel('Temperature (°F)', fontsize=11, fontweight='bold')
    axes[0].set_title('Average Hourly Weather Patterns', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].plot(hourly_stats['hour'], hourly_stats['humidity'],
                 marker='s', color='steelblue', linewidth=2, markersize=6)
    axes[1].set_ylabel('Humidity (%)', fontsize=11, fontweight='bold')
    axes[1].grid(alpha=0.3)

    axes[2].plot(hourly_stats['hour'], hourly_stats['wind_speed'],
                 marker='^', color='green', linewidth=2, markersize=6)
    axes[2].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Wind Speed (mph)', fontsize=11, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '5_hourly_weather_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Viz 6: City Comparison (Top 10 warmest)
    print("Creating Viz 6: City Comparison...")
    city_stats = df.groupby('city').agg({
        'temperature': ['mean', 'min', 'max'],
        'humidity': 'mean'
    }).reset_index()
    city_stats.columns = ['city', 'temp_mean', 'temp_min', 'temp_max', 'humidity']
    city_stats = city_stats.sort_values('temp_mean', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(city_stats))
    width = 0.6

    bars = ax.bar(x, city_stats['temp_mean'], width,
                  color=['#ff6b6b' if city == USER_CONFIG['selected_city'] else '#4ecdc4'
                         for city in city_stats['city']], alpha=0.8)
    ax.errorbar(x, city_stats['temp_mean'],
                yerr=[city_stats['temp_mean'] - city_stats['temp_min'],
                      city_stats['temp_max'] - city_stats['temp_mean']],
                fmt='none', color='black', capsize=5, linewidth=2)

    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°F)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Warmest Cities (with Min/Max Range)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(city_stats['city'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '6_city_comparison_top10.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save processed data
    processed_path = DATA_DIR / 'weather_processed.csv'
    df.to_csv(processed_path, index=False)

    print(f"\n✓ Created 6 visualizations in: {VIZ_DIR}")
    print(f"✓ Saved processed data: {processed_path}")

    return str(processed_path)


def train_model_and_check_alerts(**context):
    """
    Task 3: Train model and check alert conditions for user's city
    """
    print("=" * 70)
    print(f"TASK 3: Training Model & Checking Alerts for {USER_CONFIG['selected_city']}")
    print("=" * 70)

    # Load data
    processed_path = DATA_DIR / 'weather_processed.csv'
    df = pd.read_csv(processed_path)

    # Train model
    feature_cols = ['latitude', 'longitude', 'humidity', 'wind_speed',
                    'precipitation', 'pressure', 'cloud_cover',
                    'hour', 'day_of_week', 'month', 'is_weekend', 'is_daytime']

    X = df[feature_cols]
    y = df['temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print(f"\n✓ Model Trained: R²={r2:.4f}, RMSE={rmse:.2f}°F")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / 'weather_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Check alerts for user's city
    user_current = context['task_instance'].xcom_pull(key='user_city_current', task_ids='fetch_all_cities_data')

    city = user_current['city']
    temp = user_current['temperature']
    humidity = user_current['humidity']
    wind = user_current['wind_speed']
    precip = user_current['precipitation']

    prefs = USER_CONFIG['alert_preferences']

    print(f"\n{'=' * 70}")
    print(f"CURRENT CONDITIONS: {city}")
    print("=" * 70)
    print(f"  🌡️  Temperature: {temp:.1f}°F")
    print(f"  💧 Humidity: {humidity}%")
    print(f"  💨 Wind: {wind:.1f} mph")
    print(f"  🌧️  Precipitation: {precip} in")

    # Check alerts
    alerts = []
    alert_level = "NORMAL"

    if temp >= prefs['temp_hot_threshold']:
        alerts.append(f"🔥 HIGH TEMP: {temp:.1f}°F (threshold: {prefs['temp_hot_threshold']}°F)")
        alert_level = "WARNING"
    elif temp <= prefs['temp_cold_threshold']:
        alerts.append(f"🥶 LOW TEMP: {temp:.1f}°F (threshold: {prefs['temp_cold_threshold']}°F)")
        alert_level = "WARNING"

    if precip >= prefs['precipitation_threshold']:
        alerts.append(f"🌧️ RAIN: {precip:.2f} in (threshold: {prefs['precipitation_threshold']} in)")
        if alert_level == "NORMAL":
            alert_level = "ADVISORY"

    if wind >= prefs['wind_threshold']:
        alerts.append(f"💨 WIND: {wind:.1f} mph (threshold: {prefs['wind_threshold']} mph)")
        if alert_level == "NORMAL":
            alert_level = "ADVISORY"

    if humidity >= prefs['humidity_threshold']:
        alerts.append(f"💧 HUMIDITY: {humidity}% (threshold: {prefs['humidity_threshold']}%)")
        if alert_level == "NORMAL":
            alert_level = "ADVISORY"

    print(f"\n{'=' * 70}")
    print(f"ALERT LEVEL: {alert_level}")
    print("=" * 70)

    if alerts:
        print("\n⚠️  Active Alerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\n✅ No alerts - conditions are normal")

    # Save alert info
    ALERT_DIR.mkdir(exist_ok=True)
    alert_info = {
        'alert_level': alert_level,
        'alerts': alerts,
        'should_send_email': len(alerts) > 0,
        'city': city,
        'conditions': user_current,
        'timestamp': datetime.now().isoformat()
    }

    with open(ALERT_DIR / 'alert_status.json', 'w') as f:
        json.dump(alert_info, f, indent=2)

    context['task_instance'].xcom_push(key='alert_level', value=alert_level)
    context['task_instance'].xcom_push(key='should_send_email', value=len(alerts) > 0)

    return alert_level


def generate_email_and_report(**context):
    """
        Task 4: Generate beautiful HTML email with proper formatting
        """
    print("=" * 70)
    print("TASK 4: Generating Email & Final Report")
    print("=" * 70)

    import json
    from datetime import datetime
    from pathlib import Path

    ALERT_DIR = Path('/opt/airflow/dags/alerts')

    # Load alert info
    with open(ALERT_DIR / 'alert_status.json', 'r') as f:
        alert_info = json.load(f)

    alert_level = alert_info['alert_level']
    alerts = alert_info['alerts']
    current = alert_info['conditions']
    city = current['city']

    # Color schemes for different alert levels
    colors = {
        'NORMAL': {'bg': '#28a745', 'text': '#155724', 'border': '#c3e6cb'},
        'ADVISORY': {'bg': '#ffc107', 'text': '#856404', 'border': '#ffeaa7'},
        'WARNING': {'bg': '#dc3545', 'text': '#721c24', 'border': '#f5c6cb'}
    }

    theme = colors[alert_level]

    # Generate improved HTML email
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weather Alert - {city}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background-color: #f5f5f5;
                color: #333;
                line-height: 1.6;
            }}

            .container {{
                max-width: 600px;
                margin: 20px auto;
                background-color: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}

            .header {{
                background: linear-gradient(135deg, {theme['bg']} 0%, {theme['bg']}dd 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }}

            .header h1 {{
                font-size: 28px;
                margin-bottom: 10px;
                font-weight: 600;
            }}

            .header .level {{
                font-size: 20px;
                font-weight: 500;
                opacity: 0.95;
            }}

            .content {{
                padding: 30px;
            }}

            .section-title {{
                font-size: 20px;
                font-weight: 600;
                color: #333;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid {theme['bg']};
            }}

            .conditions-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 20px 0;
            }}

            .condition-card {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid {theme['bg']};
            }}

            .condition-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}

            .condition-value {{
                font-size: 24px;
                font-weight: 700;
                color: #333;
            }}

            .alert-box {{
                background-color: {theme['border']};
                border-left: 5px solid {theme['bg']};
                padding: 20px;
                margin: 15px 0;
                border-radius: 5px;
            }}

            .alert-box strong {{
                color: {theme['text']};
                font-size: 16px;
                display: block;
                margin-bottom: 5px;
            }}

            .no-alerts {{
                background-color: #d4edda;
                color: #155724;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                font-size: 16px;
                margin: 20px 0;
            }}

            .recommendations {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}

            .recommendations h3 {{
                color: #856404;
                margin-bottom: 10px;
                font-size: 16px;
            }}

            .recommendations ul {{
                margin-left: 20px;
                color: #856404;
            }}

            .recommendations li {{
                margin: 8px 0;
            }}

            .footer {{
                background-color: #f8f9fa;
                padding: 25px;
                text-align: center;
                color: #666;
                font-size: 13px;
                border-top: 1px solid #e0e0e0;
            }}

            .footer p {{
                margin: 5px 0;
            }}

            .badge {{
                display: inline-block;
                padding: 5px 12px;
                background-color: {theme['bg']};
                color: white;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>Weather Alert: {city}</h1>
                <div class="level">{alert_level} Level</div>
                <div class="badge">Automated Weather Monitoring</div>
            </div>

            <!-- Content -->
            <div class="content">
                <h2 class="section-title">Current Conditions</h2>

                <div class="conditions-grid">
                    <div class="condition-card">
                        <div class="condition-label">Temperature</div>
                        <div class="condition-value">{current['temperature']:.1f}°F</div>
                    </div>

                    <div class="condition-card">
                        <div class="condition-label">Humidity</div>
                        <div class="condition-value">{current['humidity']}%</div>
                    </div>

                    <div class="condition-card">
                        <div class="condition-label">Wind Speed</div>
                        <div class="condition-value">{current['wind_speed']:.1f} mph</div>
                    </div>

                    <div class="condition-card">
                        <div class="condition-label">Precipitation</div>
                        <div class="condition-value">{current['precipitation']:.2f} in</div>
                    </div>
                </div>
        """

    # Active Alerts Section
    if alerts:
        html_content += f"""
                <h2 class="section-title">Active Alerts</h2>
            """

        for alert in alerts:
            # Remove emoji codes from alert text
            alert_clean = alert.replace('🔥 ', '').replace('🥶 ', '').replace('🌧️ ', '').replace('💨 ', '').replace('💧 ',
                                                                                                                 '')
            html_content += f'<div class="alert-box"><strong>{alert_clean}</strong></div>\n'

        # Add recommendations based on alert type
        html_content += """
                <div class="recommendations">
                    <h3>Recommended Actions:</h3>
                    <ul>
            """

        if current['temperature'] >= 85:
            html_content += """
                        <li><strong>Stay Hydrated:</strong> Drink plenty of water throughout the day</li>
                        <li><strong>Limit Outdoor Activity:</strong> Avoid strenuous exercise during peak heat</li>
                        <li><strong>Check on Others:</strong> Elderly neighbors and pets are at higher risk</li>
                        <li><strong>Stay Cool:</strong> Use air conditioning or visit cooling centers</li>
                """
        elif current['temperature'] <= 32:
            html_content += """
                        <li><strong>Dress Warmly:</strong> Layer clothing and cover exposed skin</li>
                        <li><strong>Protect Pipes:</strong> Let faucets drip to prevent freezing</li>
                        <li><strong>Warm Your Car:</strong> Allow extra time for vehicle warm-up</li>
                        <li><strong>Watch for Ice:</strong> Be cautious of icy sidewalks and roads</li>
                """

        if current['precipitation'] >= 0.1:
            html_content += """
                        <li><strong>Bring Umbrella:</strong> Rain expected in your area</li>
                        <li><strong>Drive Safely:</strong> Wet roads require extra caution</li>
                """

        if current['wind_speed'] >= 20:
            html_content += """
                        <li><strong>Secure Items:</strong> Bring in outdoor furniture and decorations</li>
                        <li><strong>Tree Caution:</strong> Watch for falling branches</li>
                """

        html_content += """
                    </ul>
                </div>
            """
    else:
        html_content += """
                <div class="no-alerts">
                    <h2 style="margin-bottom: 10px;">No Active Weather Alerts</h2>
                    <p>Current weather conditions are within normal ranges. Enjoy your day!</p>
                </div>
            """

    # Footer
    html_content += f"""
            </div>

            <!-- Footer -->
            <div class="footer">
                <p><strong>Airflow Weather Prediction System</strong></p>
                <p>Report Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}</p>
                <p>Location: {city} | Temperature: {current['temperature']:.1f}°F</p>
                <p style="margin-top: 15px; font-size: 11px;">
                    This is an automated alert from your weather monitoring system.<br>
                    To view detailed visualizations, check: dags/visualizations/
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save email with UTF-8 encoding
    with open(ALERT_DIR / 'email_content.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Email content generated: {ALERT_DIR / 'email_content.html'}")

    # Generate plain text version
    text_content = f"""
    {'=' * 70}
    WEATHER ALERT: {city}
    {'=' * 70}

    Alert Level: {alert_level}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    CURRENT CONDITIONS:
    ------------------
    Temperature:   {current['temperature']:.1f}°F
    Humidity:      {current['humidity']}%
    Wind Speed:    {current['wind_speed']:.1f} mph
    Precipitation: {current['precipitation']:.2f} inches

    """

    if alerts:
        text_content += f"""
    ACTIVE ALERTS ({len(alerts)}):
    {'-' * 70}
    """
        for i, alert in enumerate(alerts, 1):
            alert_clean = alert.replace('🔥 ', '').replace('🥶 ', '').replace('🌧️ ', '').replace('💨 ', '').replace('💧 ',
                                                                                                                 '')
            text_content += f"{i}. {alert_clean}\n"

        text_content += f"""
    RECOMMENDED ACTIONS:
    {'-' * 70}
    """
        if current['temperature'] >= 85:
            text_content += "- Stay hydrated and limit outdoor activities\n"
            text_content += "- Check on elderly neighbors and pets\n"
        elif current['temperature'] <= 32:
            text_content += "- Dress warmly in layers\n"
            text_content += "- Protect exposed pipes from freezing\n"
    else:
        text_content += """
    STATUS: All Clear
    No weather alerts at this time. Conditions are normal.
    """

    text_content += f"""

    {'=' * 70}
    Weather Prediction System | {city}
    {'=' * 70}
    """

    with open(ALERT_DIR / 'email_content.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)

    # Generate final report (same as before, just cleaner)
    report = f"""
    {'=' * 70}
    COMPLETE WEATHER SYSTEM - FINAL REPORT
    {'=' * 70}

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    {'=' * 70}
    YOUR CONFIGURATION
    {'=' * 70}
    Selected City: {city}
    Email: {USER_CONFIG['email']}

    Alert Thresholds:
      - Hot Temperature: >{USER_CONFIG['alert_preferences']['temp_hot_threshold']}°F
      - Cold Temperature: <{USER_CONFIG['alert_preferences']['temp_cold_threshold']}°F
      - Precipitation: >{USER_CONFIG['alert_preferences']['precipitation_threshold']} in
      - Wind: >{USER_CONFIG['alert_preferences']['wind_threshold']} mph
      - Humidity: >{USER_CONFIG['alert_preferences']['humidity_threshold']}%

    {'=' * 70}
    CURRENT WEATHER: {city}
    {'=' * 70}
    Temperature: {current['temperature']:.1f}°F
    Humidity: {current['humidity']}%
    Wind Speed: {current['wind_speed']:.1f} mph
    Precipitation: {current['precipitation']:.2f} inches

    {'=' * 70}
    ALERT STATUS
    {'=' * 70}
    Alert Level: {alert_level}
    Active Alerts: {len(alerts)}

    """

    if alerts:
        report += "\nAlert Details:\n"
        for i, alert in enumerate(alerts, 1):
            report += f"{i}. {alert}\n"
    else:
        report += "\nNo weather alerts. Conditions are normal.\n"

    report += f"""

    {'=' * 70}
    FILES GENERATED
    {'=' * 70}
    Visualizations (6 total):
      1. Temperature by City (your city highlighted)
      2. Feature Correlation Heatmap
      3. Your City Temperature Trend (7 days)
      4. Temperature Distribution (all cities)
      5. Hourly Weather Patterns
      6. Top 10 Warmest Cities Comparison

    Data Files:
      - weather_all_cities.csv (20 cities raw data)
      - weather_processed.csv (cleaned & engineered features)

    Model Files:
      - weather_model.pkl (Random Forest model)

    Alert Files:
      - alert_status.json (alert decisions)
      - email_content.html (formatted email)
      - email_content.txt (plain text version)

    {'=' * 70}
    AVAILABLE CITIES
    {'=' * 70}
    You can change your city in the code to any of these:
    Atlanta, Austin, Boston, Chicago, Dallas, Denver, Detroit, Houston,
    Los Angeles, Miami, Minneapolis, New Orleans, New York, Philadelphia,
    Phoenix, Portland, San Diego, San Francisco, Seattle, St. Louis

    Simply update USER_CONFIG['selected_city'] at the top of the file.

    {'=' * 70}
    SYSTEM STATUS
    {'=' * 70}
    Status: OPERATIONAL
    Last Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Alert System: ACTIVE
    Email Generation: SUCCESS

    {'=' * 70}
    """

    with open(ALERT_DIR / 'FINAL_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\nEmail content saved to: {ALERT_DIR / 'email_content.html'}")
    print(f"Final report saved to: {ALERT_DIR / 'FINAL_REPORT.txt'}")

    return str(ALERT_DIR / 'FINAL_REPORT.txt')

# Define the DAG
with DAG(
        'Weather_Complete_System',
        default_args=default_args,
        description='Complete: 20 cities + 6 visualizations + email alerts',
        schedule_interval='0 */6 * * *',  # Every 6 hours
        catchup=False,
        tags=['weather', 'complete', 'lab1', 'lab2', 'production'],
) as dag:
    task_fetch = PythonOperator(
        task_id='fetch_all_cities_data',
        python_callable=fetch_all_cities_data,
    )

    task_preprocess = PythonOperator(
        task_id='preprocess_and_create_visualizations',
        python_callable=preprocess_and_create_visualizations,
    )

    task_train_alert = PythonOperator(
        task_id='train_model_and_check_alerts',
        python_callable=train_model_and_check_alerts,
    )

    task_email_report = PythonOperator(
        task_id='generate_email_and_report',
        python_callable=generate_email_and_report,
    )

    task_fetch >> task_preprocess >> task_train_alert >> task_email_report