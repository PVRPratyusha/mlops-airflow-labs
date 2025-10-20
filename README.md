# MLOps Airflow Labs - Weather Monitoring & Alert System

Apache Airflow | Python | Docker | Machine Learning | Weather API

Complete MLOps pipeline demonstrating workflow orchestration, machine learning, data visualization, and automated email alerts for weather monitoring.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Lab 1: Weather Prediction Pipeline](#lab-1-weather-prediction-pipeline)
- [Lab 2: Email Alert System](#lab-2-email-alert-system)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Results](#results)
- [Visualizations](#visualizations)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project demonstrates production-ready MLOps practices using Apache Airflow to build an automated weather monitoring system.

**What it does:**
- Monitors weather across 20 US cities
- Trains machine learning models to predict temperature
- Generates professional data visualizations
- Sends personalized email alerts for extreme weather conditions

**Key Results:**
- Model Accuracy: 94.19% R-squared
- Prediction Error: 2.65 degrees Fahrenheit RMSE
- Dataset: 3,360 hourly weather observations
- Cities: 20 major US cities

---

## Project Structure

```
mlops-airflow-labs/
├── README.md
├── .gitignore
└── airflow_lab1/
    ├── docker-compose.yaml
    ├── .env
    ├── dags/
    │   ├── airflow_weather_complete.py  # Main DAG (Lab 1 + Lab 2)
    │   ├── data/
    │   │   ├── weather_all_cities.csv
    │   │   └── weather_processed.csv
    │   ├── model/
    │   │   └── weather_model.pkl
    │   ├── visualizations/
    │   │   ├── 1_temperature_by_city.png
    │   │   ├── 2_correlation_heatmap.png
    │   │   ├── 3_your_city_temperature_trend.png
    │   │   ├── 4_temperature_distribution.png
    │   │   ├── 5_hourly_weather_patterns.png
    │   │   └── 6_city_comparison_top10.png
    │   └── alerts/
    │       ├── alert_status.json
    │       ├── email_content.html
    │       ├── email_content.txt
    │       └── FINAL_REPORT.txt
    ├── logs/
    ├── plugins/
    └── config/
```

---

## Lab 1: Weather Prediction Pipeline

### Objective

Build an automated machine learning pipeline that collects weather data, trains regression models, and generates insights through visualizations.

### Workflow Architecture

```
Step 1: Data Collection (5 minutes)
  - Fetch weather data from 20 US cities via API
  - Collect 7 days of hourly observations
  - Total: 3,360 samples

Step 2: Preprocessing & Visualization (2 minutes)
  - Feature engineering (time-based features)
  - Handle missing values
  - Generate 6 professional charts

Step 3: Model Training (3 minutes)
  - Train Linear Regression (baseline)
  - Train Random Forest (advanced)
  - Compare both models
  - Save best model

Step 4: Generate Report (1 minute)
  - Create comprehensive final report
  - Save all outputs
```

### DAG Details

**DAG Name:** `Weather_Complete_System`

**Schedule:** Every 6 hours (configurable)

**Tasks:**
1. `fetch_all_cities_data` - API data collection
2. `preprocess_and_create_visualizations` - Data processing and charts
3. `train_model_and_check_alerts` - ML training and alert detection
4. `generate_email_and_report` - Output generation

### Cities Monitored

**East Coast:** Boston, New York, Philadelphia, Miami, Atlanta

**Midwest:** Chicago, Detroit, Minneapolis, St. Louis

**South:** Houston, Dallas, Austin, New Orleans

**West Coast:** Los Angeles, San Francisco, San Diego, Portland, Seattle

**Mountain/Southwest:** Denver, Phoenix

### Dataset Features

**Total Features:** 12

**Weather Metrics:**
- Temperature (target variable)
- Humidity
- Wind speed
- Precipitation
- Atmospheric pressure
- Cloud cover

**Geographic Features:**
- Latitude
- Longitude

**Time Features:**
- Hour of day
- Day of week
- Month
- Is weekend (binary)
- Is daytime (binary)

### Model Performance

**Comparison:**

| Model | R-Squared | RMSE | MAE | Training Time |
|-------|-----------|------|-----|---------------|
| Linear Regression | 0.7143 | 5.88°F | 4.54°F | 1 second |
| Random Forest | 0.9419 | 2.65°F | 1.76°F | 10 seconds |

**Winner:** Random Forest Regressor

**Interpretation:** The Random Forest model explains 94.19% of temperature variance and predicts with an average error of only 2.65 degrees Fahrenheit.

### Why Random Forest Wins

- Captures non-linear temperature patterns
- Handles complex feature interactions
- Better for geographic data
- Robust to outliers and extreme weather

---

## Lab 2: Email Alert System

### Objective

Add intelligent email notifications to the ML pipeline, sending alerts when weather conditions exceed safety thresholds at the user's location.

### Alert System Design

**Three Severity Levels:**

**NORMAL** (Green)
- All conditions within safe thresholds
- No immediate concerns
- Email: Optional "all clear" message

**ADVISORY** (Yellow)
- Minor threshold breaches
- Conditions to monitor
- Triggers: Rain, moderate wind, high humidity
- Email: Advisory with basic recommendations

**WARNING** (Red)
- Extreme conditions detected
- Immediate action recommended
- Triggers: Very hot (over 85°F) or freezing (under 32°F)
- Email: Urgent alert with safety instructions

### Alert Thresholds

Default thresholds (customizable):

```
High Temperature: > 85°F
Low Temperature: < 32°F
Precipitation: > 0.1 inches
Wind Speed: > 20 mph
Humidity: > 90%
```

### How Alerts Work

**Detection Logic:**

```
1. Fetch current weather for user's selected city

2. Compare against thresholds:
   IF temperature >= 85°F:
      Alert Level = WARNING
      Alert Type = HIGH TEMPERATURE
      
   ELSE IF temperature <= 32°F:
      Alert Level = WARNING
      Alert Type = LOW TEMPERATURE
      
   ELSE IF precipitation >= 0.1 inches:
      Alert Level = ADVISORY
      Alert Type = PRECIPITATION
      
   ELSE IF wind_speed >= 20 mph:
      Alert Level = ADVISORY
      Alert Type = HIGH WIND
      
   ELSE IF humidity >= 90%:
      Alert Level = ADVISORY
      Alert Type = HIGH HUMIDITY
      
   ELSE:
      Alert Level = NORMAL
      No alerts

3. Generate email based on alert level

4. Include recommendations based on specific conditions
```

### Email Features

**HTML Email Includes:**
- Header with city name and alert level (color-coded)
- Current weather conditions in grid layout:
  - Temperature
  - Humidity
  - Wind speed
  - Precipitation
- List of active alerts (if any)
- Recommended safety actions
- Timestamp and location info

**Email Styling:**
- Responsive design (mobile and desktop)
- Color-coded headers (green/yellow/red)
- Professional grid layout
- Clear typography
- Action-oriented recommendations

**Email Output Location:** `dags/alerts/email_content.html`

### Test Results

**Test 1: Boston, MA**

Input Conditions:
```
Temperature: 57.7°F
Humidity: 79%
Wind Speed: 16.6 mph
Precipitation: 0.0 inches
```

System Response:
```
Alert Level: NORMAL
Active Alerts: 0
Email: "All clear" message generated
```

Result: System correctly identifies normal weather conditions

---

**Test 2: Phoenix, AZ**

Input Conditions:
```
Temperature: 87.2°F
Humidity: 21%
Wind Speed: 2.2 mph
Precipitation: 0.0 inches
```

System Response:
```
Alert Level: WARNING
Active Alerts: 1
Alert: HIGH TEMP: 87.2°F (threshold: 85°F)
Email: Warning message with heat safety tips
```

Recommendations Provided:
- Stay hydrated and limit outdoor activities
- Check on elderly neighbors and pets
- Use air conditioning or visit cooling centers

Result: System correctly triggers high temperature warning

---

### Email vs Actual Delivery

**Current Implementation:**
- Email content is GENERATED and saved as HTML file
- Can be viewed in any browser
- Demonstrates email generation logic
- No SMTP server configured (not required for lab)

**To Enable Real Email Delivery (Optional):**

Add SMTP configuration to docker-compose.yaml:
```yaml
AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
AIRFLOW__SMTP__SMTP_PORT: 587
AIRFLOW__SMTP__SMTP_USER: your_email@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD: your_app_password
```

Requires Gmail App Password (not regular password).

---

## Installation & Setup

### Prerequisites

- Docker Desktop (4GB+ RAM)
- Git
- Text editor (PyCharm recommended)

### Quick Start

```bash
# Clone repository
git clone git@github.com:YOUR-USERNAME/mlops-airflow-labs.git
cd mlops-airflow-labs/airflow_lab1

# Create folders
mkdir -p dags/{visualizations,alerts}

# Set environment
echo "AIRFLOW_UID=50000" > .env

# Initialize Airflow (first time only, 5-10 minutes)
docker-compose up airflow-init

# Start services
docker-compose up -d

# Wait 2-3 minutes, then access UI
# http://localhost:8080
# Login: airflow / airflow
```

### Verify Installation

```bash
docker-compose ps
```

All services should show "Up (healthy)".

---

## How to Run

### Configure Your Location

Before running, customize your settings:

1. Open `dags/airflow_weather_complete.py`
2. Find `USER_CONFIG` section (around line 24)
3. Update:

```python
USER_CONFIG = {
    'selected_city': 'Boston',  # Change to your city
    'email': 'your_email@example.com',  # Your email
    'alert_preferences': {
        'temp_hot_threshold': 85,  # Customize if needed
        'temp_cold_threshold': 32,
        'precipitation_threshold': 0.1,
        'wind_threshold': 20,
        'humidity_threshold': 90,
    }
}
```

4. Save file
5. Restart: `docker-compose restart airflow-scheduler`

### Run the Pipeline

1. Open browser: http://localhost:8080
2. Login: airflow / airflow
3. Find DAG: `Weather_Complete_System`
4. Toggle ON (switch turns blue)
5. Click play button, select "Trigger DAG"
6. Go to Graph view and watch tasks execute
7. Wait 10-15 minutes for completion

### View Results

**Visualizations:**
```bash
open dags/visualizations/*.png
```

**Email Alert:**
```bash
open dags/alerts/email_content.html
```

**Final Report:**
```bash
cat dags/alerts/FINAL_REPORT.txt
```

---

## Results

### Model Performance

**Random Forest Regressor:**
- R-squared: 0.9419 (94.19% accuracy)
- RMSE: 2.65°F (average prediction error)
- MAE: 1.76°F
- Dataset: 3,360 samples from 20 cities

The model explains 94.19% of temperature variance across all cities.

### Alert System Testing

**Boston Test:**
- Temperature: 57.7°F
- Alert Level: NORMAL
- Alerts: None
- Status: Correct (within thresholds)

**Phoenix Test:**
- Temperature: 87.2°F
- Alert Level: WARNING
- Alerts: High temperature (exceeds 85°F)
- Status: Correct (alert triggered)

---

## Visualizations

Six professional charts generated automatically:

**1. Temperature by City**
- Bar chart ranking all 20 cities
- Selected city highlighted in red
- Shows geographic temperature variation

**2. Feature Correlation Heatmap**
- Shows relationships between weather variables
- Identifies important features for prediction

**3. Your City Temperature Trend**
- 7-day hourly temperature line chart
- Alert threshold lines shown
- Identifies when thresholds are crossed

**4. Temperature Distribution**
- Histogram and box plot
- Shows overall temperature patterns
- Mean and median indicated

**5. Hourly Weather Patterns**
- Temperature, humidity, and wind by hour
- Shows daily weather cycles
- Peak temperature around 2-3 PM

**6. City Comparison (Top 10)**
- Top 10 warmest cities
- Error bars show min/max range
- Selected city highlighted

All visualizations saved as PNG files (150 DPI) in `dags/visualizations/`.

---

## Configuration

### Change Monitored City

Edit `dags/airflow_weather_complete.py`:

```python
'selected_city': 'Phoenix',  # Choose any of 20 cities
```

### Customize Alert Thresholds

```python
'temp_hot_threshold': 85,      # Degrees Fahrenheit
'temp_cold_threshold': 32,     # Degrees Fahrenheit
'precipitation_threshold': 0.1, # Inches
'wind_threshold': 20,          # mph
'humidity_threshold': 90,      # Percentage
```

### Adjust Schedule

```python
schedule_interval='0 */6 * * *'  # Every 6 hours

# Other options:
# '@hourly'     - Every hour
# '@daily'      - Once per day
# '0 8 * * *'   - Daily at 8 AM
```

---

## Troubleshooting

### DAG Not Appearing

```bash
# Check for Python errors
docker-compose logs airflow-scheduler | tail -50

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Tasks Failing

```bash
# View specific task logs in Airflow UI
# Click task → Log button

# Check if packages installed
docker-compose exec airflow-scheduler pip list | grep scikit
```

### Cannot Access UI

```bash
# Check services running
docker-compose ps

# Restart webserver
docker-compose restart airflow-webserver
```

### Visualizations Not Created

```bash
# Create folder manually
mkdir -p dags/visualizations

# Restart DAG
```

---

## Technologies Used

**Workflow Orchestration:**
- Apache Airflow 2.7.1
- Docker & Docker Compose
- PostgreSQL 13

**Machine Learning:**
- scikit-learn (Random Forest, Linear Regression)
- pandas (data manipulation)
- NumPy (numerical computing)

**Visualization:**
- matplotlib (charts)
- seaborn (statistical plots)

**Data Source:**
- Open-Meteo API (free weather data)

---


## Quick Commands

```bash
# Start Airflow
cd airflow_lab1
docker-compose up -d

# Stop Airflow
docker-compose down

# View logs
docker-compose logs -f airflow-scheduler

# Check status
docker-compose ps

# Restart scheduler
docker-compose restart airflow-scheduler
```

---

## Author

**Name:** Pratyusha Pillalamarri  
**Email:** pillalamarri.v@northeastern.edu  
**Course:** MLOps - Northeastern University  
**Date:** October 2025

---

## Acknowledgments

Course Instructor: Ramin Mohammadi  
Data Source: Open-Meteo API  
Framework: Apache Airflow

---

Last Updated: October 20, 2025