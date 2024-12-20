# import os
# import csv
# import time
# import boto3
# import numpy as np
# import pandas as pd
# import mysql.connector
# from io import StringIO
# from datetime import datetime, timedelta
# from botocore.exceptions import ClientError
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel

# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()
# # Database connection configuration
# db_config = {
#     'host': '127.0.0.1',        # Database hostname
#     'user': os.getenv('DB_USER'),             # Database username
#     'password': os.getenv('DB_PASSWORD'),         # Database password
#     'database': os.getenv('DB_NAME')      # Name of the database
# }

# # S3 configuration
# s3_bucket = os.getenv('S3_BUCKET')           # S3 bucket name
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),           # AWS access key ID
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),   # AWS secret access key
#     region_name=os.getenv('AWS_REGION')           # AWS region
# )

# def fetch_aggregated_data_from_mysql(plant_id, connection, start_date, end_date):
#     """
#     Fetch and aggregate data for a specific plant_id for a given date range.
#     :param plant_id: Plant ID
#     :param connection: Database connection
#     :param start_date: Start date of the range
#     :param end_date: End date of the range
#     :return: Aggregated data
#     """
#     start_timestamp = int(start_date.timestamp())   # Convert start date to Unix timestamp
#     end_timestamp = int(end_date.timestamp())       # Convert end date to Unix timestamp

#     query = f"""
#         SELECT 
#             plant_id,
#             device,
#             parameter,
#             FROM_UNIXTIME(FLOOR(sensor_time / 900) * 900) AS aggregated_time,  -- 15-minute intervals
#             CASE
#                 WHEN parameter IN ('ac_yield', 'irradiance_vertical', 'irradiance_horizontal') THEN SUM(value)
#                 WHEN parameter IN ('humidity', 'wind_speed', 'ambient_temp') THEN AVG(value)
#             END AS aggregated_value
#         FROM tbl_utc_plant_aggregate_data
#         WHERE plant_id = {plant_id}
#           AND device IN ('inverter', 'weather station')
#           AND sensor_time BETWEEN {start_timestamp} AND {end_timestamp}
#           AND parameter IN ('ac_yield', 'humidity',
#                             'irradiance_vertical',
#                             'irradiance_horizontal',
#                             'wind_speed',
#                             'ambient_temp')
#         GROUP BY plant_id, device, parameter, aggregated_time
#         ORDER BY aggregated_time;
#     """

#     cursor = connection.cursor()
#     cursor.execute(query)       # Execute the query
#     data = cursor.fetchall()    # Fetch the result
#     cursor.close()

#     aggregated_data = {}
#     for row in data:
#         plant_id, device, parameter, aggregated_time, aggregated_value = row
#         if aggregated_time not in aggregated_data:
#             aggregated_data[aggregated_time] = []
#         aggregated_data[aggregated_time].append((plant_id, device, parameter, aggregated_time, aggregated_value))

#     return aggregated_data

# def save_csv_to_s3(local_rows, plant_id):
#     """
#     Save the aggregated data as a CSV in S3.
#     :param local_rows: Local rows to save
#     :param plant_id: Plant ID
#     """
#     if not local_rows:
#         return
#     # Extract unique headers
#     headers = sorted(set(param for _, _, param, _, _ in local_rows))
#     try:
#         s3_response = s3_client.get_object(Bucket=s3_bucket, Key=f"{plant_id}.csv")
#         existing_data = s3_response['Body'].read().decode('utf-8').splitlines()
#         current_rows = list(csv.reader(existing_data))
#         total_rows = len(current_rows) - 1                          # Exclude header row

#         if total_rows >= 5000:                                      # Check if file rows are >= 5000
#             rows_to_remove = total_rows + len(local_rows) - 5000
#             current_rows = current_rows[rows_to_remove + 1:]
#             current_rows.insert(0, existing_data[0])                # Keep header row
#             rows_to_save = current_rows + local_rows
#         else:
#             rows_to_save = current_rows + local_rows

#     except ClientError as e:
#         if e.response['Error']['Code'] == 'NoSuchKey':
#             print(f"File {plant_id}.csv not found in S3. Creating a new file.")
#             rows_to_save = [['timestamp'] + headers]                # Create new CSV with headers
#             timestamped_data = {}
#             for _, _, param, timestamp, value in local_rows:
#                 if timestamp not in timestamped_data:
#                     timestamped_data[timestamp] = {param: value}
#                 else:
#                     timestamped_data[timestamp][param] = value
#             for timestamp, param_map in timestamped_data.items():
#                 row = [timestamp] + [param_map.get(header, None) for header in headers]
#                 rows_to_save.append(row)
#         else:
#             print(f"An error occurred: {e}")
#             return

#     # Write the data to a CSV buffer
#     csv_buffer = StringIO()
#     csv_writer = csv.writer(csv_buffer)
#     csv_writer.writerows(rows_to_save)
#     s3_key = f"{plant_id}.csv"
#     # Upload to S3
#     s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
#     print(f"Uploaded/Updated {plant_id}.csv to S3.")

# def save_forecasted_data_to_mysql(connection, plant_id, forecasted_data):
#     """
#     Save forecasted data into the forecasted_data table in MySQL.
#     :param connection: Database connection
#     :param plant_id: Plant ID
#     :param forecasted_data: List of tuples (sensor_time, forecasted_energy)
#     """
#     try:
#         cursor = connection.cursor()
#         for sensor_time, forecasted_energy in forecasted_data:
#             query = """
#                 INSERT INTO forecasted_data (plant_id, sensor_time, forecasted_energy)
#                 VALUES (%s, %s, %s)
#                 ON DUPLICATE KEY UPDATE forecasted_energy = VALUES(forecasted_energy);
#             """
#             cursor.execute(query, (plant_id, sensor_time, forecasted_energy))
#         connection.commit()
#         print(f"Forecasted data for plant_id {plant_id} saved to MySQL.")
#     except mysql.connector.Error as e:
#         print(f"Error saving forecasted data to MySQL: {e}")
#     finally:
#         if cursor:
#             cursor.close()

# try:
#     # Establish connection to the database
#     connection = mysql.connector.connect(**db_config)
#     cursor = connection.cursor()
    
#     # Fetch plant IDs, dc_capacity, and computed_param from the query
#     query = f"""SELECT 
#                     plant_id, dc_capacity, computed_param
#                 FROM 
#                     tbl_plant
#                 WHERE 
#                     computed_param IS NOT NULL 
#                     AND computed_param LIKE '%"forecast": 1%';"""
              
#     cursor.execute(query)
#     result = cursor.fetchall()
#     plant_details = [(row[0], row[1]) for row in result]            # List of tuples (plant_id, dc_capacity)

#     print("Plant IDs with forecast enabled:", [row[0] for row in result])

#     for plant_id, dc_capacity in plant_details:
#         print(f"Processing data for plant_id {plant_id}")
#         local_rows = []

#         # Check if file exists in S3
#         file_exists = False
#         try:
#             s3_client.get_object(Bucket=s3_bucket, Key=f"{plant_id}.csv")
#             file_exists = True
#         except ClientError as e:
#             if e.response['Error']['Code'] != 'NoSuchKey':
#                 print(f"Error checking file existence in S3: {e}")

#         if file_exists:
#             start_date = datetime.now() - timedelta(days=1)           # Fetch only 1 day's data
#         else:
#             start_date = datetime.now() - timedelta(days=60)          # Fetch 60 days' data
#             print("Waiting for 5 minutes before processing next plant to manage server load...")
#             time.sleep(300)  # Wait for 5 minutes

#         end_date = datetime.now()
#         current_date = start_date
#         while current_date <= end_date:
#             next_date = current_date + timedelta(days=1)
#             # Fetch and aggregate data for the current day
#             aggregated_data = fetch_aggregated_data_from_mysql(plant_id, connection, current_date, next_date)
#             for timestamp, rows in aggregated_data.items():
#                 for row in rows:
#                     local_rows.append(row)
#             current_date = next_date
#         # Save the aggregated data to S3
#         save_csv_to_s3(local_rows, plant_id)
        
#         # Generate forecasted data
        
#         # Function to fill missing values with the next available data below the row
#         def fill_missing_weather_data(data, columns):
#             for col in columns:
#                 data[col] = data[col].ffill(limit=1)
#                 data[col] = data[col].interpolate(method='linear', limit_direction='both')
#                 data[col].fillna(0, inplace=True)
#             return data
                
#         # Load the data from the CSV file
#         s3_response = s3_client.get_object(Bucket=s3_bucket, Key=f"{plant_id}.csv")
#         existing_data = s3_response['Body'].read().decode('utf-8')
#         data = pd.read_csv(StringIO(existing_data))
        
#         # null check for s3 csv file 
#         if data.isnull().values.all():
#             raise ValueError(f"The CSV file {plant_id}.csv is completely null.")
        
#         else:
#             print(f"CSV file {plant_id}.csv loaded successfully and contains valid data.")

#             data = data.drop(columns=['timestamp'])

#             # Keep only necessary columns
#             data = data[['ac_yield', 'ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']]

#             # Drop rows with NaN in 'yield' column first
#             data = data.dropna(subset=['ac_yield']).reset_index(drop=True)

#             # Apply the fill_missing_weather_data function to the weather columns
#             weather_columns = ['ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']
#             data = fill_missing_weather_data(data, weather_columns)

#             # Normalize the features
#             scaler = MinMaxScaler()
#             data[['ac_yield', 'ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']] = scaler.fit_transform(data[['ac_yield', 'ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']])

#             # Prepare the dataset for GPR
#             window_size = 96  # 15-minute intervals for the last 24 hours
#             X, y = [], []
            
#             for i in range(window_size, len(data)):
#                 X.append(data[['ac_yield', 'ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']].iloc[i - window_size:i].values.flatten())
#                 y.append(data['ac_yield'].iloc[i])

#             X, y = np.array(X), np.array(y)

#             # Split into training and testing sets
#             split_index = int(0.8 * len(X))
#             X_train, X_test = X[:split_index], X[split_index:]
#             y_train, y_test = y[:split_index], y[split_index:]

#             fraction = 0.1
#             sample_indices = np.random.choice(np.arange(len(X_train)), size=int(fraction * len(X_train)), replace=False)
#             X_train_sampled = X_train[sample_indices]
#             y_train_sampled = y_train[sample_indices]

#             if np.any(np.isnan(y_train_sampled)):
#                 valid_indices = ~np.isnan(y_train_sampled)
#                 X_train_sampled = X_train_sampled[valid_indices]
#                 y_train_sampled = y_train_sampled[valid_indices]

#             kernel = (ExpSineSquared(length_scale=24, periodicity=24) * RBF(length_scale=5)) + WhiteKernel(noise_level=1)
#             gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
#             gpr_model.fit(X_train_sampled, y_train_sampled)

#             # Make predictions with GPR
#             gpr_predictions = gpr_model.predict(X_test)
#             gpr_predictions = gpr_predictions * dc_capacity
            
#             forecast_with_offset = np.zeros(95)
#             forecast_with_offset[25:70] = gpr_predictions[:45]

#             # Round the values
#             gpr_predictions_limited_rounded = np.round(gpr_predictions, 4)

#             forecasted_data = forecast_with_offset
#             # print(forecast_with_offset)
            
#             save_forecasted_data_to_mysql(connection, plant_id, forecasted_data)

# except mysql.connector.Error as e:
#     print(f"Error while connecting to the database: {e}")
# finally:
#     # Close the database connection
#     if 'cursor' in locals() and cursor:
#         cursor.close()
#     if 'connection' in locals() and connection and connection.is_connected():
#         connection.close()































































import os
import csv
import time
import pytz
import boto3
import numpy as np
import pandas as pd
import mysql.connector
from io import StringIO
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Database connection configuration
db_config = {
    'host': '127.0.0.1',        # Database hostname
    'user': os.getenv('DB_USER'),             # Database username
    'password': os.getenv('DB_PASSWORD'),         # Database password
    'database': os.getenv('DB_NAME')      # Name of the database
}

# S3 configuration
s3_bucket = os.getenv('S3_BUCKET')           # S3 bucket name
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),           # AWS access key ID
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),   # AWS secret access key
    region_name=os.getenv('AWS_REGION')           # AWS region
)

def fetch_aggregated_data_from_mysql(plant_id, connection, start_date, end_date):
    """
    Fetch and aggregate data for a specific plant_id for a given date range.
    :param plant_id: Plant ID
    :param connection: Database connection
    :param start_date: Start date of the range
    :param end_date: End date of the range
    :return: Aggregated data
    """
    start_timestamp = int(start_date.timestamp())   # Convert start date to Unix timestamp
    end_timestamp = int(end_date.timestamp())       # Convert end date to Unix timestamp

    query = f"""
        SELECT 
            plant_id,
            device,
            parameter,
            FROM_UNIXTIME(FLOOR(sensor_time / 900) * 900) AS aggregated_time,  -- 15-minute intervals
            CASE
                WHEN parameter IN ('ac_yield', 'irradiance_vertical', 'irradiance_horizontal') THEN SUM(value)
                WHEN parameter IN ('humidity', 'wind_speed', 'ambient_temp') THEN AVG(value)
            END AS aggregated_value
        FROM tbl_utc_plant_aggregate_data
        WHERE plant_id = {plant_id}
          AND device IN ('inverter', 'weather station')
          AND sensor_time BETWEEN {start_timestamp} AND {end_timestamp}
          AND parameter IN ('ac_yield', 'humidity',
                            'irradiance_vertical',
                            'irradiance_horizontal',
                            'wind_speed',
                            'ambient_temp')
        GROUP BY plant_id, device, parameter, aggregated_time
        ORDER BY aggregated_time;
    """

    cursor = connection.cursor()
    cursor.execute(query)       # Execute the query
    data = cursor.fetchall()    # Fetch the result
    cursor.close()

    aggregated_data = {}
    for row in data:
        plant_id, device, parameter, aggregated_time, aggregated_value = row
        if aggregated_time not in aggregated_data:
            aggregated_data[aggregated_time] = []
        aggregated_data[aggregated_time].append((plant_id, device, parameter, aggregated_time, aggregated_value))

    return aggregated_data

def save_csv_to_s3(local_rows, plant_id):
    """
    Save the aggregated data as a CSV in S3.
    :param local_rows: Local rows to save
    :param plant_id: Plant ID
    """
    if not local_rows:
        return
    # Extract unique headers
    headers = sorted(set(param for _, _, param, _, _ in local_rows))
    try:
        s3_response = s3_client.get_object(Bucket=s3_bucket, Key=f"{plant_id}.csv")
        existing_data = s3_response['Body'].read().decode('utf-8').splitlines()
        current_rows = list(csv.reader(existing_data))
        total_rows = len(current_rows) - 1                          # Exclude header row

        if total_rows >= 5000:                                      # Check if file rows are >= 5000
            rows_to_remove = total_rows + len(local_rows) - 5000
            current_rows = current_rows[rows_to_remove + 1:]
            current_rows.insert(0, existing_data[0])                # Keep header row
            rows_to_save = current_rows + local_rows
        else:
            rows_to_save = current_rows + local_rows

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File {plant_id}.csv not found in S3. Creating a new file.")
            rows_to_save = [['timestamp'] + headers]                # Create new CSV with headers
            timestamped_data = {}
            for _, _, param, timestamp, value in local_rows:
                if timestamp not in timestamped_data:
                    timestamped_data[timestamp] = {param: value}
                else:
                    timestamped_data[timestamp][param] = value
            for timestamp, param_map in timestamped_data.items():
                row = [timestamp] + [param_map.get(header, None) for header in headers]
                rows_to_save.append(row)
        else:
            print(f"An error occurred: {e}")
            return

    # Write the data to a CSV buffer
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerows(rows_to_save)
    s3_key = f"{plant_id}.csv"
    # Upload to S3
    s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"Uploaded/Updated {plant_id}.csv to S3.")

def save_forecasted_data_to_mysql(connection, plant_id, sensor_times, forecasted_data):
    """
    Save forecasted data into the forecasted_data table in MySQL.
    :param connection: Database connection
    :param plant_id: Plant ID
    :param forecasted_data: List of tuples (sensor_time, forecasted_energy)
    """
    # zipped_data=zip(sensor_times, forecasted_data)
    print(f"zip data:{sensor_times}")
    try:
        cursor = connection.cursor()
        for sensor_times, forecasted_energy in zip(sensor_times, forecasted_data):
            query = """
                INSERT INTO forecasted_data (plant_id, sensor_time, forecasted_energy)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE forecasted_energy = VALUES(forecasted_energy);
            """
            cursor.execute(query, (plant_id, sensor_times, forecasted_energy))
        connection.commit()
        print(f"Forecasted data for plant_id {plant_id} saved to MySQL.")
    except mysql.connector.Error as e:
        print(f"Error saving forecasted data to MySQL: {e}")
    finally:
        if cursor:
            cursor.close()

try:
    # Establish connection to the database
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch plant IDs, dc_capacity, and computed_param from the query
    query = f"""SELECT 
                    plant_id
                FROM 
                    tbl_plant
                WHERE 
                    forecast_enabled = 1;"""
              
    cursor.execute(query)
    result = cursor.fetchall()
    plant_details = [(row[0]) for row in result]            # List of tuples (plant_id, dc_capacity)

    print("Plant IDs with forecast enabled:", [row[0] for row in result])

    for plant_id in plant_details:
        print(f"Processing data for plant_id {plant_id}")
        local_rows = []

        # Check if file exists in S3
        file_exists = False
        try:
            s3_client.get_object(Bucket=s3_bucket, Key=f"{plant_id}.csv")
            file_exists = True
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                print(f"Error checking file existence in S3: {e}")

        if file_exists:
            start_date = datetime.now() - timedelta(days=1)           # Fetch only 1 day's data
        else:
            start_date = datetime.now() - timedelta(days=60)          # Fetch 60 days' data
            print("Waiting for 5 minutes before processing next plant to manage server load...")
            time.sleep(10)  # Wait for 5 minutes

        end_date = datetime.now()
        
        current_date = start_date
        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)
            print(f"current{current_date},end{next_date}")
            # Fetch and aggregate data for the current day
            aggregated_data = fetch_aggregated_data_from_mysql(plant_id, connection, current_date, next_date)
            for timestamp, rows in aggregated_data.items():
                for row in rows:
                    local_rows.append(row)
            current_date = next_date
        # Save the aggregated data to S3
        save_csv_to_s3(local_rows, plant_id)
        
        # Generate forecasted data
        def calculate_dynamic_start_end(data, steps_per_day):
            num_days = len(data) // steps_per_day
            start_indices = [i % steps_per_day for i in range(num_days * steps_per_day) if data['ac_yield'].iloc[i] > 0]
            end_indices = [i % steps_per_day for i in range(num_days * steps_per_day) if data['ac_yield'].iloc[i] > 0]
            most_frequent_start = pd.Series(start_indices).mode().iloc[0]
            most_frequent_end = pd.Series(end_indices).mode().iloc[-1]
            return most_frequent_start, most_frequent_end
        
        # Function to fill missing values with the next available data below the row
        def fill_missing_weather_data(data, columns):
            for col in columns:
                data[col] = data[col].ffill(limit=1)
                data[col] = data[col].interpolate(method='linear', limit_direction='both')
                data[col].fillna(0, inplace=True)
            return data
                
        # Load the data from the CSV file
        s3_response = s3_client.get_object(Bucket=s3_bucket, Key=f"{plant_id}.csv")
        existing_data = s3_response['Body'].read().decode('utf-8')
        data = pd.read_csv(StringIO(existing_data))
        
        # null check for s3 csv file 
        if data.isnull().values.all():
            raise ValueError(f"The CSV file {plant_id}.csv is completely null.")
        
        else:
            print(f"CSV file {plant_id}.csv loaded successfully and contains valid data.")

            data = data.drop(columns=['timestamp'])

            # Keep only necessary columns
            data = data[['ac_yield', 'ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']]

            # Drop rows with NaN in 'yield' column first
            data = data.dropna(subset=['ac_yield']).reset_index(drop=True)

            # Apply the fill_missing_weather_data function to the weather columns
            weather_columns = ['ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']
            data = fill_missing_weather_data(data, weather_columns)

            # Normalize the features
            scaler = MinMaxScaler()
            data[['ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']] = scaler.fit_transform(data[['ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']])

            # Prepare the dataset for GPR
            window_size = 96  # 15-minute intervals for the last 24 hours
            X, y = [], []
            
            for i in range(window_size, len(data)):
                X.append(data[['ambient_temp', 'humidity', 'irradiance_horizontal', 'irradiance_vertical', 'wind_speed']].iloc[i - window_size:i].values.flatten())
                y.append(data['ac_yield'].iloc[i])

            X, y = np.array(X), np.array(y)

            # Split into training and testing sets
            split_index = int(0.8 * len(X))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            fraction = 0.1
            sample_indices = np.random.choice(np.arange(len(X_train)), size=int(fraction * len(X_train)), replace=False)
            X_train_sampled = X_train[sample_indices]
            y_train_sampled = y_train[sample_indices]

            if np.any(np.isnan(y_train_sampled)):
                valid_indices = ~np.isnan(y_train_sampled)
                X_train_sampled = X_train_sampled[valid_indices]
                y_train_sampled = y_train_sampled[valid_indices]

            kernel = (ExpSineSquared(length_scale=24, periodicity=24) * RBF(length_scale=5)) + WhiteKernel(noise_level=1)
            gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gpr_model.fit(X_train_sampled, y_train_sampled)

            # Make predictions with GPR
            gpr_predictions = gpr_model.predict(X_test)
            gpr_predictions = gpr_predictions *70
            
            steps_per_day = window_size
            dynamic_start, dynamic_end = calculate_dynamic_start_end(data, steps_per_day)

            # Adjust forecast to start and end dynamically
            forecast_with_offset = np.zeros(steps_per_day)
            forecast_with_offset[dynamic_start:dynamic_end] = gpr_predictions[:dynamic_end - dynamic_start]

            # Drop indices 0–20 and 77–96
            forecast_with_offset = forecast_with_offset[21:76]


            forecasted_data = forecast_with_offset
            print(forecast_with_offset)
            
            # Generate the next day's date and the range (05:00 to 19:00)
            # Get the next day's date in UTC
            next_day = datetime.utcnow() + timedelta(days=1)

            # Define start time (05:00 UTC) and end time (19:00 UTC)
            gmt = pytz.timezone("GMT")
            start_time = gmt.localize(datetime(next_day.year, next_day.month, next_day.day, 5, 0, 0))
            print(f"start_time: {start_time}")

            end_time = gmt.localize(datetime(next_day.year, next_day.month, next_day.day, 19, 0, 0))
            print(f"end_time: {end_time}")

            sensor_times = []

            # Generate 15-minute intervals as epoch time
            current_time = start_time
            while current_time <= end_time:
                sensor_times.append(int(current_time.timestamp()))  # Convert to Unix timestamp
                current_time += timedelta(minutes=15)  # Add 15 minutes

            # Output the result
            print(sensor_times) 

            # Ensure the forecasted data aligns with the number of time points
            # if len(sensor_times) != len(forecasted_data):
            #     raise ValueError("Mismatch between forecasted data and sensor times.")
            
            save_forecasted_data_to_mysql(connection, plant_id,sensor_times, forecasted_data)

except mysql.connector.Error as e:
    print(f"Error while connecting to the database: {e}")
finally:
    # Close the database connection
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'connection' in locals() and connection and connection.is_connected():
        connection.close()


