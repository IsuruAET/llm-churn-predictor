import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import openai
from collections import defaultdict
import os
from dotenv import load_dotenv
import re
from typing import Optional
import csv
from datetime import datetime, date, timedelta
import pandas as pd

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChurnRequest(BaseModel):
    churn_count: int
    non_churn_count: int
    given_date: str = date.today().strftime('%Y-%m-%d')  # Default to today
    num_weeks: int = 20  # Default to 20 weeks
    model: str = "gpt-3.5-turbo"
    custom_prompt: Optional[str] = None
    shuffled_data: Optional[list] = None  # Add support for shuffled data

# Model pricing (per 1K tokens) - Updated as of 2025
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5": {"input": 0.005, "output": 0.015}
}

def log_prediction_to_csv(data):
    """Log prediction results to CSV file"""
    csv_file = 'prediction_logs.csv'
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow([
                'Date and Time', 'Total Customers', 'Churn Distribution', 'Actual Churn Customer IDs', 
                'Actual Non Churn Customer IDs', 'Predicted Churn Customer IDs', 
                'Predicted Non Churn Customer IDs', 'Matched', 'Mismatched', 
                'False Positives', 'Input Tokens', 'Output Tokens', 
                'Total Tokens', 'Model', 'Total Cost', 'Additional Prompt', 'Given Date', 'Number of Weeks'
            ])
        
        # Calculate metrics
        total_customers = data['total_customers']
        actual_churn_count = len(data['actual_churned_customers'])
        actual_non_churn_count = total_customers - actual_churn_count
        churn_distribution = f"{actual_churn_count}:{actual_non_churn_count}"
        
        actual_churn_ids = ','.join(data['actual_churned_customers'])
        actual_non_churn_ids = ','.join([cid for cid in data.get('all_customer_ids', []) if cid not in data['actual_churned_customers']])
        predicted_churn_ids = ','.join(data['churned_customers'])
        predicted_non_churn_ids = ','.join([cid for cid in data.get('all_customer_ids', []) if cid not in data['churned_customers']])
        
        # Calculate matches and mismatches using the specified conditions
        actual_set = set(data['actual_churned_customers'])
        predicted_set = set(data['churned_customers'])
        
        correctly_identified = len(actual_set.intersection(predicted_set))
        missed_churners = len(actual_set - predicted_set)
        false_alarms = len(predicted_set - actual_set)
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_customers,
            churn_distribution,
            actual_churn_ids,
            actual_non_churn_ids,
            predicted_churn_ids,
            predicted_non_churn_ids,
            correctly_identified,
            missed_churners,
            false_alarms,
            data['usage']['input_tokens'],
            data['usage']['output_tokens'],
            data['usage']['total_tokens'],
            data['usage']['model'],
            data['usage']['total_cost'],
            data.get('custom_prompt', ''),
            data.get('given_date', ''),
            data.get('num_weeks', '')
        ])

@app.get("/prediction-logs")
def get_prediction_logs():
    """Retrieve all prediction logs"""
    csv_file = 'prediction_logs.csv'
    if not os.path.exists(csv_file):
        return {"logs": []}
    
    try:
        df = pd.read_csv(csv_file)
        # Replace NaN values with None for JSON serialization
        df = df.where(pd.notna(df), None)
        
        # Convert to records and handle any remaining NaN issues
        records = []
        for index, row in df.iterrows():
            record = {}
            for col, value in row.items():
                if pd.isna(value) or value == 'nan':
                    record[col] = None
                else:
                    # Ensure ID columns are properly formatted as comma-separated strings
                    if 'Customer IDs' in col and isinstance(value, str):
                        # Clean up any potential formatting issues and handle empty values
                        cleaned_value = value.strip()
                        record[col] = cleaned_value if cleaned_value and cleaned_value != 'nan' else ''
                    else:
                        record[col] = value
            record['row_index'] = index  # Add row index for deletion
            records.append(record)
        
        return {"logs": records}
    except Exception as e:
        return {"logs": [], "error": str(e)}

@app.get("/prediction-logs/csv")
def get_prediction_logs_csv():
    """Retrieve prediction logs in CSV format for download"""
    csv_file = 'prediction_logs.csv'
    if not os.path.exists(csv_file):
        return {"error": "No prediction logs file found"}
    
    try:
        df = pd.read_csv(csv_file)
        
        # Add row index column
        df['Row Index'] = df.index
        
        # Calculate accuracy for each row and add it after False Positives column
        if 'False Positives' in df.columns:
            # Calculate accuracy: (Matched + (Total Customers - Matched - Mismatched - False Positives)) / Total Customers
            df['Accuracy'] = df.apply(
                lambda row: f"{((row['Matched'] + (row['Total Customers'] - row['Matched'] - row['Mismatched'] - row['False Positives'])) / row['Total Customers'] * 100):.1f}%" 
                if row['Total Customers'] > 0 else "0.0%", 
                axis=1
            )
            
            # Reorder columns to put Accuracy after False Positives
            cols = list(df.columns)
            false_positives_idx = cols.index('False Positives')
            cols.insert(false_positives_idx + 1, cols.pop(cols.index('Accuracy')))
            df = df[cols]
        
        # Reorder columns to put Row Index first
        cols = ['Row Index'] + [col for col in df.columns if col != 'Row Index']
        df = df[cols]
        
        # Process ID columns to format them properly for CSV download
        id_columns = [
            'Actual Churn Customer IDs',
            'Actual Non Churn Customer IDs', 
            'Predicted Churn Customer IDs',
            'Predicted Non Churn Customer IDs'
        ]
        
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: '\n'.join([id.strip() for id in str(x).split(',') if id.strip() and len(id.strip()) == 36])
                    if pd.notna(x) and str(x).strip() and str(x).strip() != 'nan' else ''
                )
        
        # Convert to CSV string
        csv_content = df.to_csv(index=False)
        return {"csv_content": csv_content}
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}

@app.delete("/prediction-logs/{row_index}")
def delete_prediction_log(row_index: int):
    """Delete a specific prediction log by row index"""
    csv_file = 'prediction_logs.csv'
    if not os.path.exists(csv_file):
        return {"error": "No prediction logs file found"}
    
    try:
        df = pd.read_csv(csv_file)
        
        if row_index < 0 or row_index >= len(df):
            return {"error": f"Row index {row_index} out of range"}
        
        # Remove the row
        df = df.drop(index=row_index).reset_index(drop=True)
        
        # Save back to CSV
        df.to_csv(csv_file, index=False)
        
        return {"success": True, "message": f"Row {row_index} deleted successfully"}
    except Exception as e:
        return {"error": f"Failed to delete row: {str(e)}"}

@app.get("/dataset")
def get_dataset(churn_count: int = 1, non_churn_count: int = 4, given_date: str = None, num_weeks: int = 20):
    """Get the dataset that will be used for prediction"""
    # Use provided given_date or default to today
    if given_date is None:
        given_date = date.today().strftime('%Y-%m-%d')
    
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor(dictionary=True)

        # Check if the table exists
        cursor.execute("SHOW TABLES LIKE 'customer_tx_weekly'")
        tables = cursor.fetchall()
        if not tables:
            return {"error": "Table 'customer_tx_weekly' not found", "dataset": []}
        
        # Check date range in the table and adjust if needed
        cursor.execute("SELECT MAX(week_end_date) as max_date FROM customer_tx_weekly")
        date_range = cursor.fetchone()
        if date_range and date_range['max_date']:
            if given_date > date_range['max_date'].strftime('%Y-%m-%d'):
                given_date = date_range['max_date'].strftime('%Y-%m-%d')

        # Main query to get data
        query = f"""
        SELECT 
            t.customer_id,
            t.week_end_date,
            t.order_count,
            t.order_total, 
            t.discount_total,
            t.loyalty_earned,
            CASE 
                WHEN DATEDIFF('{given_date}', COALESCE(last_orders.last_order_date, '1900-01-01')) > 90 THEN 1
                ELSE 0
            END AS is_churn
        FROM customer_tx_weekly t
        LEFT JOIN (
            SELECT 
                customer_id,
                MAX(week_end_date) AS last_order_date
            FROM customer_tx_weekly
            WHERE week_end_date <= '{given_date}'
            GROUP BY customer_id
        ) AS last_orders ON t.customer_id = last_orders.customer_id
        WHERE t.week_end_date BETWEEN DATE_SUB('{given_date}', INTERVAL {num_weeks} WEEK) AND '{given_date}'
        ORDER BY is_churn DESC, t.customer_id, t.week_end_date DESC
        """
        
        cursor.execute(query)
        all_rows = cursor.fetchall()
        
        if not all_rows:
            cursor.close()
            conn.close()
            return {"error": "No data found for the specified parameters", "dataset": []}

        # Group by customer_id to get unique customers
        customer_data = {}
        for row in all_rows:
            customer_id = row['customer_id']
            if customer_id not in customer_data:
                customer_data[customer_id] = []
            customer_data[customer_id].append(row)
        
        # Separate churn and non-churn customers
        churn_customers = []
        non_churn_customers = []
        
        for customer_id, rows in customer_data.items():
            is_churn = any(row['is_churn'] == 1 for row in rows)
            if is_churn:
                churn_customers.append(customer_id)
            else:
                non_churn_customers.append(customer_id)
        
        # Select the required number of customers
        selected_churn = churn_customers[:churn_count]
        selected_non_churn = non_churn_customers[:non_churn_count]
        
        # Get data for selected customers
        selected_customers = selected_churn + selected_non_churn
        final_rows = []
        
        for customer_id in selected_customers:
            if customer_id in customer_data:
                final_rows.extend(customer_data[customer_id])
        
        cursor.close()
        conn.close()

        # Convert datetime objects to strings for JSON serialization
        for row in final_rows:
            if 'week_end_date' in row and row['week_end_date']:
                row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')

        return {"dataset": final_rows}
        
    except Exception as e:
        return {"error": f"Database error: {str(e)}", "dataset": []}

@app.get("/dataset/shuffled")
def get_shuffled_dataset(churn_count: int = 1, non_churn_count: int = 4, given_date: str = None, num_weeks: int = 20):
    """Get the dataset with customer groups shuffled"""
    # Use provided given_date or default to today
    if given_date is None:
        given_date = date.today().strftime('%Y-%m-%d')
    
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor(dictionary=True)

        # Check if the table exists
        cursor.execute("SHOW TABLES LIKE 'customer_tx_weekly'")
        tables = cursor.fetchall()
        if not tables:
            return {"error": "Table 'customer_tx_weekly' not found", "dataset": []}
        
        # Check date range in the table and adjust if needed
        cursor.execute("SELECT MAX(week_end_date) as max_date FROM customer_tx_weekly")
        date_range = cursor.fetchone()
        if date_range and date_range['max_date']:
            if given_date > date_range['max_date'].strftime('%Y-%m-%d'):
                given_date = date_range['max_date'].strftime('%Y-%m-%d')

        # Main query to get data
        query = f"""
        SELECT 
            t.customer_id,
            t.week_end_date,
            t.order_count,
            t.order_total, 
            t.discount_total,
            t.loyalty_earned,
            CASE 
                WHEN DATEDIFF('{given_date}', COALESCE(last_orders.last_order_date, '1900-01-01')) > 90 THEN 1
                ELSE 0
            END AS is_churn
        FROM customer_tx_weekly t
        LEFT JOIN (
            SELECT 
                customer_id,
                MAX(week_end_date) AS last_order_date
            FROM customer_tx_weekly
            WHERE week_end_date <= '{given_date}'
            GROUP BY customer_id
        ) AS last_orders ON t.customer_id = last_orders.customer_id
        WHERE t.week_end_date BETWEEN DATE_SUB('{given_date}', INTERVAL {num_weeks} WEEK) AND '{given_date}'
        ORDER BY is_churn DESC, t.customer_id, t.week_end_date DESC
        """
        
        cursor.execute(query)
        all_rows = cursor.fetchall()
        
        if not all_rows:
            cursor.close()
            conn.close()
            return {"error": "No data found for the specified parameters", "dataset": []}

        # Group data by customer_id
        grouped_data = defaultdict(list)
        for row in all_rows:
            grouped_data[row['customer_id']].append(row)

        # Shuffle the customer groups
        customer_ids = list(grouped_data.keys())
        random.shuffle(customer_ids)

        # Reconstruct the dataset with shuffled order
        shuffled_rows = []
        for customer_id in customer_ids:
            shuffled_rows.extend(grouped_data[customer_id])

        cursor.close()
        conn.close()

        # Convert datetime objects to strings for JSON serialization
        for row in shuffled_rows:
            if 'week_end_date' in row and row['week_end_date']:
                row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')

        return {"dataset": shuffled_rows}
        
    except Exception as e:
        return {"error": f"Database error: {str(e)}", "dataset": []}

@app.post("/predict-churn")
def predict_churn(request: ChurnRequest):
    # Use shuffled data if provided, otherwise fetch from database
    if request.shuffled_data:
        rows = request.shuffled_data
    else:
        try:
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                database=os.getenv("DB_NAME")
            )
            cursor = conn.cursor(dictionary=True)

            # Check if the table exists
            cursor.execute("SHOW TABLES LIKE 'customer_tx_weekly'")
            tables = cursor.fetchall()
            if not tables:
                return {"error": "Table 'customer_tx_weekly' not found", "dataset": []}
            
            # Check date range in the table and adjust if needed
            cursor.execute("SELECT MAX(week_end_date) as max_date FROM customer_tx_weekly")
            date_range = cursor.fetchone()
            if date_range and date_range['max_date']:
                if request.given_date > date_range['max_date'].strftime('%Y-%m-%d'):
                    request.given_date = date_range['max_date'].strftime('%Y-%m-%d')

            # Main query to get data
            query = f"""
            SELECT 
                t.customer_id,
                t.week_end_date,
                t.order_count,
                t.order_total, 
                t.discount_total,
                t.loyalty_earned,
                CASE 
                    WHEN DATEDIFF('{request.given_date}', COALESCE(last_orders.last_order_date, '1900-01-01')) > 90 THEN 1
                    ELSE 0
                END AS is_churn
            FROM customer_tx_weekly t
            LEFT JOIN (
                SELECT 
                    customer_id,
                    MAX(week_end_date) AS last_order_date
                FROM customer_tx_weekly
                WHERE week_end_date <= '{request.given_date}'
                GROUP BY customer_id
            ) AS last_orders ON t.customer_id = last_orders.customer_id
            WHERE t.week_end_date BETWEEN DATE_SUB('{request.given_date}', INTERVAL {request.num_weeks} WEEK) AND '{request.given_date}'
            ORDER BY is_churn DESC, t.customer_id, t.week_end_date DESC
            """
            
            cursor.execute(query)
            all_rows = cursor.fetchall()
            
            if not all_rows:
                cursor.close()
                conn.close()
                return {"error": "No data found for the specified parameters", "dataset": []}

            # Group by customer_id to get unique customers
            customer_data = {}
            for row in all_rows:
                customer_id = row['customer_id']
                if customer_id not in customer_data:
                    customer_data[customer_id] = []
                customer_data[customer_id].append(row)
            
            # Separate churn and non-churn customers
            churn_customers = []
            non_churn_customers = []
            
            for customer_id, rows in customer_data.items():
                is_churn = any(row['is_churn'] == 1 for row in rows)
                if is_churn:
                    churn_customers.append(customer_id)
                else:
                    non_churn_customers.append(customer_id)
            
            # Select the required number of customers
            selected_churn = churn_customers[:request.churn_count]
            selected_non_churn = non_churn_customers[:request.non_churn_count]
            
            # Get data for selected customers
            selected_customers = selected_churn + selected_non_churn
            rows = []
            
            for customer_id in selected_customers:
                if customer_id in customer_data:
                    rows.extend(customer_data[customer_id])
            
            cursor.close()
            conn.close()

            # Convert datetime objects to strings for JSON serialization
            for row in rows:
                if 'week_end_date' in row and row['week_end_date']:
                    row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')
                    
        except Exception as e:
            return {"error": f"Database error: {str(e)}", "dataset": []}

    grouped_data = defaultdict(list)
    for row in rows:
        grouped_data[row['customer_id']].append(row)

    # Combine all customer data into one prompt
    customer_blocks = []
    for customer_id, weeks in grouped_data.items():
        block = f"Customer ID: {customer_id}\n"
        for w in weeks:
            block += (
                f"Week: {w['week_end_date']}, "
                f"Orders: {w['order_count']}, "
                f"Total: {w['order_total']}, "
                f"Discount: {w['discount_total']}, "
                f"Loyalty: {w['loyalty_earned']}\n"
            )
        customer_blocks.append(block)

    all_customers_text = "\n---\n".join(customer_blocks)
    # save csv
    with open('customer_data.csv', 'w') as f:
        f.write(all_customers_text)

    # Calculate the prediction target date (next week after given_date)
    given_date_obj = datetime.strptime(request.given_date, '%Y-%m-%d').date()
    prediction_target_date = (given_date_obj + timedelta(days=7)).strftime('%Y-%m-%d')

    system_message = {
        "role": "system",
        "content": (
            "You are a churn prediction analyst.\n"
            f"Given weekly order history per customer up to {request.given_date}, identify which customers are likely to churn in the week following {request.given_date}.\n"
            "Only return a list of customer_ids who are likely to churn."
        )
    }

    default_content = (
        f"Here is the weekly order data for the past {request.num_weeks} weeks leading up to {request.given_date} for multiple customers.\n"
        f"---\n{all_customers_text}\n---\n"
        f"Consider a customer as 'churned' if they have been inactive (no orders) for the recent 12 weeks.\n"
        f"Based on this historical pattern analysis, which customers will churn in the week of {prediction_target_date}? Respond with a list of customer_ids only."
        f"\n\nNote: Use the {request.num_weeks} weeks of data ending on {request.given_date} to identify customers at risk of churning in the following week."
    )
    
    user_message = {
        "role": "user",
        "content": (
            default_content + f"\n\n{request.custom_prompt}" if request.custom_prompt else default_content
        )
    }

    # Use max_completion_tokens for GPT-5, max_tokens for other models
    if request.model == "gpt-5":
        response = client.chat.completions.create(
            model=request.model,
            messages=[system_message, user_message],
            max_completion_tokens=5000,  # Increase from 300 to 1000
            response_format={"type": "text"},
            seed=42
        )
    else:
        response = client.chat.completions.create(
            model=request.model,
            messages=[system_message, user_message],
            temperature=0.0,
            max_tokens=300
        )

    output = response.choices[0].message.content.strip()
    predicted_ids = list(set(re.findall(r"[a-f0-9\\-]{36}", output)))  # Remove duplicates

    # Get actual churned customers from the selected sample
    actual_churned = [customer_id for customer_id, weeks in grouped_data.items() 
                     if any(w['is_churn'] == 1 for w in weeks)]

    # Calculate cost based on selected model
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    
    model_pricing = MODEL_PRICING.get(request.model, MODEL_PRICING["gpt-3.5-turbo"])
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    # Prepare response data
    response_data = {
        "churned_customers": predicted_ids, 
        "actual_churned_customers": actual_churned,
        "raw_output": output,
        "total_customers": len(grouped_data),
        "all_customer_ids": list(grouped_data.keys()),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "model": response.model
        },
        "custom_prompt": request.custom_prompt,
        "given_date": request.given_date,
        "num_weeks": request.num_weeks
    }

    # Log prediction to CSV
    log_prediction_to_csv(response_data)

    return response_data
