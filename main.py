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
from datetime import datetime
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
    model: str = "gpt-3.5-turbo"
    custom_prompt: Optional[str] = None
    shuffled_data: Optional[list] = None  # Add support for shuffled data

# Model pricing (per 1K tokens)
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
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
                'Total Tokens', 'Model', 'Total Cost', 'Additional Prompt'
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
            data.get('custom_prompt', '')
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
def get_dataset(churn_count: int = 1, non_churn_count: int = 4):
    """Get the dataset that will be used for prediction"""
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    cursor = conn.cursor(dictionary=True)

    query = f"""
    WITH churn_customers AS (
        SELECT customer_id
        FROM (
            SELECT customer_id, MAX(week_end_date) AS last_week
            FROM sample_data
            WHERE is_churn = 1
            GROUP BY customer_id
            ORDER BY last_week DESC
            LIMIT {churn_count}
        ) AS ordered_churn
    ),
    non_churn_customers AS (
        SELECT customer_id
        FROM (
            SELECT customer_id, MAX(week_end_date) AS last_week
            FROM sample_data
            WHERE is_churn = 0
            GROUP BY customer_id
            ORDER BY last_week DESC
            LIMIT {non_churn_count}
        ) AS ordered_non_churn
    ),
    selected_customers AS (
        SELECT customer_id FROM churn_customers
        UNION ALL
        SELECT customer_id FROM non_churn_customers
    )
    SELECT 
        sd.customer_id,
        sd.week_end_date,
        sd.order_count,
        sd.order_total, 
        sd.discount_total,
        sd.loyalty_earned,
        sd.is_churn
    FROM sample_data sd
    JOIN selected_customers sc ON sd.customer_id = sc.customer_id
    ORDER BY sd.is_churn, sd.customer_id, sd.week_end_date DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert datetime objects to strings for JSON serialization
    for row in rows:
        if 'week_end_date' in row and row['week_end_date']:
            row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')

    return {"dataset": rows}

@app.get("/dataset/shuffled")
def get_shuffled_dataset(churn_count: int = 1, non_churn_count: int = 4):
    """Get the dataset with customer groups shuffled"""
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    cursor = conn.cursor(dictionary=True)

    query = f"""
    WITH churn_customers AS (
        SELECT customer_id
        FROM (
            SELECT customer_id, MAX(week_end_date) AS last_week
            FROM sample_data
            WHERE is_churn = 1
            GROUP BY customer_id
            ORDER BY last_week DESC
            LIMIT {churn_count}
        ) AS ordered_churn
    ),
    non_churn_customers AS (
        SELECT customer_id
        FROM (
            SELECT customer_id, MAX(week_end_date) AS last_week
            FROM sample_data
            WHERE is_churn = 0
            GROUP BY customer_id
            ORDER BY last_week DESC
            LIMIT {non_churn_count}
        ) AS ordered_non_churn
    ),
    selected_customers AS (
        SELECT customer_id FROM churn_customers
        UNION ALL
        SELECT customer_id FROM non_churn_customers
    )
    SELECT 
        sd.customer_id,
        sd.week_end_date,
        sd.order_count,
        sd.order_total, 
        sd.discount_total,
        sd.loyalty_earned,
        sd.is_churn
    FROM sample_data sd
    JOIN selected_customers sc ON sd.customer_id = sc.customer_id
    ORDER BY sd.is_churn, sd.customer_id, sd.week_end_date DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert datetime objects to strings for JSON serialization
    for row in rows:
        if 'week_end_date' in row and row['week_end_date']:
            row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')

    # Group data by customer_id
    grouped_data = defaultdict(list)
    for row in rows:
        grouped_data[row['customer_id']].append(row)

    # Shuffle the customer groups
    customer_ids = list(grouped_data.keys())
    random.shuffle(customer_ids)

    # Reconstruct the dataset with shuffled order
    shuffled_rows = []
    for customer_id in customer_ids:
        shuffled_rows.extend(grouped_data[customer_id])

    return {"dataset": shuffled_rows}

@app.post("/predict-churn")
def predict_churn(request: ChurnRequest):
    # Use shuffled data if provided, otherwise fetch from database
    if request.shuffled_data:
        rows = request.shuffled_data
    else:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor(dictionary=True)

        query = f"""
        WITH churn_customers AS (
            SELECT customer_id
            FROM (
                SELECT customer_id, MAX(week_end_date) AS last_week
                FROM sample_data
                WHERE is_churn = 1
                GROUP BY customer_id
                ORDER BY last_week DESC
                LIMIT {request.churn_count}
            ) AS ordered_churn
        ),
        non_churn_customers AS (
            SELECT customer_id
            FROM (
                SELECT customer_id, MAX(week_end_date) AS last_week
                FROM sample_data
                WHERE is_churn = 0
                GROUP BY customer_id
                ORDER BY last_week DESC
                LIMIT {request.non_churn_count}
            ) AS ordered_non_churn
        ),
        selected_customers AS (
            SELECT customer_id FROM churn_customers
            UNION ALL
            SELECT customer_id FROM non_churn_customers
        )
        SELECT 
            sd.customer_id,
            sd.week_end_date,
            sd.order_count,
            sd.order_total, 
            sd.discount_total,
            sd.loyalty_earned,
            sd.is_churn
        FROM sample_data sd
        JOIN selected_customers sc ON sd.customer_id = sc.customer_id
        ORDER BY sd.is_churn, sd.customer_id, sd.week_end_date DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Convert datetime objects to strings for JSON serialization
        for row in rows:
            if 'week_end_date' in row and row['week_end_date']:
                row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')

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

    system_message = {
        "role": "system",
        "content": (
            "You are a churn prediction analyst.\n"
            "Given weekly order history per customer, identify which customers are likely to churn next week.\n"
            "Only return a list of customer_ids who are likely to churn."
        )
    }

    default_content = (
        f"Here is the recent weekly order data (last 20 weeks) for multiple customers.\n"
        f"---\n{all_customers_text}\n---\n"
        f"Consider a customer as 'churned' if they have been inactive (no orders) for the recent 12 weeks.\n"
        f"Which customers will churn next week? Respond with a list of customer_ids only."
    )
    
    user_message = {
        "role": "user",
        "content": (
            default_content + f"\n\n{request.custom_prompt}" if request.custom_prompt else default_content
        )
    }

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
        "custom_prompt": request.custom_prompt
    }

    # Log prediction to CSV
    log_prediction_to_csv(response_data)

    return response_data
