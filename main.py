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
import time

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChurnRequest(BaseModel):
    churn_count: int
    non_churn_count: int
    given_date: str = date.today().strftime('%Y-%m-%d')
    num_weeks: int = 20
    model: str = "gpt-3.5-turbo"
    custom_prompt: Optional[str] = None
    data: list

MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "o3": {"input": 0.00015, "output": 0.0006},
    "o4-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5-mini": {"input": 0.001, "output": 0.003},
    "gpt-5": {"input": 0.005, "output": 0.015}
}

MAX_TOKENS_PER_REQUEST = 15000
MAX_CHARS_PER_CHUNK = 12000

def chunk_customer_blocks(blocks, max_chars=MAX_CHARS_PER_CHUNK):
    chunks, current_chunk = [], []
    current_length = 0
    
    for block in blocks:
        block_length = len(block)
        if current_length + block_length > max_chars:
            chunks.append("\n---\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(block)
        current_length += block_length
    
    if current_chunk:
        chunks.append("\n---\n".join(current_chunk))
    
    return chunks

def log_prediction_to_csv(data):
    csv_file = 'prediction_logs.csv'
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'Given Date', 'Total Customers', 'Churn Distribution', 'Actual Churn Customer IDs', 
                'Actual Non Churn Customer IDs', 'Predicted Churn Customer IDs', 
                'Predicted Non Churn Customer IDs', 'Matched', 'Mismatched', 
                'False Positives', 'Input Tokens', 'Output Tokens', 
                'Total Tokens', 'Model', 'Total Cost', 'Response Time (seconds)', 'Additional Prompt', 'Number of Weeks'
            ])
        
        total_customers = data['total_customers']
        actual_churn_count = len(data['actual_churned_customers'])
        actual_non_churn_count = total_customers - actual_churn_count
        churn_distribution = f"{actual_churn_count}:{actual_non_churn_count}"
        
        actual_churn_ids = ','.join(data['actual_churned_customers'])
        actual_non_churn_ids = ','.join([cid for cid in data.get('all_customer_ids', []) if cid not in data['actual_churned_customers']])
        predicted_churn_ids = ','.join(data['churned_customers'])
        predicted_non_churn_ids = ','.join([cid for cid in data.get('all_customer_ids', []) if cid not in data['churned_customers']])
        
        actual_set = set(data['actual_churned_customers'])
        predicted_set = set(data['churned_customers'])
        
        correctly_identified = len(actual_set.intersection(predicted_set))
        missed_churners = len(actual_set - predicted_set)
        false_alarms = len(predicted_set - actual_set)
        
        writer.writerow([
            data.get('given_date', ''),
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
            data.get('response_time', ''),
            data.get('custom_prompt', ''),
            data.get('num_weeks', '')
        ])

@app.get("/prediction-logs")
def get_prediction_logs():
    csv_file = 'prediction_logs.csv'
    if not os.path.exists(csv_file):
        return {"logs": []}
    
    df = pd.read_csv(csv_file)
    df = df.where(pd.notna(df), None)
    
    records = []
    for index, row in df.iterrows():
        record = {}
        for col, value in row.items():
            if pd.isna(value) or value == 'nan':
                record[col] = None
            else:
                if 'Customer IDs' in col and isinstance(value, str):
                    cleaned_value = value.strip()
                    record[col] = cleaned_value if cleaned_value and cleaned_value != 'nan' else ''
                else:
                    record[col] = value
        record['row_index'] = index
        records.append(record)
    
    return {"logs": records}

@app.get("/prediction-logs/csv")
def get_prediction_logs_csv():
    csv_file = 'prediction_logs.csv'
    if not os.path.exists(csv_file):
        return {"error": "No prediction logs file found"}
    
    df = pd.read_csv(csv_file)
    df['Row Index'] = df.index
    
    if 'Churn Distribution' in df.columns:
        df['Real Churn Ratio'] = df['Churn Distribution'].apply(
            lambda x: f"{x.split(':')[0]}:{x.split(':')[1]}" if ':' in str(x) else "0:0"
        )
    
    if 'Predicted Churn Customer IDs' in df.columns:
        df['Predict Churn Ratio'] = df['Predicted Churn Customer IDs'].apply(
            lambda x: f"{len([id.strip() for id in str(x).split(',') if id.strip() and len(id.strip()) == 36])}:{df.loc[df['Predicted Churn Customer IDs'] == x, 'Total Customers'].iloc[0] - len([id.strip() for id in str(x).split(',') if id.strip() and len(id.strip()) == 36])}"
            if pd.notna(x) and str(x).strip() and str(x).strip() != 'nan' else "0:0"
        )
    
    if 'False Positives' in df.columns:
        df['Accuracy'] = df.apply(
            lambda row: f"{((row['Matched'] + (row['Total Customers'] - row['Matched'] - row['Mismatched'] - row['False Positives'])) / row['Total Customers'] * 100):.1f}%" 
            if row['Total Customers'] > 0 else "0.0%", 
            axis=1
        )
        
        df['Recall'] = df.apply(
            lambda row: f"{(row['Matched'] / (row['Matched'] + row['Mismatched']) * 100):.1f}%" 
            if (row['Matched'] + row['Mismatched']) > 0 else "0.0%", 
            axis=1
        )
    
    column_order = [
        'Row Index',
        'Model',
        'Given Date',
        'Total Customers',
        'Real Churn Ratio',
        'Predict Churn Ratio',
        'Accuracy',
        'Recall',
        'Matched',
        'Mismatched',
        'False Positives',
        'Input Tokens',
        'Output Tokens',
        'Total Tokens',
        'Total Cost',
        'Response Time (seconds)',
        'Number of Weeks',
        'Additional Prompt'
    ]
    
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    csv_content = df.to_csv(index=False)
    return {"csv_content": csv_content}

@app.delete("/prediction-logs/{row_index}")
def delete_prediction_log(row_index: int):
    csv_file = 'prediction_logs.csv'
    if not os.path.exists(csv_file):
        return {"error": "No prediction logs file found"}
    
    df = pd.read_csv(csv_file)
    
    if row_index < 0 or row_index >= len(df):
        return {"error": f"Row index {row_index} out of range"}
    
    df = df.drop(index=row_index).reset_index(drop=True)
    df.to_csv(csv_file, index=False)
    
    return {"success": True, "message": f"Row {row_index} deleted successfully"}

@app.get("/dataset")
def get_dataset(churn_count: int = 1, non_churn_count: int = 4, given_date: str = None, num_weeks: int = 20):
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SHOW TABLES LIKE 'customer_tx_weekly'")
    tables = cursor.fetchall()
    if not tables:
        cursor.close()
        conn.close()
        return {"error": "Table 'customer_tx_weekly' not found", "dataset": []}
    
    cursor.execute("SELECT MAX(week_end_date) as max_date FROM customer_tx_weekly")
    date_range = cursor.fetchone()
    if date_range and date_range['max_date']:
        if given_date > date_range['max_date'].strftime('%Y-%m-%d'):
            given_date = date_range['max_date'].strftime('%Y-%m-%d')

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

    customer_data = {}
    for row in all_rows:
        customer_id = row['customer_id']
        if customer_id not in customer_data:
            customer_data[customer_id] = []
        customer_data[customer_id].append(row)
    
    churn_customers = []
    non_churn_customers = []
    
    for customer_id, rows in customer_data.items():
        is_churn = any(row['is_churn'] == 1 for row in rows)
        if is_churn:
            churn_customers.append(customer_id)
        else:
            non_churn_customers.append(customer_id)
    
    selected_churn = churn_customers[:churn_count]
    selected_non_churn = non_churn_customers[:non_churn_count]
    
    selected_customers = selected_churn + selected_non_churn
    final_rows = []
    
    for customer_id in selected_customers:
        if customer_id in customer_data:
            final_rows.extend(customer_data[customer_id])
    
    cursor.close()
    conn.close()

    for row in final_rows:
        if 'week_end_date' in row and row['week_end_date']:
            row['week_end_date'] = row['week_end_date'].strftime('%Y-%m-%d')

    return {"dataset": final_rows}

@app.post("/predict-churn")
def predict_churn(request: ChurnRequest):
    if not request.data:
        return {"error": "Dataset must be provided via data parameter. Please load the dataset in the frontend first.", "dataset": []}
    
    rows = request.data

    grouped_data = defaultdict(list)
    for row in rows:
        grouped_data[row['customer_id']].append(row)

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
    with open('customer_data.csv', 'w') as f:
        f.write(all_customers_text)

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

    chunks = chunk_customer_blocks(customer_blocks)
    
    predicted_ids = set()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    total_response_time = 0.0
    
    default_content_prefix = (
        f"Here is the weekly order data for the past {request.num_weeks} weeks leading up to {request.given_date} for multiple customers.\n"
        f"---\n"
    )
    
    default_content_suffix = (
        f"\n---\n"
        f"A customer is considered 'churned' if and only if they have been inactive (no orders) for the most recent 12 consecutive weeks. If this 12-week inactivity condition is met, they are definitely a churn customer.\n"
        f"Based on this historical pattern analysis, which customers will churn in the week of {prediction_target_date}? Respond with a list of customer_ids only."
        f"\n\nNote: Use the {request.num_weeks} weeks of data ending on {request.given_date} to identify customers at risk of churning in the following week."
    )
    
    for chunk_text in chunks:
        user_message = {
            "role": "user",
            "content": (
                f"{default_content_prefix}{chunk_text}{default_content_suffix}"
                f"{f'\n\n{request.custom_prompt}' if request.custom_prompt else ''}"
            )
        }
        
        chunk_start_time = time.time()
        
        if request.model in ["gpt-5", "gpt-5-mini", "o4-mini", "o3"]:
            response = client.chat.completions.create(
                model=request.model,
                messages=[system_message, user_message],
                max_completion_tokens=5000,
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
        
        chunk_end_time = time.time()
        chunk_response_time = chunk_end_time - chunk_start_time
        total_response_time += chunk_response_time
        
        chunk_output = response.choices[0].message.content.strip()
        chunk_predicted_ids = re.findall(r"[a-f0-9\-]{36}", chunk_output)
        predicted_ids.update(chunk_predicted_ids)
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        model_pricing = MODEL_PRICING.get(request.model, MODEL_PRICING["gpt-3.5-turbo"])
        chunk_input_cost = (response.usage.prompt_tokens / 1000) * model_pricing["input"]
        chunk_output_cost = (response.usage.completion_tokens / 1000) * model_pricing["output"]
        total_cost += chunk_input_cost + chunk_output_cost
    
    predicted_ids = list(predicted_ids)

    actual_churned = [customer_id for customer_id, weeks in grouped_data.items() 
                     if any(w['is_churn'] == 1 for w in weeks)]

    # Calculate mismatches and false positives
    actual_set = set(actual_churned)
    predicted_set = set(predicted_ids)
    
    mismatched_ids = list(actual_set - predicted_set)  # Actual churned but not predicted
    false_positive_ids = list(predicted_set - actual_set)  # Predicted but not actual
    matched_ids = list(actual_set.intersection(predicted_set))

    total_tokens = total_input_tokens + total_output_tokens

    response_data = {
        "churned_customers": predicted_ids, 
        "actual_churned_customers": actual_churned,
        "mismatched_ids": mismatched_ids,
        "false_positive_ids": false_positive_ids,
        "matched_ids": matched_ids,
        "raw_output": f"Processed {len(chunks)} chunks with batching. Total predicted churn customers: {len(predicted_ids)}",
        "total_customers": len(grouped_data),
        "all_customer_ids": list(grouped_data.keys()),
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "input_cost": round(total_cost - (total_output_tokens / 1000) * MODEL_PRICING.get(request.model, MODEL_PRICING["gpt-3.5-turbo"])["output"], 6),
            "output_cost": round((total_output_tokens / 1000) * MODEL_PRICING.get(request.model, MODEL_PRICING["gpt-3.5-turbo"])["output"], 6),
            "total_cost": round(total_cost, 6),
            "model": request.model
        },
        "response_time": round(total_response_time, 3),
        "custom_prompt": request.custom_prompt,
        "given_date": request.given_date,
        "num_weeks": request.num_weeks
    }

    log_prediction_to_csv(response_data)

    return response_data

class AnalysisRequest(BaseModel):
    mismatched_ids: list
    false_positive_ids: list
    customer_data: dict
    given_date: str
    num_weeks: int
    model: str = "gpt-4o"
    custom_prompt: Optional[str] = None

@app.post("/analyze-mismatches")
def analyze_mismatches(request: AnalysisRequest):
    if not request.mismatched_ids and not request.false_positive_ids:
        return {"error": "No mismatches or false positives to analyze"}
    
    # Prepare customer data for analysis
    analysis_data = []
    
    for customer_id in request.mismatched_ids + request.false_positive_ids:
        if customer_id in request.customer_data:
            customer_weeks = request.customer_data[customer_id]
            analysis_data.append({
                "customer_id": customer_id,
                "type": "mismatch" if customer_id in request.mismatched_ids else "false_positive",
                "weeks": customer_weeks
            })
    
    # Create analysis prompt
    system_message = {
        "role": "system",
        "content": (
            "You are a churn prediction analyst expert. Analyze why the prediction model failed for these specific customers. "
            "For each customer, provide specific reasons why they were mismatched or false positive. "
            "Focus on behavioral patterns, trends, and specific indicators that the model should have considered."
        )
    }
    
    user_message_content = f"""
Analysis Date: {request.given_date}
Historical Weeks Analyzed: {request.num_weeks}

The following customers had prediction errors:

MISMATCHED IDs (Actual churned but not predicted):
{', '.join(request.mismatched_ids) if request.mismatched_ids else 'None'}

FALSE POSITIVE IDs (Predicted to churn but didn't actually churn):
{', '.join(request.false_positive_ids) if request.false_positive_ids else 'None'}

Customer Data:
"""
    
    for item in analysis_data:
        user_message_content += f"\n--- Customer {item['customer_id']} ({item['type'].upper()}) ---\n"
        for week in item['weeks']:
            user_message_content += (
                f"Week: {week['week_end_date']}, "
                f"Orders: {week['order_count']}, "
                f"Total: {week['order_total']}, "
                f"Discount: {week['discount_total']}, "
                f"Loyalty: {week['loyalty_earned']}\n"
            )
    
    user_message_content += f"""

Provide ONLY brief reasons with evidence for each customer:

**Mismatched Customers (Why model missed their churn):**
{', '.join([f'Customer {id}: [Brief reason with specific data evidence]' for id in request.mismatched_ids]) if request.mismatched_ids else 'None'}

**False Positive Customers (Why model incorrectly flagged them):**
{', '.join([f'Customer {id}: [Brief reason with specific data evidence]' for id in request.false_positive_ids]) if request.false_positive_ids else 'None'}

For each reason, include 1-2 specific data points (e.g., "declining orders from 3 to 0 over last 4 weeks", "order total dropped from $150 to $25"). Keep explanations concise but evidence-based.
"""
    
    user_message = {"role": "user", "content": user_message_content}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            max_completion_tokens=1000,
            response_format={"type": "text"},
            temperature=0.1
        )
        
        analysis_result = response.choices[0].message.content.strip()
        
        return {
            "analysis": analysis_result,
            "model": "gpt-4o"
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
