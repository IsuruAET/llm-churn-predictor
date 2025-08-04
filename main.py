from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
import openai
from collections import defaultdict
import os
from dotenv import load_dotenv
import re
from typing import Optional

load_dotenv()

app = FastAPI()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChurnRequest(BaseModel):
    churn_count: int
    non_churn_count: int
    model: str = "gpt-3.5-turbo"
    custom_prompt: Optional[str] = None

# Model pricing (per 1K tokens)
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
}

@app.post("/predict-churn")
def predict_churn(request: ChurnRequest):
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

    return {
        "churned_customers": predicted_ids, 
        "actual_churned_customers": actual_churned,
        "raw_output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "model": response.model
        }
    }
