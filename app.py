import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor LLM", layout="wide")
st.title("🧠 Churn Prediction using OpenAI LLM")

col1, col2, col3 = st.columns(3)

with col1:
    churn_count = st.number_input("Churn Customer Sample Size", min_value=1, value=1)

with col2:
    non_churn_count = st.number_input("Non-Churn Customer Sample Size", min_value=1, value=4)

with col3:
    model = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-4o"],
        index=0,
        help="Choose the OpenAI model for prediction"
    )

# Custom prompt section
st.subheader("📝 Custom Prompt (Optional)")
custom_prompt = st.text_area(
    "Additional Instructions",
    value="",
    height=150,
    help="Add additional instructions to append to the default prompt. Leave empty to use only default prompt."
)

if st.button("Predict Churn"):
    with st.spinner("Processing..."):
        res = requests.post("http://localhost:8000/predict-churn", json={
            "churn_count": churn_count,
            "non_churn_count": non_churn_count,
            "model": model,
            "custom_prompt": custom_prompt if custom_prompt.strip() else None
        })

        if res.status_code == 200:
            data = res.json()
            st.success("✅ Prediction complete!")
            
            # Display usage information
            if "usage" in data:
                usage = data["usage"]
                st.info(f"**Model:** {usage['model']} | **Total Cost:** ${usage['total_cost']:.6f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Tokens", usage['input_tokens'])
                with col2:
                    st.metric("Output Tokens", usage['output_tokens'])
                with col3:
                    st.metric("Total Tokens", usage['total_tokens'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Predicted Churned Customers:**")
                if data["churned_customers"]:
                    for customer_id in data["churned_customers"]:
                        st.write(f"- {customer_id}")
                else:
                    st.write("No customers predicted to churn")
            
            with col2:
                st.write("**Actual Churned Customers:**")
                if data["actual_churned_customers"]:
                    for customer_id in data["actual_churned_customers"]:
                        st.write(f"- {customer_id}")
                else:
                    st.write("No actual churned customers in sample")
        else:
            st.error("❌ Failed to get predictions")