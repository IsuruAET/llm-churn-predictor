import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Predictor LLM", layout="wide")

# Create tabs
tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 Prediction Logs"])

with tab1:
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

with tab2:
    st.title("📊 Prediction Logs")
    
    # Add refresh button
    if st.button("🔄 Refresh Logs"):
        st.rerun()
    
    try:
        # Fetch prediction logs
        res = requests.get("http://localhost:8000/prediction-logs")
        
        if res.status_code == 200:
            data = res.json()
            
            if data["logs"]:
                # Convert to DataFrame for better display
                df = pd.DataFrame(data["logs"])
                
                # Format the DataFrame for better display
                if not df.empty:
                    # Format cost columns
                    if 'Total Cost' in df.columns:
                        df['Total Cost'] = df['Total Cost'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "$0.000000")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", len(df))
                    with col2:
                        avg_accuracy = df['Matched'].sum() / df['Total Customers'].sum() * 100 if df['Total Customers'].sum() > 0 else 0
                        st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
                    with col3:
                        total_cost = sum([float(str(x).replace('$', '')) for x in df['Total Cost'] if pd.notna(x)])
                        st.metric("Total Cost", f"${total_cost:.6f}")
                    with col4:
                        total_tokens = df['Total Tokens'].sum() if 'Total Tokens' in df.columns else 0
                        st.metric("Total Tokens", f"{total_tokens:,}")
                    
                    # Display the grid
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Date and Time": st.column_config.DatetimeColumn("Date and Time", format="DD-MM-YYYY HH:mm:ss"),
                            "Total Cost": st.column_config.TextColumn("Total Cost", help="Cost in USD"),
                            "Additional Prompt": st.column_config.TextColumn("Additional Prompt", help="Custom prompt used", width="medium")
                        }
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="prediction_logs.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No prediction logs found. Run some predictions first!")
            else:
                st.info("No prediction logs found. Run some predictions first!")
        else:
            st.error("❌ Failed to fetch prediction logs")
            
    except Exception as e:
        st.error(f"❌ Error loading prediction logs: {str(e)}")
        st.info("Make sure the backend server is running on http://localhost:8000")