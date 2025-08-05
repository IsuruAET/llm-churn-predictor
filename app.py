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
                    st.write("**Actual Churned Customers:**")
                    if data["actual_churned_customers"]:
                        for customer_id in data["actual_churned_customers"]:
                            st.write(f"- {customer_id}")
                    else:
                        st.write("No actual churned customers in sample")
                
                with col2:
                    st.write("**Predicted Churned Customers:**")
                    if data["churned_customers"]:
                        for customer_id in data["churned_customers"]:
                            st.write(f"- {customer_id}")
                    else:
                        st.write("No customers predicted to churn")
            else:
                st.error("❌ Failed to get predictions")

import streamlit as st
import pandas as pd
import requests
from datetime import datetime

with tab2:
    st.title("📊 Prediction Logs")

    if st.button("🔄 Refresh Logs"):
        st.rerun()

    try:
        res = requests.get("http://localhost:8000/prediction-logs")

        if res.status_code == 200:
            data = res.json()

            if data["logs"]:
                df = pd.DataFrame(data["logs"])

                if not df.empty:
                    if 'Total Cost' in df.columns:
                        df['Total Cost'] = df['Total Cost'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "$0.000000")

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

                    # Convert comma-separated IDs into HTML <ul><li>...</li></ul> format
                    id_columns = [
                        'Actual Churn Customer IDs',
                        'Actual Non Churn Customer IDs',
                        'Predicted Churn Customer IDs',
                        'Predicted Non Churn Customer IDs'
                    ]

                    for col in id_columns:
                        if col in df.columns:
                            df[col] = df[col].apply(
                                lambda x: "<ul>" + "".join(f"<li>{i.strip()}</li>" for i in str(x).split(',')) + "</ul>"
                                if pd.notna(x) and str(x).strip() else ''
                            )

                    # Render HTML table manually
                    st.subheader("📋 Prediction Log (List View)")

                    html_table = "<table style='width:100%; border-collapse: collapse;' border='1'>"
                    html_table += "<tr>" + "".join(f"<th style='padding:8px'>{col}</th>" for col in df.columns) + "</tr>"

                    for _, row in df.iterrows():
                        html_table += "<tr>" + "".join(
                            f"<td style='vertical-align:top; padding:8px'>{row[col]}</td>" for col in df.columns
                        ) + "</tr>"

                    html_table += "</table>"

                    st.markdown(html_table, unsafe_allow_html=True)

                    # CSV download
                    csv = df.copy()
                    for col in id_columns:
                        if col in csv.columns:
                            csv[col] = csv[col].str.replace(r'<[^>]+>', '', regex=True).str.replace('\n', ', ')
                    # Generate filename with date and time
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"prediction_logs_{timestamp}.csv"
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv.to_csv(index=False),
                        file_name=filename,
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
