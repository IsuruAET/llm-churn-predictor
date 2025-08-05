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
        churn_count = st.slider("Churn Customer Sample Size", min_value=1, max_value=10, value=1)

    with col2:
        non_churn_count = st.slider("Non-Churn Customer Sample Size", min_value=1, max_value=40, value=4)

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

    # Initialize session state for auto-refresh
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

    if st.button("🔄 Refresh Logs") or st.session_state.auto_refresh:
        st.session_state.auto_refresh = False
        st.rerun()

    try:
        res = requests.get("http://localhost:8000/prediction-logs")

        if res.status_code == 200:
            data = res.json()

            if data["logs"]:
                df = pd.DataFrame(data["logs"])

                if not df.empty:
                    # Sort by timestamp or ID to show most recent records first
                    if 'Timestamp' in df.columns:
                        df = df.sort_values('Timestamp', ascending=False)
                    elif 'ID' in df.columns:
                        df = df.sort_values('ID', ascending=False)
                    else:
                        # If no timestamp/ID column, reverse the order to show newest first
                        df = df.iloc[::-1].reset_index(drop=True)
                    if 'Total Cost' in df.columns:
                        df['Total Cost'] = df['Total Cost'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "$0.000000")

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

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Predictions", len(df))
                    with col2:
                        # Calculate overall accuracy: (True Positives + True Negatives) / Total Predictions
                        true_positives = df['Matched'].sum()  # Correctly identified churners
                        true_negatives = df['Total Customers'].sum() - (df['Matched'].sum() + df['Mismatched'].sum() + df['False Positives'].sum())
                        total_predictions = df['Total Customers'].sum()
                        accuracy = (true_positives + true_negatives) / total_predictions * 100 if total_predictions > 0 else 0
                        st.metric("Accuracy", f"{accuracy:.1f}%", help="Overall correct predictions")
                    with col3:
                        # Calculate recall: True Positives / (True Positives + False Negatives)
                        false_negatives = df['Mismatched'].sum()  # Missed churners
                        recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
                        st.metric("Recall", f"{recall:.1f}%", help="Correct detection of actual positives")
                    with col4:
                        total_cost = sum([float(str(x).replace('$', '')) for x in df['Total Cost'] if pd.notna(x)])
                        st.metric("Total Cost", f"${total_cost:.6f}")
                    with col5:
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
                                lambda x: "<ul>" + "".join(f"<li>{id.strip()}</li>" for id in str(x).split(',') if id.strip() and len(id.strip()) == 36) + "</ul>"
                                if pd.notna(x) and str(x).strip() and str(x).strip() != 'nan' else ''
                            )

                    # Move row_index to first column and rename it
                    if 'row_index' in df.columns:
                        # Reorder columns to put row_index first
                        cols = ['row_index'] + [col for col in df.columns if col != 'row_index']
                        df = df[cols]
                        # Rename the column
                        df = df.rename(columns={'row_index': 'Row Index'})

                    # Render HTML table manually with delete buttons
                    st.subheader("📋 Prediction Log (List View)")

                    html_table = "<table style='width:100%; border-collapse: collapse; background-color: black; color: white; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;' border='1'>"
                    html_table += "<tr style='background-color: #333;'>" + "".join(f"<th style='padding:8px; color: white; font-weight: bold; border: 1px solid #555; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;'>{col}</th>" for col in df.columns) + "<th style='padding:8px; color: white; font-weight: bold; border: 1px solid #555; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;'>Actions</th></tr>"

                    for _, row in df.iterrows():
                        row_index = row.get('Row Index', _)
                        html_table += "<tr style='background-color: black;'>" + "".join(
                            f"<td style='vertical-align:top; padding:8px; color: white; border: 1px solid #555; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;'>{row[col]}</td>" for col in df.columns
                        ) + f"<td style='vertical-align:top; padding:8px; text-align:center; border: 1px solid #555; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;'>"
                        html_table += f"<button onclick='deleteRow({row_index})' style='background-color: #ff4444; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;'>🗑️ Delete</button>"
                        html_table += "</td></tr>"

                    html_table += "</table>"

                    # Add JavaScript for delete functionality
                    delete_js = """
                    <script>
                    function deleteRow(rowIndex) {
                        if (confirm('Are you sure you want to delete this record?')) {
                            fetch(`http://localhost:8000/prediction-logs/${rowIndex}`, {
                                method: 'DELETE'
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    // Show success message and trigger page reload
                                    alert('Record deleted successfully!');
                                    // Force a hard reload to refresh the data
                                    window.location.href = window.location.href;
                                } else {
                                    alert('Error: ' + data.error);
                                }
                            })
                            .catch(error => {
                                alert('Error: ' + error);
                            });
                        }
                    }
                    </script>
                    """
                    
                    # Use st.components.html to execute JavaScript
                    import streamlit.components.v1 as components
                    components.html(html_table + delete_js, height=600, scrolling=True)

                    # CSV download using dedicated endpoint
                    try:
                        csv_res = requests.get("http://localhost:8000/prediction-logs/csv")
                        if csv_res.status_code == 200:
                            csv_data = csv_res.json()
                            if "csv_content" in csv_data:
                                # Generate filename with date and time
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"prediction_logs_{timestamp}.csv"
                                
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data["csv_content"],
                                    file_name=filename,
                                    mime="text/csv"
                                )
                            else:
                                st.error("❌ Failed to get CSV content")
                        else:
                            st.error("❌ Failed to get CSV data")
                    except Exception as e:
                        st.error(f"❌ Error downloading CSV: {str(e)}")
                else:
                    st.info("No prediction logs found. Run some predictions first!")
            else:
                st.info("No prediction logs found. Run some predictions first!")
        else:
            st.error("❌ Failed to fetch prediction logs")

    except Exception as e:
        st.error(f"❌ Error loading prediction logs: {str(e)}")
        st.info("Make sure the backend server is running on http://localhost:8000")
