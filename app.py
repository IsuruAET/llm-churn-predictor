import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta, datetime
import copy
import random

st.set_page_config(page_title="Churn Predictor LLM", layout="wide")

# Create tabs
tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 Prediction Logs"])

with tab1:
    st.title("🧠 Churn Prediction using OpenAI LLM")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        churn_count = st.slider("Churn Customer Sample Size", min_value=1, max_value=100, value=1)

    with col2:
        non_churn_count = st.slider("Non-Churn Customer Sample Size", min_value=1, max_value=400, value=4)

    with col3:
        default_date = date.today()
        given_date = st.date_input(
            "Given Date",
            value=default_date,
            max_value=default_date,
            help="The reference date for churn analysis"
        )

    with col4:
        num_weeks = st.slider("Number of Weeks", min_value=10, max_value=52, value=20, help="Number of weeks of historical data to analyze")

    # Initialize session state for dataset
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'dataset_data' not in st.session_state:
        st.session_state.dataset_data = None
    if 'original_dataset_data' not in st.session_state:
        st.session_state.original_dataset_data = None
    if 'last_churn_count' not in st.session_state:
        st.session_state.last_churn_count = churn_count
    if 'last_non_churn_count' not in st.session_state:
        st.session_state.last_non_churn_count = non_churn_count
    if 'last_given_date' not in st.session_state:
        st.session_state.last_given_date = given_date
    if 'last_num_weeks' not in st.session_state:
        st.session_state.last_num_weeks = num_weeks

    # Check if inputs have changed and reset dataset if needed
    if (st.session_state.last_churn_count != churn_count or 
        st.session_state.last_non_churn_count != non_churn_count or
        st.session_state.last_given_date != given_date or
        st.session_state.last_num_weeks != num_weeks):
        st.session_state.dataset_loaded = False
        st.session_state.dataset_data = None
        st.session_state.original_dataset_data = None
        st.session_state.last_churn_count = churn_count
        st.session_state.last_non_churn_count = non_churn_count
        st.session_state.last_given_date = given_date
        st.session_state.last_num_weeks = num_weeks

    # Load Data button
    if st.button("📊 Load Data"):
        with st.spinner("Loading dataset..."):
            dataset_res = requests.get(
                f"http://localhost:8000/dataset",
                params={
                    "churn_count": churn_count,
                    "non_churn_count": non_churn_count,
                    "given_date": given_date.strftime('%Y-%m-%d'),
                    "num_weeks": num_weeks
                }
            )
            
            if dataset_res.status_code == 200:
                dataset_data = dataset_res.json()
                
                if dataset_data.get("error"):
                    st.error(f"❌ {dataset_data['error']}")
                elif dataset_data.get("dataset"):
                    if len(dataset_data["dataset"]) > 0:
                        st.session_state.dataset_data = dataset_data
                        st.session_state.original_dataset_data = copy.deepcopy(dataset_data)
                        st.session_state.dataset_loaded = True
                        st.success(f"✅ Dataset loaded successfully! Found {len(dataset_data['dataset'])} records")
                    else:
                        st.warning("No dataset found - empty result")
                else:
                    st.warning("No dataset found - unexpected response format")
            else:
                st.error(f"❌ Failed to load dataset (HTTP {dataset_res.status_code})")

    # Display dataset if loaded
    if st.session_state.dataset_loaded and st.session_state.dataset_data:
        st.subheader("📊 Dataset Used for Prediction")
        
        # Display analysis parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Analysis Date:** {given_date.strftime('%Y-%m-%d')}")
        with col2:
            st.info(f"**Historical Weeks:** {num_weeks}")
        with col3:
            prediction_date = (given_date + timedelta(days=7)).strftime('%Y-%m-%d')
            st.info(f"**Prediction Target:** {prediction_date}")
        
        # Customer ID filter dropdown
        df_dataset = pd.DataFrame(st.session_state.dataset_data["dataset"])
        unique_customer_ids = sorted(df_dataset['customer_id'].unique())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_customer_id = st.selectbox(
                "Filter by Customer ID",
                options=["All Customers"] + unique_customer_ids,
                index=0,
                help="Select a specific customer to filter the dataset"
            )
        
        with col2:
            st.write("")
        
        with col3:
            st.write("")
        
        # Filter dataset based on selection
        if selected_customer_id != "All Customers":
            filtered_dataset = df_dataset[df_dataset['customer_id'] == selected_customer_id]
            st.info(f"📊 Showing data for Customer ID: **{selected_customer_id}** ({len(filtered_dataset)} records)")
        else:
            filtered_dataset = df_dataset
            st.info(f"📊 Showing all customers ({len(filtered_dataset)} records)")
        
        # Shuffle and Reset buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔀 Shuffle Customer Groups"):
                if st.session_state.dataset_data and st.session_state.dataset_data.get("dataset"):
                    current_dataset = copy.deepcopy(st.session_state.dataset_data)
                    
                    # Group data by customer_id
                    customer_groups = {}
                    for row in current_dataset["dataset"]:
                        customer_id = row['customer_id']
                        if customer_id not in customer_groups:
                            customer_groups[customer_id] = []
                        customer_groups[customer_id].append(row)
                    
                    # Shuffle the customer groups
                    customer_ids = list(customer_groups.keys())
                    random.shuffle(customer_ids)
                    
                    # Reconstruct the dataset with shuffled order
                    shuffled_dataset = []
                    for customer_id in customer_ids:
                        shuffled_dataset.extend(customer_groups[customer_id])
                    
                    # Update the session state with shuffled data
                    current_dataset["dataset"] = shuffled_dataset
                    st.session_state.dataset_data = current_dataset
                    st.success("✅ Customer groups shuffled successfully!")
                    st.rerun()
                else:
                    st.warning("No dataset loaded to shuffle")
        
        with col2:
            if st.button("🔄 Reset to Original Order"):
                if st.session_state.original_dataset_data:
                    st.session_state.dataset_data = st.session_state.original_dataset_data
                    st.success("✅ Reset to original order!")
                    st.rerun()
                else:
                    st.warning("No original data to reset to")
        
        with col3:
            st.write("")
        
        # Add churn status column for better visualization
        filtered_dataset['Churn Status'] = filtered_dataset['is_churn'].apply(lambda x: "🔴 Churned" if x == 1 else "🟢 Active")
        
        # Reorder columns for better display
        display_columns = ['customer_id', 'Churn Status', 'week_end_date', 'order_count', 'order_total', 'discount_total', 'loyalty_earned']
        df_display = filtered_dataset[display_columns].copy()
        df_display.columns = ['Customer ID', 'Churn Status', 'Week End Date', 'Order Count', 'Order Total', 'Discount Total', 'Loyalty Earned']
        
        # Display all data in one datagrid
        st.dataframe(df_display, use_container_width=True)
        
        # Show summary stats
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Records", len(filtered_dataset))
        with col2:
            churned_records = filtered_dataset['is_churn'].sum()
            st.metric("Churned Records", churned_records)
        with col3:
            active_records = len(filtered_dataset) - churned_records
            st.metric("Active Records", active_records)
        with col4:
            unique_customers = filtered_dataset['customer_id'].nunique()
            st.metric("Unique Customers", unique_customers)
        with col5:
            churned_customers = filtered_dataset[filtered_dataset['is_churn'] == 1]['customer_id'].nunique()
            active_customers = unique_customers - churned_customers
            churn_distribution = f"{churned_customers}:{active_customers}"
            st.metric("Churn Distribution", churn_distribution, help="Churned:Active customers")

    # Predict Churn button (only show if data is loaded)
    if st.session_state.dataset_loaded:
        st.write("---")
        
        # Churn Prediction section
        st.subheader("🔮 Churn Prediction")
        
        # Show which dataset will be used
        if st.session_state.dataset_data == st.session_state.original_dataset_data:
            st.info("📊 **Dataset:** Using original dataset order")
        else:
            st.info("📊 **Dataset:** Using shuffled dataset order")
        
        st.info("💡 **Note:** The current dataset (original or shuffled) will be sent to the backend for prediction.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox(
                "Select Model",
                ["gpt-3.5-turbo", "gpt-4o", "o3", "o4-mini", "gpt-5-mini", "gpt-5"],
                index=0,
                help="Choose the OpenAI model for prediction"
            )
        
        with col2:
            st.write("")
        
        # Default prompt display
        st.subheader("📋 Default Prompt")
        default_prompt_text = f"""You are a churn prediction analyst.
Given weekly order history per customer up to {given_date.strftime('%Y-%m-%d')}, identify which customers are likely to churn in the week following {given_date.strftime('%Y-%m-%d')}.
Only return a list of customer_ids who are likely to churn.

Here is the recent weekly order data (last {num_weeks} weeks) for multiple customers.
---
[Customer data will be inserted here]
---
A customer is considered 'churned' if and only if they have been inactive (no orders) for the most recent 12 consecutive weeks. If this 12-week inactivity condition is met, they are definitely a churn customer.
Which customers will churn next week? Respond with a list of customer_ids only."""
        
        st.text_area(
            "Default Prompt",
            value=default_prompt_text,
            height=200,
            disabled=True,
            help="This is the default prompt that will be used for prediction"
        )
        
        # Custom prompt section
        st.subheader("📝 Custom Prompt (Optional)")
        custom_prompt = st.text_area(
            "Additional Instructions",
            value="",
            height=150,
            help="Add additional instructions to append to the default prompt. Leave empty to use only default prompt."
        )
        
        if st.button("🔮 Predict Churn"):
            if not st.session_state.dataset_loaded or not st.session_state.dataset_data:
                st.error("❌ Please load the dataset first before making predictions!")
                st.stop()
                
            with st.spinner("Processing prediction..."):
                request_data = {
                    "churn_count": churn_count,
                    "non_churn_count": non_churn_count,
                    "given_date": given_date.strftime('%Y-%m-%d'),
                    "num_weeks": num_weeks,
                    "model": model,
                    "custom_prompt": custom_prompt if custom_prompt.strip() else None,
                    "data": st.session_state.dataset_data["dataset"]
                }
                
                res = requests.post("http://localhost:8000/predict-churn", json=request_data)

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
                    
                    # Display analysis parameters used
                    if "given_date" in data and "num_weeks" in data:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Analysis Date:** {data['given_date']}")
                        with col2:
                            st.info(f"**Historical Weeks:** {data['num_weeks']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        actual_count = len(data["actual_churned_customers"]) if data["actual_churned_customers"] else 0
                        st.write(f"**Actual Churned Customers ({actual_count}):**")
                        if data["actual_churned_customers"]:
                            sorted_actual = sorted(data["actual_churned_customers"])
                            for customer_id in sorted_actual:
                                st.write(f"- {customer_id}")
                        else:
                            st.write("No actual churned customers in sample")
                    
                    with col2:
                        predicted_count = len(data["churned_customers"]) if data["churned_customers"] else 0
                        st.write(f"**Predicted Churned Customers ({predicted_count}):**")
                        if data["churned_customers"]:
                            sorted_predicted = sorted(data["churned_customers"])
                            for customer_id in sorted_predicted:
                                st.write(f"- {customer_id}")
                        else:
                            st.write("No customers predicted to churn")
                else:
                    st.error("❌ Failed to get predictions")
    else:
        st.info("👆 Click 'Load Data' to fetch the dataset first")

with tab2:
    st.title("📊 Prediction Logs")

    # Initialize session state for auto-refresh
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

    if st.button("🔄 Refresh Logs") or st.session_state.auto_refresh:
        st.session_state.auto_refresh = False
        st.rerun()

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
                    df = df.iloc[::-1].reset_index(drop=True)
                
                if 'Total Cost' in df.columns:
                    df['Total Cost'] = df['Total Cost'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "$0.000000")

                # Calculate accuracy for each row
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
                    
                    # Reorder columns
                    cols = list(df.columns)
                    false_positives_idx = cols.index('False Positives')
                    cols.insert(false_positives_idx + 1, cols.pop(cols.index('Accuracy')))
                    accuracy_idx = cols.index('Accuracy')
                    cols.insert(accuracy_idx + 1, cols.pop(cols.index('Recall')))
                    df = df[cols]

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Predictions", len(df))
                with col2:
                    true_positives = df['Matched'].sum()
                    true_negatives = df['Total Customers'].sum() - (df['Matched'].sum() + df['Mismatched'].sum() + df['False Positives'].sum())
                    total_predictions = df['Total Customers'].sum()
                    accuracy = (true_positives + true_negatives) / total_predictions * 100 if total_predictions > 0 else 0
                    st.metric("Accuracy", f"{accuracy:.1f}%", help="Overall correct predictions. Formula: (True Positives + True Negatives) / Total Predictions")
                with col3:
                    false_negatives = df['Mismatched'].sum()
                    recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
                    st.metric("Recall", f"{recall:.1f}%", help="Correct detection of actual positives. Formula: True Positives / (True Positives + False Negatives)")
                with col4:
                    total_cost = sum([float(str(x).replace('$', '')) for x in df['Total Cost'] if pd.notna(x)])
                    st.metric("Total Cost", f"${total_cost:.6f}")
                with col5:
                    total_tokens = df['Total Tokens'].sum() if 'Total Tokens' in df.columns else 0
                    st.metric("Total Tokens", f"{total_tokens:,}")

                # Create a copy for display with counts instead of ID lists
                df_display = df.copy()
                
                # Convert ID columns to counts for display
                id_columns = [
                    'Actual Churn Customer IDs',
                    'Actual Non Churn Customer IDs',
                    'Predicted Churn Customer IDs',
                    'Predicted Non Churn Customer IDs'
                ]

                for col in id_columns:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].apply(
                            lambda x: len([id.strip() for id in str(x).split(',') if id.strip() and len(id.strip()) == 36])
                            if pd.notna(x) and str(x).strip() and str(x).strip() != 'nan' else 0
                        )
                        new_col_name = col.replace(' Customer IDs', ' Count')
                        df_display = df_display.rename(columns={col: new_col_name})
                
                # Keep original df for CSV download
                df_csv = df.copy()
                
                # Move row_index to first column and rename it
                if 'row_index' in df.columns:
                     cols = ['row_index'] + [col for col in df.columns if col != 'row_index']
                     df = df[cols]
                     df = df.rename(columns={'row_index': 'Row Index'})
                     
                     if 'row_index' in df_display.columns:
                         cols_display = ['row_index'] + [col for col in df_display.columns if col != 'row_index']
                         df_display = df_display[cols_display]
                         df_display = df_display.rename(columns={'row_index': 'Row Index'})
                     
                     if 'row_index' in df_csv.columns:
                         cols_csv = ['row_index'] + [col for col in df_csv.columns if col != 'row_index']
                         df_csv = df_csv[cols_csv]
                         df_csv = df_csv.rename(columns={'row_index': 'Row Index'})

                # Create a simplified dataframe with only the requested columns
                st.subheader("📋 Prediction Log")
                
                # Define the columns to display in the specified order
                display_columns = [
                    'Row Index',
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
                
                # Create a new dataframe with calculated columns
                df_simple = df_display.copy()
                
                # Calculate Real Churn Ratio from Churn Distribution
                if 'Churn Distribution' in df_simple.columns:
                    df_simple['Real Churn Ratio'] = df_simple['Churn Distribution'].apply(
                        lambda x: f"{x.split(':')[0]}:{x.split(':')[1]}" if isinstance(x, str) and ':' in x else "0:0"
                    )
                
                # Calculate Predict Churn Ratio from predicted churn count vs predicted non-churn count
                if 'Predicted Churn Count' in df_simple.columns and 'Total Customers' in df_simple.columns:
                    df_simple['Predict Churn Ratio'] = df_simple.apply(
                        lambda row: f"{row['Predicted Churn Count']}:{row['Total Customers'] - row['Predicted Churn Count']}"
                        if row['Total Customers'] > 0 else "0:0", axis=1
                    )
                
                # Filter to only include available columns
                available_columns = [col for col in display_columns if col in df_simple.columns]
                df_simple = df_simple[available_columns].copy()
                
                # Display the dataframe without the duplicate index
                st.dataframe(df_simple, use_container_width=True, hide_index=True)
                
                # CSV download button under the dataframe
                csv_res = requests.get("http://localhost:8000/prediction-logs/csv")
                if csv_res.status_code == 200:
                    csv_data = csv_res.json()
                    if "csv_content" in csv_data:
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
                
                # Add delete functionality below the CSV button
                st.subheader("🗑️ Delete Records")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    selected_row = st.selectbox(
                        "Select row to delete:",
                        options=df_simple['Row Index'].tolist(),
                        format_func=lambda x: f"Row {x}"
                    )
                
                if st.button("Delete Selected Row", type="secondary"):
                    if selected_row is not None:
                        delete_response = requests.delete(f"http://localhost:8000/prediction-logs/{selected_row}")
                        if delete_response.status_code == 200:
                            st.success(f"Row {selected_row} deleted successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete row {selected_row}")
            else:
                st.info("No prediction logs found. Run some predictions first!")
        else:
            st.info("No prediction logs found. Run some predictions first!")
    else:
        st.error("❌ Failed to fetch prediction logs")
        st.info("Make sure the backend server is running on http://localhost:8000")
