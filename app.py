import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
from components.upload import render_upload_section
from components.dashboard import render_dashboard
from utils.data_cleaning import clean_data
from utils.visualization import plot_numeric_distributions, plot_categorical_distributions, plot_correlation_heatmap, plot_time_series
from utils.insights import generate_insights, generate_ai_insights
from models.anomaly_detection import detect_anomalies_zscore, detect_anomalies_isolation_forest
from models.predictive_analytics import run_time_series_forecast, run_regression_prediction
from utils.export import export_to_excel, export_to_pdf
import plotly.express as px

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 AI-Powered Data Analysis Dashboard")
    st.markdown("Upload your dataset to automatically generate insights, visualizations, and summary statistics.")

    # Sidebar for navigation or settings
    with st.sidebar:
        st.markdown("Configure your analysis preferences here.")

    # File upload component
    df_raw = render_upload_section()

    if df_raw is not None:
        st.success("File uploaded successfully!")
        
        # Clean data automatically
        with st.spinner("Cleaning and preprocessing data..."):
            df_clean, cleaning_summary = clean_data(df_raw)
            
        st.info(f"**Data Cleaning Summary**: {cleaning_summary}")
        
        # Store in session state so other components can access it
        st.session_state['df'] = df_clean
        
        # Interactive Filters
        st.sidebar.subheader("Filter Data")
        cat_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category']
        df_filtered = df_clean.copy()
        
        if cat_cols:
            filter_col = st.sidebar.selectbox("Filter by Category:", ["None"] + cat_cols, key='filter_col')
            if filter_col != "None":
                unique_vals = df_clean[filter_col].dropna().unique()
                selected_vals = st.sidebar.multiselect(f"Select {filter_col} values", unique_vals, default=unique_vals, key='filter_vals')
                if selected_vals:
                    df_filtered = df_clean[df_clean[filter_col].isin(selected_vals)]
                    st.sidebar.info(f"Filtered {len(df_clean) - len(df_filtered)} rows out.")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview", 
            "📈 Auto Vis", 
            "🔗 Correlation", 
            "💡 Insights",
            "🛠️ Custom",
            "🚨 Anomaly Detection",
            "🔮 Predictive Analytics"
        ])
        
        with tab1:
            render_dashboard(df_filtered)
            
        with tab2:
            st.header("Automated Visualizations")
            st.markdown("We automatically generated these charts based on your data types.")
            
            st.subheader("Numeric Distributions")
            plot_numeric_distributions(df_filtered)
            
            st.markdown("---")
            st.subheader("Categorical Distributions")
            plot_categorical_distributions(df_filtered)
            
            st.markdown("---")
            st.subheader("Time Series Analysis")
            plot_time_series(df_filtered)
            
        with tab3:
            st.header("Correlation Analysis")
            st.markdown("Discover relationships between numeric variables.")
            plot_correlation_heatmap(df_filtered)

            
        with tab4:
            st.header("💡 Business Insights")
            
            st.subheader("Statistical General Insights")
            insights = generate_insights(df_filtered)
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.warning("Could not generate significant statistical insights for this dataset.")
                
            st.markdown("---")
            st.subheader("🤖 AI-Generated Deep Insights")
            st.markdown("Use the power of LLMs to analyze your dataset's specific trends.")
            
            if st.button("Generate AI Insights"):
                with st.spinner("Analyzing data"):
                    ai_insights_list = generate_ai_insights(df_filtered)
                    if ai_insights_list:
                        for ai_insight in ai_insights_list:
                             st.success(ai_insight)
                            
        with tab5:
            st.header("🛠️ Custom Dashboard Builder")
            st.markdown("Build your own custom visualizations.")
            
            col_x, col_y, col_chart = st.columns(3)
            with col_x:
                x_axis = st.selectbox("X-Axis", df_filtered.columns, key='custom_x')
            with col_y:
                y_axis = st.selectbox("Y-Axis", df_filtered.columns, key='custom_y')
            with col_chart:
                chart_options = [
                    "Bar", "Line", "Scatter", "BoxPlot", 
                    "Histogram", "Pie", "Area", "Violin", 
                    "Density Heatmap", "Strip", "Funnel"
                ]
                chart_type = st.selectbox("Chart Type", chart_options, key='custom_chart')
                
            if st.button("Generate Custom Chart"):
                
                try:
                    if chart_type == "Bar":
                        # Aggregate if needed to prevent huge bar charts
                        agg_df = df_filtered.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f"Bar Chart: {y_axis} by {x_axis}")
                    elif chart_type == "Line":
                        fig = px.line(df_filtered.sort_values(by=x_axis), x=x_axis, y=y_axis, title=f"Line Chart: {y_axis} over {x_axis}")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df_filtered, x=x_axis, y=y_axis, title=f"Scatter Plot: {y_axis} vs {x_axis}")
                    elif chart_type == "BoxPlot":
                        fig = px.box(df_filtered, x=x_axis, y=y_axis, title=f"Box Plot: {y_axis} grouped by {x_axis}")
                    elif chart_type == "Histogram":
                        fig = px.histogram(df_filtered, x=x_axis, y=y_axis, title=f"Histogram: {y_axis} vs {x_axis}")
                    elif chart_type == "Pie":
                        # Pie chart needs names and values
                        agg_df = df_filtered.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.pie(agg_df, names=x_axis, values=y_axis, title=f"Pie Chart: {y_axis} by {x_axis}")
                    elif chart_type == "Area":
                        agg_df = df_filtered.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.area(agg_df, x=x_axis, y=y_axis, title=f"Area Chart: {y_axis} over {x_axis}")
                    elif chart_type == "Violin":
                        fig = px.violin(df_filtered, x=x_axis, y=y_axis, title=f"Violin Plot: {y_axis} grouped by {x_axis}")
                    elif chart_type == "Density Heatmap":
                        fig = px.density_heatmap(df_filtered, x=x_axis, y=y_axis, title=f"Density Heatmap: {y_axis} vs {x_axis}")
                    elif chart_type == "Strip":
                        fig = px.strip(df_filtered, x=x_axis, y=y_axis, title=f"Strip Plot: {y_axis} grouped by {x_axis}")
                    elif chart_type == "Funnel":
                        agg_df = df_filtered.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.funnel(agg_df, x=x_axis, y=y_axis, title=f"Funnel Chart: {y_axis} by {x_axis}")
                        
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate chart. Error: {str(e)}")

        with tab6:
            st.header("🚨 Anomaly Detection")
            st.markdown("Detect outliers and anomalies in your data using Machine Learning.")
            
            anomaly_method = st.radio("Select Method", ["Z-Score (Single Column)", "Isolation Forest (Multi-Column)"], horizontal=True)
            
            numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.warning("No numeric columns available for anomaly detection.")
            else:
                if anomaly_method.startswith("Z-Score"):
                     target_col = st.selectbox("Select Column to check:", numeric_columns, key='zscore_col')
                     threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
                     
                     if st.button("Detect Anomalies (Z-Score)"):
                          df_anom = df_filtered.copy()
                          df_anom['Is_Anomaly'] = detect_anomalies_zscore(df_anom, target_col, threshold)
                          
                          anom_count = df_anom['Is_Anomaly'].sum()
                          st.info(f"Detected {anom_count} anomalies out of {len(df_anom)} rows.")
                          
                          if anom_count > 0:
                              fig = px.scatter(df_anom, x=df_anom.index, y=target_col, color='Is_Anomaly', 
                                               color_discrete_map={True: 'red', False: 'blue'}, title=f"Anomalies in {target_col}")
                              st.plotly_chart(fig, use_container_width=True)
                              st.dataframe(df_anom[df_anom['Is_Anomaly']])
                              
                elif anomaly_method.startswith("Isolation"):
                     selected_cols = st.multiselect("Select Columns for Isolation Forest:", numeric_columns, default=numeric_columns[:2] if len(numeric_columns)>=2 else numeric_columns, key='if_cols')
                     contamination = st.slider("Expected Anomaly Proportion (Contamination)", 0.01, 0.20, 0.05, 0.01)
                     
                     if st.button("Detect Anomalies (Isolation Forest)"):
                          if len(selected_cols) == 0:
                              st.warning("Please select at least one column.")
                          else:
                              df_anom = df_filtered.copy()
                              df_anom['Is_Anomaly'] = detect_anomalies_isolation_forest(df_anom, selected_cols, contamination)
                              anom_count = df_anom['Is_Anomaly'].sum()
                              st.info(f"Detected {anom_count} anomalies out of {len(df_anom)} rows.")
                              
                              if anom_count > 0 and len(selected_cols) >= 2:
                                  fig = px.scatter(df_anom, x=selected_cols[0], y=selected_cols[1], color='Is_Anomaly',
                                                   color_discrete_map={True: 'red', False: 'blue'}, title="Isolation Forest Anomalies")
                                  st.plotly_chart(fig, use_container_width=True)
                                  st.dataframe(df_anom[df_anom['Is_Anomaly']])

        with tab7:
            st.header("🔮 Predictive Analytics")
            st.markdown("Forecast future trends or predict target variables.")
            
            pred_method = st.radio("Select Prediction Type", ["Time Series Forecasting (Prophet)", "Regression (Random Forest)"], horizontal=True)
            
            if pred_method.startswith("Time Series"):
                date_cols = df_filtered.select_dtypes(include=['datetime', 'object']).columns.tolist()
                num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                
                col_d, col_t = st.columns(2)
                with col_d:
                     ts_date = st.selectbox("Select Date/Time Column", date_cols, key='ts_date')
                with col_t:
                     ts_target = st.selectbox("Select Target to Forecast", num_cols, key='ts_target')
                     
                periods = st.slider("Forecast Periods Ahead", 7, 365, 30)
                
                if st.button("Run Forecast"):
                     with st.spinner("Training Prophet model..."):
                         fig, forecast = run_time_series_forecast(df_filtered.copy(), ts_date, ts_target, periods)
                         if fig:
                              st.plotly_chart(fig, use_container_width=True)
                              with st.expander("View Forecast Data"):
                                  display_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                                  display_df = display_df.rename(columns={
                                      'ds': 'Date',
                                      'yhat': 'Predicted_Value',
                                      'yhat_lower': 'Lower_Bound_Estimate',
                                      'yhat_upper': 'Upper_Bound_Estimate'
                                  })
                                  st.dataframe(display_df)
                         else:
                              st.error(forecast) # Error message returned
                              
            elif pred_method.startswith("Regression"):
                 num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                 
                 if len(num_cols) < 2:
                     st.warning("Need at least 2 numeric columns for regression.")
                 else:
                     target_reg = st.selectbox("Select Target Variable (Y)", num_cols, key='reg_y')
                     features_reg = st.multiselect("Select Feature Variables (X)", [c for c in num_cols if c != target_reg], default=[c for c in num_cols if c != target_reg][:3], key='reg_x')
                     
                     if st.button("Run Regression"):
                          if not features_reg:
                              st.error("Please select at least one feature.")
                          else:
                              with st.spinner("Training Random Forest..."):
                                  fig, importances, msg = run_regression_prediction(df_filtered.copy(), target_reg, features_reg)
                                  if fig:
                                       st.plotly_chart(fig, use_container_width=True)
                                       st.subheader("Feature Importances")
                                       st.bar_chart(importances.set_index('Feature'))
                                  else:
                                       st.error(msg)
                                       
        # Add Export Section inside sidebar at the bottom
        st.sidebar.markdown("---")
        st.sidebar.header("📥 Export Reports")
        st.sidebar.markdown("Download your processed data and reports.")
        
        # Generate files for download
        try:
             excel_data = export_to_excel(df_filtered)
             st.sidebar.download_button(
                 label="📥 Download Cleaned Data (Excel)",
                 data=excel_data,
                 file_name="cleaned_dataset.xlsx",
                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
             )
        except Exception as e:
             st.sidebar.error(f"Excel Export Error: {e}")
             
        try:
             # Gather insights for the PDF
             st_stat_insights = generate_insights(df_filtered)
             
             if st.sidebar.button("Generate & Download PDF Report"):
                  with st.spinner("Generating PDF..."):
                       # Generate AI insights right here for the PDF if key exists, otherwise pass empty
                       pdf_ai_insights = generate_ai_insights(df_filtered)
                       pdf_bytes = export_to_pdf(cleaning_summary, st_stat_insights, pdf_ai_insights)
                       st.sidebar.download_button(
                           label="📄 Click to Download PDF",
                           data=pdf_bytes,
                           file_name="ai_analysis_report.pdf",
                           mime="application/pdf"
                       )
        except Exception as e:
             st.sidebar.error(f"PDF Export Error: {e}")

if __name__ == "__main__":
    main()
