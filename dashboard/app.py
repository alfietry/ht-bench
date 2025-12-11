"""
Streamlit dashboard for LLM hypothesis testing benchmark visualization
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import numpy as np
from pathlib import Path
import sys
import scipy.stats as stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

st.set_page_config(
    page_title="LLM Hypothesis Testing Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Switch to a light, leaderboard-style aesthetic
pio.templates.default = "plotly_white"

st.markdown("""
<style>
    :root {
        color-scheme: light;
    }
    body {
        background: #f5f5f8;
        color: #111827;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at top, #ffffff 0, #f5f5f8 45%, #ececf2 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    h1, h2, h3, h4 {
        color: #111827;
        letter-spacing: 0.02em;
        font-weight: 650;
    }
    /* Header bar with title */
    header[data-testid="stHeader"] {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        padding: 0.4rem 1.5rem 0.2rem 1.5rem;
    }
    header[data-testid="stHeader"]::after {
        content: "LLM Hypothesis Testing Benchmark Leaderboard";
        color: #111827;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    /* Metric cards */
    .stMetric {
        background: #ffffff !important;
        padding: 14px 16px !important;
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04) !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        border-bottom: 1px solid #e5e7eb;
        position: sticky;
        top: 3.1rem; /* directly under Streamlit header / Deploy bar */
        z-index: 50;
        background: #f5f5f8;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        margin-top: -0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #e5f2ff;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        border: 1px solid #bfdbfe;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1d4ed8;
        border-color: #1d4ed8;
        color: #f9fafb;
        font-weight: 600;
    }
    /* Selects and sidebar */
    div[data-baseweb="select"] > div {
        border-color: #d1d5db !important;
        background: #ffffff !important;
        color: #111827 !important;
    }
    .stSidebar, .stSidebarContent {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }
    .stMultiSelect, .stSelectbox label {
        color: #374151;
        font-weight: 500;
    }
    /* Sidebar filter multiselect chips (models, prompts, test types) */
    section[data-testid="stSidebar"] div[data-baseweb="tag"] {
        background-color: #dcfce7 !important;
        color: #166534 !important;
        border-radius: 999px !important;
        border: 1px solid #bbf7d0 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="tag"] svg {
        fill: #166534 !important;
    }
    /* Dataframes */
    .stDataFrame, .stTable {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #ffffff;
    }
    /* Buttons (used for filters/reload) */
    .stButton button, .stDownloadButton button {
        background: #047857;
        color: #ecfdf5;
        border-radius: 999px;
        border: 1px solid #059669;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.4rem 1.2rem;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background: #065f46;
        border-color: #047857;
    }
    /* Alerts */
    .stAlert {
        background: #eff6ff !important;
        border-left: 4px solid #2563eb !important;
        color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results(results_dir: Path = config.RESULTS_DIR):
    """Load all result JSON files"""
    results = []
    if not results_dir.exists():
        return []
        
    for file in results_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    return results

@st.cache_data
def prepare_dataframe(results: list) -> pd.DataFrame:
    """Convert results to DataFrame with enhanced metrics"""
    rows = []
    
    for result in results:
        eval_data = result.get('evaluation', {})
        ground_truth = result.get('ground_truth', {})
        parsed = result.get('parsed_results', {})
        
        # Calculate errors
        p_val_pred = parsed.get('p_value')
        p_val_true = ground_truth.get('p_value')
        p_val_error = abs(p_val_pred - p_val_true) if (p_val_pred is not None and p_val_true is not None) else None
        
        row = {
            'timestamp': result.get('timestamp', ''),
            'model': result.get('model', 'unknown'),
            'prompt_type': result.get('prompt_type', 'unknown'),
            'test_type': result.get('input_data', {}).get('test_type', 'unknown'),
            'overall_accuracy': eval_data.get('overall_accuracy', 0),
            'test_method_accuracy': eval_data.get('test_method', 0),
            'decision_accuracy': 1.0 if eval_data.get('decision', {}).get('correct', False) else 0.0,
            'p_value_accuracy': 1.0 if eval_data.get('p_value', {}).get('within_tolerance', False) else 0.0,
            'reasoning_quality': eval_data.get('reasoning_quality', {}).get('percentage', 0) / 100,
            'has_hallucinations': eval_data.get('hallucinations', {}).get('has_hallucinations', False),
            'completeness': sum(eval_data.get('completeness', {}).values()) / 5.0 if eval_data.get('completeness') else 0,
            'predicted_decision': parsed.get('decision'),
            'true_decision': ground_truth.get('decision'),
            'predicted_p_value': p_val_pred,
            'true_p_value': p_val_true,
            'p_value_error': p_val_error,
            'latency_seconds': result.get('latency_seconds'),
            'prompt_text': result.get('prompt') or result.get('input_prompt', ''),
            'response_text': result.get('raw_response') or result.get('response', '')
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and CI for a series"""
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return np.mean(a), 0.0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def create_leaderboard(df: pd.DataFrame):
    """Create a research-grade leaderboard with CIs"""
    models = df['model'].unique()
    leaderboard_data = []
    
    for model in models:
        model_df = df[df['model'] == model]
        acc_mean, acc_ci = calculate_confidence_interval(model_df['overall_accuracy'])
        latency_series = model_df['latency_seconds'].dropna()
        latency_mean = latency_series.mean() if not latency_series.empty else None
        
        leaderboard_data.append({
            'Model': model,
            'Samples': len(model_df),
            'Accuracy': acc_mean,
            '95% CI': acc_ci,
            'Decision Acc': model_df['decision_accuracy'].mean(),
            'Reasoning Score': model_df['reasoning_quality'].mean(),
            'Hallucination Rate': model_df['has_hallucinations'].mean(),
            'Latency (s)': latency_mean
        })
    
    lb_df = pd.DataFrame(leaderboard_data).sort_values('Accuracy', ascending=False)
    
    # Format for display
    display_df = lb_df.copy()
    display_df['Accuracy'] = display_df.apply(lambda x: f"{x['Accuracy']:.1%} ± {x['95% CI']:.1%}", axis=1)
    display_df['Decision Acc'] = display_df['Decision Acc'].apply(lambda x: f"{x:.1%}")
    display_df['Reasoning Score'] = display_df['Reasoning Score'].apply(lambda x: f"{x:.2f}")
    display_df['Hallucination Rate'] = display_df['Hallucination Rate'].apply(lambda x: f"{x:.1%}")
    display_df['Latency (s)'] = display_df['Latency (s)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    
    return display_df[['Model', 'Samples', 'Accuracy', 'Decision Acc', 'Reasoning Score', 'Hallucination Rate', 'Latency (s)']]

def create_radar_chart(df: pd.DataFrame, models: list):
    """Create radar chart comparing models across metrics"""
    metrics = ['test_method_accuracy', 'decision_accuracy', 'p_value_accuracy', 
               'reasoning_quality', 'completeness']
    metric_labels = ['Test Method', 'Decision', 'P-Value', 'Reasoning', 'Completeness']
    
    fig = go.Figure()
    
    for model in models:
        model_data = df[df['model'] == model]
        if len(model_data) == 0:
            continue
        
        values = [model_data[m].mean() for m in metrics]
        # Close the loop
        values.append(values[0])
        theta = metric_labels + [metric_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Capabilities Fingerprint",
        height=500
    )
    
    return fig


def group_models_for_radars(models: list) -> dict:
    """Bucket models into radar families"""
    categories = {
        "GPT Models": [],
        "Grok Models": [],
        "Claude Models": [],
        "Gemini / DeepSeek / Others": []
    }

    for model in models:
        name = str(model).lower()
        if "gpt" in name:
            categories["GPT Models"].append(model)
        elif "grok" in name:
            categories["Grok Models"].append(model)
        elif "claude" in name:
            categories["Claude Models"].append(model)
        else:
            categories["Gemini / DeepSeek / Others"].append(model)

    return categories

def create_p_value_scatter(df: pd.DataFrame):
    """Create scatter plot of True vs Predicted P-values"""
    fig = px.scatter(
        df, 
        x='true_p_value', 
        y='predicted_p_value', 
        color='model',
        hover_data=['test_type', 'prompt_type'],
        title="P-Value Correlation: Ground Truth vs Predicted",
        labels={'true_p_value': 'Ground Truth P-Value', 'predicted_p_value': 'Predicted P-Value'}
    )
    
    # Add y=x line
    fig.add_shape(
        type="line", line=dict(dash='dash', color='gray'),
        x0=0, y0=0, x1=1, y1=1
    )
    
    fig.update_layout(height=600)
    return fig

def create_heatmap(df: pd.DataFrame):
    """Create heatmap of performance across test types and models"""
    pivot = df.pivot_table(
        values='overall_accuracy',
        index='model',
        columns='test_type',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Test Type", y="Model", color="Accuracy"),
        color_continuous_scale=[
            [0.0, "#0f1c2e"],
            [0.5, "#513b8a"],
            [1.0, "#f5d76e"]
        ],
        text_auto=True,
        aspect="auto",
        title="Model Performance Heatmap by Test Type"
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    st.markdown(" ")

    # Sidebar Controls
    st.sidebar.header("Configuration")
    if st.sidebar.button("Reload results"):
        load_results.clear()
        prepare_dataframe.clear()
        st.rerun()

    results = load_results()
    if not results:
        st.warning("No results found. Please run the benchmark first.")
        return

    df = prepare_dataframe(results)

    # Drop models whose overall accuracy never rises above zero to avoid cluttering visuals
    model_accuracy = df.groupby('model')['overall_accuracy'].mean()
    viable_models = model_accuracy[model_accuracy > 0].index.tolist()
    if not viable_models:
        st.warning("All tracked models currently have 0 accuracy; rerun the benchmark to collect valid data.")
        return
    df = df[df['model'].isin(viable_models)].copy()

    if df.empty:
        st.warning("No valid benchmark rows were found. Run the benchmark to populate the dashboard.")
        return
    
    # Sidebar Filters
    st.sidebar.subheader("Filters")
    
    all_models = df['model'].unique()
    selected_models = st.sidebar.multiselect("Models", all_models, default=all_models)
    
    all_prompts = df['prompt_type'].unique()
    selected_prompts = st.sidebar.multiselect("Prompt Strategies", all_prompts, default=all_prompts)
    
    all_tests = df['test_type'].unique()
    selected_tests = st.sidebar.multiselect("Test Types", all_tests, default=all_tests)
    
    # Apply filters
    filtered_df = df[
        (df['model'].isin(selected_models)) &
        (df['prompt_type'].isin(selected_prompts)) &
        (df['test_type'].isin(selected_tests))
    ].copy()
    
    if filtered_df.empty:
        st.error("No data matches the selected filters.")
        return

    display_df = filtered_df[[
        'timestamp', 'model', 'prompt_type', 'test_type',
        'overall_accuracy', 'decision_accuracy', 'p_value_accuracy',
        'reasoning_quality', 'has_hallucinations', 'latency_seconds'
    ]].sort_values('overall_accuracy', ascending=False)
    display_df['latency_seconds'] = pd.to_numeric(display_df['latency_seconds'], errors='coerce').round(2)

    # Layout closer to leaderboard style:
    tab_overview, tab_detailed, tab_stats, tab_qual = st.tabs([
        "Leaderboard", 
        "Detailed Analysis", 
        "Statistical Deep Dive",
        "Qualitative Inspector"
    ])

    with tab_overview:
        # Summary cards centered above leaderboard
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Samples", len(filtered_df))
        col2.metric("Avg Accuracy", f"{filtered_df['overall_accuracy'].mean():.1%}")
        col3.metric("Avg Reasoning", f"{filtered_df['reasoning_quality'].mean():.2f}")
        col4.metric("Hallucination Rate", f"{filtered_df['has_hallucinations'].mean():.1%}")
        latency_series = filtered_df['latency_seconds'].dropna()
        latency_display = f"{latency_series.mean():.2f}s" if not latency_series.empty else "N/A"
        col5.metric("Avg Latency", latency_display)

        st.markdown("### Global Model Leaderboard")
        leaderboard = create_leaderboard(filtered_df)
        st.dataframe(leaderboard, width='stretch')

        st.markdown("### Capability Profiles by Model Family")
        available_models = [m for m in selected_models if m in filtered_df['model'].unique()]
        radar_groups = group_models_for_radars(available_models)
        ordered_groups = [
            ("GPT Models", radar_groups.get("GPT Models", [])),
            ("Grok Models", radar_groups.get("Grok Models", [])),
            ("Claude Models", radar_groups.get("Claude Models", [])),
            ("Gemini / DeepSeek / Others", radar_groups.get("Gemini / DeepSeek / Others", []))
        ]

        for row_groups in [ordered_groups[:2], ordered_groups[2:]]:
            cols = st.columns(2)
            for col, (title, models_subset) in zip(cols, row_groups):
                with col:
                    st.markdown(f"#### {title}")
                    if models_subset:
                        radar_fig = create_radar_chart(filtered_df, models_subset)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.info("No models selected in this family.")

    with tab_detailed:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance by Test Type")
            heatmap = create_heatmap(filtered_df)
            st.plotly_chart(heatmap, width='stretch')
            
        with col2:
            st.markdown("### Prompt Strategy Impact")
            prompt_perf = filtered_df.groupby(['model', 'prompt_type'])['overall_accuracy'].mean().reset_index()
            fig_prompt = px.bar(prompt_perf, x='prompt_type', y='overall_accuracy', color='model', barmode='group',
                              title="Accuracy by Prompt Strategy")
            st.plotly_chart(fig_prompt, width='stretch')
            
        st.markdown("### Reasoning Quality Distribution")
        fig_box = px.box(filtered_df, x='model', y='reasoning_quality', color='model', 
                        title="Distribution of Reasoning Quality Scores")
        st.plotly_chart(fig_box, width='stretch')

        st.markdown("### Latency by Model")
        latency_df = filtered_df.dropna(subset=['latency_seconds'])
        if not latency_df.empty:
            latency_stats = latency_df.groupby('model')['latency_seconds'].mean().reset_index()
            fig_latency = px.bar(
                latency_stats,
                x='model',
                y='latency_seconds',
                color='model',
                title="Average Response Latency",
                labels={'model': 'Model', 'latency_seconds': 'Latency (s)'}
            )
            fig_latency.update_layout(showlegend=False)
            st.plotly_chart(fig_latency, width='stretch')
        else:
            st.info("Latency data is not available for the selected filters.")

        st.markdown("### Detailed Results")
        st.dataframe(display_df, width='stretch', height=400)

    with tab_stats:
        st.markdown("### P-Value Estimation Accuracy")
        st.info("This plot compares the ground truth p-value (x-axis) with the model's predicted p-value (y-axis). Points closer to the diagonal dashed line indicate better performance.")
        
        # Filter out None values for p-values
        p_val_df = filtered_df.dropna(subset=['predicted_p_value', 'true_p_value'])
        if not p_val_df.empty:
            scatter = create_p_value_scatter(p_val_df)
            st.plotly_chart(scatter, width='stretch')
        else:
            st.warning("No p-value data available for the selected filters.")
            
        st.markdown("### Error Distribution (MAE)")
        error_df = filtered_df.groupby('model')[['p_value_error']].mean().reset_index()
        fig_err = px.bar(error_df, x='model', y='p_value_error', title="Mean Absolute Error in P-Value Estimation")
        st.plotly_chart(fig_err, width='stretch')

    with tab_qual:
        st.markdown("### Individual Response Inspector")
        st.markdown("Select a row to view the full prompt and model response.")
        
        # Interactive dataframe
        # Using a simpler selection mechanism for compatibility
        
        # Create a display column for the selectbox
        filtered_df['display_label'] = filtered_df.apply(
            lambda x: f"{x['model']} | {x['test_type']} | {x['prompt_type']} (Acc: {x['overall_accuracy']})", axis=1
        )
        
        selected_row_label = st.selectbox("Select Sample", filtered_df['display_label'].tolist())
        
        if selected_row_label:
            row = filtered_df[filtered_df['display_label'] == selected_row_label].iloc[0]
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Prompt")
                st.text_area("Input", row['prompt_text'], height=400, disabled=True)
            with c2:
                st.markdown(f"#### {row['model']} Response")
                st.text_area("Output", row['response_text'], height=400, disabled=True)
                
            st.markdown("#### Ground Truth vs Prediction")
            st.json({
                "True Decision": row['true_decision'],
                "Predicted Decision": row['predicted_decision'],
                "True P-Value": row['true_p_value'],
                "Predicted P-Value": row['predicted_p_value'],
                "Accuracy Score": row['overall_accuracy'],
                "Latency (s)": row['latency_seconds']
            })

    # Footer
    st.markdown("---")
    st.markdown(f"*Generated by LLM Hypothesis Testing Benchmark | Total Records: {len(df)}*")
    
    # Download
    csv = display_df.to_csv(index=False)
    st.sidebar.download_button(
        "Download Full Results CSV",
        csv,
        "benchmark_results.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()
