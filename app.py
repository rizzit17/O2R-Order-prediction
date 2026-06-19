import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="O2R : Order Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PATHS — update BASE_DIR to your project folder
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'xgboost_order_model.pkl')
PARQUET_PATH = os.path.join(BASE_DIR, 'processed', 'retailer_day_features.parquet')
RAW_CSV_PATH = os.path.join(BASE_DIR, 'data', "Jan - May '26 Data.csv")
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4f8bf9;
    }
    div[data-testid="stMetric"] {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2b2b3d;
        border-left: 5px solid #4f8bf9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #4f8bf9;
        color: #4f8bf9;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA (cached so it doesn't reload every interaction)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_feature_data():
    df = pd.read_parquet(PARQUET_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_raw_data():
    df = pd.read_csv(RAW_CSV_PATH)
    df['createdAt'] = pd.to_datetime(df['createdAt'], dayfirst=True)
    confirmed = df[df['orderStatus'].isin(['Delivered', 'PartiallyDelivered'])]
    orders = confirmed.drop_duplicates(subset='orderNumber')[[
        'orderNumber', 'customerId', 'createdAt',
        'hubName', 'shopType', 'retailerType', 'orderSource'
    ]].copy()
    return orders

@st.cache_data
def get_retailer_profiles(orders):
    profile = orders.groupby('customerId').agg(
        hub           = ('hubName',      lambda x: x.mode()[0]),
        shop_type     = ('shopType',     lambda x: x.mode()[0]),
        retailer_type = ('retailerType', lambda x: x.mode()[0] if x.notna().any() else 'Unknown'),
        total_orders  = ('orderNumber',  'count'),
        last_order    = ('createdAt',    'max'),
        primary_source= ('orderSource',  lambda x: x.mode()[0]),
        app_ratio     = ('orderSource',  lambda x: round((x == 'App').mean() * 100, 1))
    ).reset_index()
    return profile

# ─────────────────────────────────────────────
# GENERATE PREDICTIONS FOR A DATE
# ─────────────────────────────────────────────
def generate_predictions(grid, model_data, target_date, threshold):
    FEATURE_COLS = model_data['feature_cols']
    model        = model_data['model']

    day_data = grid[grid['date'] == pd.Timestamp(target_date)].copy()
    if len(day_data) == 0:
        return None

    X = np.nan_to_num(day_data[FEATURE_COLS].values, nan=0.0)
    day_data = day_data.copy()
    day_data['order_probability'] = model.predict_proba(X)[:, 1]
    day_data['will_order']        = (day_data['order_probability'] >= threshold).astype(int)
    return day_data

def apply_chart_style(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family='Inter',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    try:
        st.image(
            os.path.join(BASE_DIR, "O2R-logo.jpg"),
            width=160
        )
    except:
        st.title("O2R")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Call Priority List",
            "Model Performance",
            "Next Order Schedule",
            "Retailer Deep Dive"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Call Threshold")
    
    if "slider_in" not in st.session_state:
        st.session_state.slider_in = 0.40
    if "num_in" not in st.session_state:
        st.session_state.num_in = 0.40

    def sync_slider():
        st.session_state.slider_in = st.session_state.num_in
    def sync_num():
        st.session_state.num_in = st.session_state.slider_in

    st.slider(
        "Minimum probability to call",
        min_value=0.10, max_value=0.90,
        step=0.05,
        key="slider_in",
        on_change=sync_num,
        help="Retailers above this probability will be flagged for calling"
    )
    st.number_input(
        "Manual Input",
        min_value=0.10, max_value=0.90,
        step=0.01,
        key="num_in",
        on_change=sync_slider,
        label_visibility="collapsed"
    )
    threshold = st.session_state.slider_in
    st.caption(f"Currently calling retailers with >{int(threshold*100)}% order probability")

    st.markdown("---")
    st.markdown("### Prediction Date")
    pred_date = st.date_input(
        "Select date",
        value=datetime(2026, 5, 31),
        min_value=datetime(2026, 1, 1),
        max_value=datetime(2026, 5, 31)
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="background-color: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 4px solid #4f8bf9; margin-bottom: 20px;">
        <p style="margin: 0; color: #aaa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Active Date</p>
        <p style="margin: 4px 0 12px 0; font-weight: 600; font-size: 1.1rem; color: white;">{pred_date.strftime('%d %b %Y')}</p>
        <p style="margin: 0; color: #aaa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Current Threshold</p>
        <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.1rem; color: white;">{int(threshold*100)}%</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Built for DS Group Internship 2026")

# ─────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────
try:
    with st.spinner("Loading model and data..."):
        model_data = load_model()
        grid       = load_feature_data()
        orders     = load_raw_data()
        profiles   = get_retailer_profiles(orders)

    predictions = generate_predictions(grid, model_data, pred_date, threshold)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have run all 5 notebooks first and the processed/ and models/ folders exist.")
    st.stop()

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.title("O2R : Retailer Order Prediction")
    st.markdown("**O2R Call Centre Optimization | Jan–May 2026**")
    st.markdown("---")

    if predictions is not None:
        total      = len(predictions)
        to_call    = predictions['will_order'].sum()
        to_skip    = total - to_call
        reduction  = (1 - to_call / total) * 100
        avg_prob   = predictions['order_probability'].mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Retailers", f"{total:,}")
        with col2:
            st.metric("📞 Call Today", f"{int(to_call):,}",
                      delta=f"-{int(to_skip):,} skipped", delta_color="normal")
        with col3:
            st.metric("Call Reduction", f"{reduction:.1f}%",
                      help="vs calling everyone")
        with col4:
            st.metric("Avg Order Probability", f"{avg_prob:.1f}%")

        st.markdown("---")

        # Business impact
        st.subheader("Estimated Business Impact")
        current_calls   = total
        cost_per_min    = 8
        avg_call_min    = 2
        current_cost    = current_calls * cost_per_min * avg_call_min
        
        new_calls  = int(to_call)
        new_cost   = new_calls * cost_per_min * avg_call_min
        savings    = current_cost - new_cost

        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div style="flex: 1; background-color: rgba(248, 113, 113, 0.05); border: 1px solid rgba(248, 113, 113, 0.2); border-radius: 12px; padding: 24px;">
                <h4 style="color: #f87171; margin-top: 0; font-weight: 600;">Before (Current Process)</h4>
                <div style="font-size: 1.1rem; color: #ddd;">
                    <p style="margin: 8px 0;">Daily calls to make: <strong style="color: white; font-size: 1.2rem;">{current_calls:,}</strong></p>
                    <p style="margin: 8px 0;">AI voice cost: ₹{cost_per_min}/min × {avg_call_min} min avg</p>
                </div>
                <hr style="border-color: rgba(248, 113, 113, 0.2); margin: 16px 0;">
                <h3 style="color: #f87171; margin-bottom: 0; font-size: 1.5rem;">Daily Cost: ₹{current_cost:,}</h3>
            </div>
            <div style="flex: 1; background-color: rgba(74, 222, 128, 0.05); border: 1px solid rgba(74, 222, 128, 0.2); border-radius: 12px; padding: 24px;">
                <h4 style="color: #4ade80; margin-top: 0; font-weight: 600;">After (With Model)</h4>
                <div style="font-size: 1.1rem; color: #ddd;">
                    <p style="margin: 8px 0;">Daily calls to make: <strong style="color: white; font-size: 1.2rem;">{new_calls:,}</strong></p>
                    <p style="margin: 8px 0;">AI voice cost: ₹{cost_per_min}/min × {avg_call_min} min avg</p>
                </div>
                <hr style="border-color: rgba(74, 222, 128, 0.2); margin: 16px 0;">
                <h3 style="color: #4ade80; margin-bottom: 0; font-size: 1.5rem;">Daily Cost: ₹{new_cost:,}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.success(f"**Daily savings:** ₹{savings:,} &nbsp;|&nbsp; **Monthly run-rate:** ₹{savings*26:,}")

        st.markdown("---")

        # Probability distribution
        st.subheader(f"Order Probability Distribution — {pred_date}")
        fig = px.histogram(
            predictions,
            x='order_probability',
            nbins=50,
            color_discrete_sequence=['#4f8bf9'],
            labels={'order_probability': 'Predicted Order Probability'}
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold ({int(threshold*100)}%)",
                      annotation_position="top right")
        fig = apply_chart_style(fig)
        fig.update_layout(
            xaxis_title="Order Probability",
            yaxis_title="Number of Retailers",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hub breakdown
        st.subheader("Call Distribution by Hub")
        merged = predictions.merge(profiles[['customerId','hub']], on='customerId', how='left')
        hub_summary = merged.groupby('hub').agg(
            total=('customerId','count'),
            to_call=('will_order','sum')
        ).reset_index()
        hub_summary['to_skip'] = hub_summary['total'] - hub_summary['to_call']
        hub_summary = hub_summary.sort_values('to_call', ascending=False)

        fig2 = go.Figure(data=[
            go.Bar(name='Call', x=hub_summary['hub'], y=hub_summary['to_call'],
                   marker_color='#4f8bf9'),
            go.Bar(name='Skip', x=hub_summary['hub'], y=hub_summary['to_skip'],
                   marker_color='#374151')
        ])
        fig2 = apply_chart_style(fig2)
        fig2.update_layout(
            barmode='stack',
            xaxis_tickangle=-35,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: CALL PRIORITY LIST
# ─────────────────────────────────────────────
elif page == "Call Priority List":
    st.title(f"Call Priority List — {pred_date}")
    st.markdown("Retailers ranked by order probability. Call from top to bottom.")
    st.markdown("---")

    if predictions is not None:
        # Merge with profiles
        merged = predictions.merge(profiles, on='customerId', how='left')
        merged = merged.sort_values('order_probability', ascending=False).reset_index(drop=True)
        merged.index += 1
        merged.index.name = 'Rank'

        # Filters
        with st.expander("Filter Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                hub_filter = st.multiselect("Filter by Hub",
                                             options=sorted(merged['hub'].dropna().unique()),
                                             default=[])
            with col2:
                shop_filter = st.multiselect("Filter by Shop Type",
                                              options=sorted(merged['shop_type'].dropna().unique()),
                                              default=[])
            with col3:
                show_only = st.radio("Show", ["All Retailers", "Call List Only", "Skip List Only"],
                                      horizontal=True)

        filtered = merged.copy()
        if hub_filter:
            filtered = filtered[filtered['hub'].isin(hub_filter)]
        if shop_filter:
            filtered = filtered[filtered['shop_type'].isin(shop_filter)]
        if show_only == "Call List Only":
            filtered = filtered[filtered['will_order'] == 1]
        elif show_only == "Skip List Only":
            filtered = filtered[filtered['will_order'] == 0]

        st.markdown("---")
        
        calls = (filtered['will_order'] == 1).sum()
        skips = (filtered['will_order'] == 0).sum()
        st.markdown(f"""
        <div style="display: flex; gap: 15px; margin-bottom: 15px; align-items: center;">
            <span style="color: #aaa;">Showing <strong>{len(filtered):,}</strong> retailers</span>
            <div style="background-color: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3); padding: 6px 12px; border-radius: 6px;">
                <span style="color: #4ade80; font-weight: 600; font-size: 0.9rem;">📞 CALL: {calls:,}</span>
            </div>
            <div style="background-color: rgba(156, 163, 175, 0.1); border: 1px solid rgba(156, 163, 175, 0.3); padding: 6px 12px; border-radius: 6px;">
                <span style="color: #9ca3af; font-weight: 600; font-size: 0.9rem;">SKIP: {skips:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Format display table
        display = filtered[[
            'customerId', 'hub', 'shop_type', 'retailer_type',
            'order_probability', 'will_order',
            'days_since_last_order', 'avg_gap_between_orders',
            'orders_last_7_days', 'total_orders', 'primary_source'
        ]].copy()

        display['order_probability'] = (display['order_probability'] * 100).round(1)
        display['avg_gap_between_orders'] = display['avg_gap_between_orders'].round(1)
        display['Action'] = display['will_order'].map({1: '📞 CALL', 0: 'SKIP'})

        display = display.rename(columns={
            'customerId':             'Retailer ID',
            'hub':                    'Hub',
            'shop_type':              'Shop Type',
            'retailer_type':          'Retailer Type',
            'order_probability':      'Probability %',
            'days_since_last_order':  'Days Since Last Order',
            'avg_gap_between_orders': 'Avg Gap (days)',
            'orders_last_7_days':     'Orders (7d)',
            'total_orders':           'Total Orders',
            'primary_source':         'Preferred Channel'
        }).drop(columns=['will_order'])

        # Color code rows
        def color_rows(row):
            if row['Action'] == '📞 CALL':
                return ['background-color: #0d2b1a'] * len(row)
            return ['background-color: #1a0d0d'] * len(row)

        st.dataframe(
            display.style.apply(color_rows, axis=1),
            column_config={
                "Probability %": st.column_config.ProgressColumn(
                    "Probability %",
                    help="Predicted probability of ordering",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True,
            height=600
        )

        # Download button
        csv = display.to_csv(index=True).encode('utf-8-sig')
        st.download_button(
            label="Download Call List as CSV",
            data=csv,
            file_name=f"call_priority_{pred_date}.csv",
            mime='text/csv'
        )

# ─────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("XGBoost trained on Jan–Apr 2026, evaluated on May 2026.")
    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance")
    FEATURE_COLS = model_data['feature_cols']
    model        = model_data['model']
    importances  = model.feature_importances_

    feat_df = pd.DataFrame({
        'Feature':    FEATURE_COLS,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig = px.bar(
        feat_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        title='XGBoost Feature Importance'
    )
    fig = apply_chart_style(fig)
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Threshold analysis table (recomputed from May test set)
    st.subheader("Threshold Analysis — Choose Your Operating Point")
    st.markdown("Adjust the threshold in the sidebar to see the tradeoff between call volume and order capture.")

    if predictions is not None:
        # Build threshold table for the selected date
        probs = predictions['order_probability'].values
        actuals = predictions['ordered_today'].values if 'ordered_today' in predictions.columns else None

        if actuals is not None:
            rows = []
            total_r = len(probs)
            total_orders = actuals.sum()

            for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                preds_t        = (probs >= t).astype(int)
                calls          = preds_t.sum()
                captured       = ((preds_t == 1) & (actuals == 1)).sum()
                missed         = total_orders - captured
                precision      = captured / calls if calls > 0 else 0
                recall         = captured / total_orders if total_orders > 0 else 0
                call_reduction = (1 - calls / total_r) * 100
                rows.append({
                    'Threshold': f"{int(t*100)}%",
                    'Calls Made': f"{int(calls):,}",
                    'Call Reduction': f"{call_reduction:.0f}%",
                    'Orders Captured': f"{int(captured):,}",
                    'Orders Missed': f"{int(missed):,}",
                    'Precision': f"{precision*100:.1f}%",
                    'Recall': f"{recall*100:.1f}%"
                })

            thresh_df = pd.DataFrame(rows)

            def highlight_threshold(row):
                t_val = int(row['Threshold'].replace('%','')) / 100
                if abs(t_val - threshold) < 0.01:
                    return ['background-color: #1a3a5c; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.dataframe(
                thresh_df.style.apply(highlight_threshold, axis=1),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Highlighted row = your current threshold ({int(threshold*100)}%)")
        else:
            st.info("Threshold analysis requires actual labels. Available when evaluating on historical dates.")

    st.markdown("---")
    st.subheader("How To Read This")
    st.markdown("""
    - **Precision** = Of all retailers we called, what % actually ordered? (Higher = less wasted calls)
    - **Recall** = Of all retailers who would order, what % did we catch? (Higher = fewer missed orders)
    - **Call Reduction** = How many fewer calls vs calling everyone
    - **Recommended threshold: 40%** — good balance between call reduction and order capture
    """)

# ─────────────────────────────────────────────
# PAGE 4: NEXT ORDER SCHEDULE
# ─────────────────────────────────────────────
elif page == "Next Order Schedule":
    st.title("📅 Next Order Schedule")
    st.markdown("Predicted date when each retailer will place their next order.")
    st.markdown("---")

    schedule_path = os.path.join(OUTPUTS_DIR, 'next_order_prediction.csv')

    if os.path.exists(schedule_path):
        schedule = pd.read_csv(schedule_path)
        schedule['predicted_next_order_date'] = pd.to_datetime(schedule['predicted_next_order_date'])
        schedule['last_order_date'] = pd.to_datetime(schedule['last_order_date'])

        # Summary
        today = pd.Timestamp('2026-05-31')
        due_today     = (schedule['predicted_next_order_date'].dt.date == today.date()).sum()
        due_tomorrow  = (schedule['predicted_next_order_date'].dt.date == (today + timedelta(days=1)).date()).sum()
        due_this_week = (schedule['predicted_next_order_date'] <= today + timedelta(days=7)).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Due Today",       f"{due_today:,}")
        col2.metric("Due Tomorrow",    f"{due_tomorrow:,}")
        col3.metric("Due This Week",   f"{due_this_week:,}")

        st.markdown("---")

        # Filters
        with st.expander("Filter Options", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                date_filter = st.date_input("Show retailers due on or before",
                                             value=today + timedelta(days=3))
            with col_b:
                hub_f = st.multiselect("Filter by Hub",
                                        options=sorted(schedule['hubName'].dropna().unique()),
                                        default=[])

        filtered_s = schedule[schedule['predicted_next_order_date'].dt.date <= date_filter]
        if hub_f:
            filtered_s = filtered_s[filtered_s['hubName'].isin(hub_f)]
        filtered_s = filtered_s.sort_values('predicted_next_order_date')

        st.markdown(f"**{len(filtered_s):,} retailers** due by {date_filter}")

        display_s = filtered_s[[
            'customerId', 'hubName', 'shopType',
            'last_order_date', 'historical_avg_gap_days',
            'predicted_days_until_next', 'predicted_next_order_date'
        ]].copy()

        display_s['last_order_date'] = display_s['last_order_date'].dt.date
        display_s['predicted_next_order_date'] = display_s['predicted_next_order_date'].dt.date
        display_s['historical_avg_gap_days'] = display_s['historical_avg_gap_days'].round(1)
        display_s['predicted_days_until_next'] = display_s['predicted_days_until_next'].round(1)

        display_s = display_s.rename(columns={
            'customerId':               'Retailer ID',
            'hubName':                  'Hub',
            'shopType':                 'Shop Type',
            'last_order_date':          'Last Order Date',
            'historical_avg_gap_days':  'Avg Gap (days)',
            'predicted_days_until_next':'Days Until Next',
            'predicted_next_order_date':'Predicted Order Date'
        })

        st.dataframe(display_s, use_container_width=True, height=500, hide_index=True)

        csv_s = display_s.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "Download Schedule as CSV",
            data=csv_s,
            file_name="next_order_schedule.csv",
            mime='text/csv'
        )

        st.markdown("---")
        st.subheader("Orders Predicted Per Day")
        daily_pred = schedule.groupby(schedule['predicted_next_order_date'].dt.date).size().reset_index()
        daily_pred.columns = ['Date', 'Predicted Orders']
        daily_pred = daily_pred[daily_pred['Date'] >= today.date()]

        fig = px.bar(daily_pred.head(14), x='Date', y='Predicted Orders',
                     color_discrete_sequence=['#4f8bf9'],
                     title='Predicted Order Volume — Next 14 Days')
        fig = apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Next order prediction file not found. Please run Notebook 5 first.")
        st.code("Run: 05_next_order_prediction.ipynb")

# ─────────────────────────────────────────────
# PAGE 5: RETAILER DEEP DIVE
# ─────────────────────────────────────────────
elif page == "Retailer Deep Dive":
    st.title("Retailer Deep Dive")
    st.markdown("Analyze individual retailers, their order history, and AI explanations for their score.")
    st.markdown("---")

    if predictions is not None:
        retailer_list = sorted(predictions['customerId'].unique().tolist())
        selected_id = st.selectbox("Search / Select Retailer ID", options=retailer_list)
        
        if selected_id:
            # 1. Retailer Profile Stats
            r_pred = predictions[predictions['customerId'] == selected_id].iloc[0]
            r_prof_df = profiles[profiles['customerId'] == selected_id]
            r_prof = r_prof_df.iloc[0] if not r_prof_df.empty else None
            
            st.subheader(f"Retailer: {selected_id}")
            
            hub = r_prof['hub'] if r_prof is not None else "Unknown"
            prob_val = r_pred['order_probability']
            prob_color = "#4ade80" if prob_val >= threshold else "#f87171"
            
            days_since = r_pred['days_since_last_order']
            avg_gap = r_pred['avg_gap_between_orders']
            gap_color = "#f87171" if days_since > avg_gap else "#4ade80"
            
            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="flex: 1; background: #1e1e2e; border-radius: 12px; padding: 20px; border-left: 5px solid #a855f7; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #aaa; font-size: 0.9rem;">Hub</p>
                    <h3 style="margin: 5px 0 0 0; color: #a855f7; font-size: 1.8rem;">{hub}</h3>
                </div>
                <div style="flex: 1; background: #1e1e2e; border-radius: 12px; padding: 20px; border-left: 5px solid {prob_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #aaa; font-size: 0.9rem;">Probability to Order</p>
                    <h3 style="margin: 5px 0 0 0; color: {prob_color}; font-size: 1.8rem;">{prob_val*100:.1f}%</h3>
                </div>
                <div style="flex: 1; background: #1e1e2e; border-radius: 12px; padding: 20px; border-left: 5px solid {gap_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #aaa; font-size: 0.9rem;">Days Since Last Order</p>
                    <h3 style="margin: 5px 0 0 0; color: {gap_color}; font-size: 1.8rem;">{days_since:.0f}</h3>
                </div>
                <div style="flex: 1; background: #1e1e2e; border-radius: 12px; padding: 20px; border-left: 5px solid #eab308; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #aaa; font-size: 0.9rem;">Avg Gap (Days)</p>
                    <h3 style="margin: 5px 0 0 0; color: #eab308; font-size: 1.8rem;">{avg_gap:.1f}</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            col_chart, col_shap = st.columns([1, 1])
            
            # 2. Historical Timeline
            with col_chart:
                st.subheader("Order History")
                r_orders = orders[orders['customerId'] == selected_id].copy()
                if not r_orders.empty:
                    r_orders['date'] = r_orders['createdAt'].dt.date
                    daily_orders = r_orders.groupby('date').size().reset_index(name='orders')
                    
                    end_date = pd.Timestamp(pred_date).date()
                    start_date = end_date - timedelta(days=60)
                    date_range = pd.date_range(start=start_date, end=end_date).date
                    timeline = pd.DataFrame({'date': date_range})
                    timeline = timeline.merge(daily_orders, on='date', how='left').fillna(0)
                    
                    fig = px.bar(timeline, x='date', y='orders', 
                                 title="Orders Placed (Last 60 Days)",
                                 color_discrete_sequence=['#4f8bf9'])
                    fig.update_traces(marker_line_width=0, opacity=0.9)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)',
                        bargap=0.2,
                        yaxis=dict(tickformat="d", dtick=1) # force integer ticks
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No historical orders found.")
            
            # 3. SHAP Explanation
            with col_shap:
                st.subheader("AI Decision Explanation")
                st.markdown("Why did the model give this probability score?")
                
                try:
                    FEATURE_COLS = model_data['feature_cols']
                    model        = model_data['model']
                    
                    X_selected = np.nan_to_num(r_pred[FEATURE_COLS].values.reshape(1, -1), nan=0.0)
                    X_selected_df = pd.DataFrame(X_selected, columns=FEATURE_COLS).astype(float)
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_selected_df)
                    
                    # Matplotlib dark theme
                    plt.style.use("dark_background")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Waterfall plot
                    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
                    
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    st.pyplot(fig, clear_figure=True)
                    
                    # Top feature summary
                    vals = shap_values[0].values
                    f_importance = sorted(zip(FEATURE_COLS, vals), key=lambda x: abs(x[1]), reverse=True)
                    top_f = f_importance[0]
                    direction = "increased" if top_f[1] > 0 else "decreased"
                    
                    st.success(f"**Key Driver**: The feature `{top_f[0]}` {direction} the probability the most.")
                    
                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {e}")