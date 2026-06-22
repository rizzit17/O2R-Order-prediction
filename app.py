import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, warnings
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="O2R : Order Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — UPDATE BASE_DIR TO YOUR PROJECT FOLDER
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, 'models',    'xgboost_order_model.pkl')
REG_MODEL_PATH = os.path.join(BASE_DIR, 'models',    'xgboost_next_order_model.pkl')
ENCODER_PATH   = os.path.join(BASE_DIR, 'processed', 'label_encoders.pkl')
PROFILE_PATH   = os.path.join(BASE_DIR, 'processed', 'retailer_profiles.parquet')
RAW_CSV_PATH   = os.path.join(BASE_DIR, 'data', "Jan - May '26 Data.csv")
OUTPUTS_DIR    = os.path.join(BASE_DIR, 'outputs')

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
.section-header {
    background: linear-gradient(90deg, #1e3a5f, #0d1b2a);
    padding: 10px 18px; border-radius: 8px;
    color: #e0e0e0; font-size: 1.05rem; font-weight: 600;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_reg_model():
    if not os.path.exists(REG_MODEL_PATH):
        return None
    with open(REG_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_profile():
    return pd.read_parquet(PROFILE_PATH)

@st.cache_data
def load_orders():
    df = pd.read_csv(RAW_CSV_PATH)
    df['createdAt'] = pd.to_datetime(df['createdAt'], dayfirst=True)
    confirmed = df[df['orderStatus'].isin(['Delivered','PartiallyDelivered'])]
    orders = confirmed.drop_duplicates(subset='orderNumber')[[
        'orderNumber','customerId','createdAt',
        'hubName','shopType','retailerType','orderSource'
    ]].copy()
    return orders.sort_values(['customerId','createdAt'])

@st.cache_data
def load_june_schedule():
    path = os.path.join(OUTPUTS_DIR, 'next_order_prediction_june2026.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['predicted_next_order_date'] = pd.to_datetime(df['predicted_next_order_date'])
        df['last_order_date']           = pd.to_datetime(df['last_order_date'])
        return df
    return None

@st.cache_data
def load_june_call_schedule():
    path = os.path.join(OUTPUTS_DIR, 'june_2026_call_schedule.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER (same logic as Notebook 4)
# ─────────────────────────────────────────────────────────────────────────────
def build_features_for_date(orders, profile, encoders, target_date):
    target_date = pd.Timestamp(target_date)
    hist = orders[orders['createdAt'] < target_date].copy()

    last_order = hist.groupby('customerId')['createdAt'].max().reset_index()
    last_order.columns = ['customerId','last_order_date']
    last_order['days_since_last_order'] = (target_date - last_order['last_order_date']).dt.days

    def cnt(n):
        cut = target_date - pd.Timedelta(days=n)
        return (hist[hist['createdAt'] >= cut]
                .groupby('customerId')['orderNumber'].count()
                .reset_index()
                .rename(columns={'orderNumber': f'orders_last_{n}_days'}))

    hist_s = hist.sort_values(['customerId','createdAt'])
    hist_s['gap'] = hist_s.groupby('customerId')['createdAt'].diff().dt.days
    gap_stats = hist_s.groupby('customerId')['gap'].agg(
        avg_gap_between_orders='mean',
        std_gap_between_orders='std',
        median_gap='median'
    ).reset_index().fillna({'avg_gap_between_orders':30,
                             'std_gap_between_orders':0,
                             'median_gap':30})

    total_sf = (hist.groupby('customerId')['orderNumber'].count()
                .reset_index().rename(columns={'orderNumber':'total_orders_so_far'}))

    app_r = (hist.groupby('customerId')
             .apply(lambda x: (x['orderSource']=='App').mean())
             .reset_index())
    app_r.columns = ['customerId','app_order_ratio']

    f = (profile[['customerId','hubName','shopType','retailerType','tenure_days']]
         .merge(last_order[['customerId','last_order_date','days_since_last_order']], on='customerId', how='left')
         .merge(cnt(3),    on='customerId', how='left')
         .merge(cnt(7),    on='customerId', how='left')
         .merge(cnt(14),   on='customerId', how='left')
         .merge(cnt(30),   on='customerId', how='left')
         .merge(total_sf,  on='customerId', how='left')
         .merge(gap_stats, on='customerId', how='left')
         .merge(app_r,     on='customerId', how='left'))

    fill = {
        'days_since_last_order':999,'orders_last_3_days':0,
        'orders_last_7_days':0,'orders_last_14_days':0,'orders_last_30_days':0,
        'total_orders_so_far':0,'avg_gap_between_orders':30,
        'std_gap_between_orders':0,'median_gap':30,'app_order_ratio':0.5,'tenure_days':0
    }
    f = f.fillna(fill)

    f['days_overdue']      = (f['days_since_last_order'] - f['avg_gap_between_orders']).clip(lower=0)
    f['is_overdue']        = (f['days_overdue'] > 0).astype(int)
    f['order_regularity']  = 1 / (f['std_gap_between_orders'] + 1)
    f['overdue_ratio']     = (f['days_since_last_order'] / (f['avg_gap_between_orders']+1)).clip(upper=10).round(3)
    f['day_of_week']       = target_date.dayofweek
    f['day_of_month']      = target_date.day
    f['week_of_month']     = (target_date.day - 1) // 7 + 1
    f['month']             = target_date.month
    f['is_weekend']        = int(target_date.dayofweek >= 5)
    f['is_month_start']    = int(target_date.day <= 3)
    f['is_month_end']      = int(target_date.day >= 28)
    f['date']              = target_date

    for col in ['hubName','shopType','retailerType']:
        le    = encoders[col]
        known = set(le.classes_)
        f[col]          = f[col].apply(lambda x: x if str(x) in known else le.classes_[0])
        f[col+'_enc']   = le.transform(f[col].astype(str))

    return f

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading model and data..."):
        model_data  = load_model()
        encoders    = load_encoders()
        profile     = load_profile()
        orders      = load_orders()
        reg_data    = load_reg_model()
        june_sched  = load_june_schedule()
        june_calls  = load_june_call_schedule()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.info("Run all 5 notebooks first, then relaunch this app.")
    st.stop()

FEATURE_COLS = model_data['feature_cols']
model        = model_data['model']
history_end  = orders['createdAt'].max().normalize()
forecast_min = (history_end + pd.Timedelta(days=1)).to_pydatetime()
forecast_max = (history_end + pd.Timedelta(days=90)).to_pydatetime()
default_pred = forecast_min

PAGE_OPTIONS = {
    'landing': 'Landing Page',
    'overview': 'Overview',
    'daily_call_list': 'Daily Call List',
    'june_schedule': 'June Schedule',
    'model_performance': 'Model Performance',
    'next_order_dates': 'Next Order Dates',
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
if 'current_page' not in st.session_state:
    st.session_state.current_page = PAGE_OPTIONS['landing']
if 'page_selector' not in st.session_state:
    st.session_state.page_selector = st.session_state.current_page
elif st.session_state.page_selector != st.session_state.current_page:
    st.session_state.page_selector = st.session_state.current_page

with st.sidebar:
    st.markdown("**O2R Order Prediction**")
    st.markdown(f"*Model trained: {model_data.get('trained_on', 'Historical data')}*")
    st.markdown(f"*Forecast window: {forecast_min.strftime('%b %d, %Y')} to {forecast_max.strftime('%b %d, %Y')}*")
    st.markdown("---")

    page_options = [
        PAGE_OPTIONS["landing"],
        PAGE_OPTIONS["overview"],
        PAGE_OPTIONS["daily_call_list"],
        PAGE_OPTIONS["june_schedule"],
        PAGE_OPTIONS["model_performance"],
        PAGE_OPTIONS["next_order_dates"],
    ]
    page = st.radio(
        "",
        page_options,
        index=page_options.index(st.session_state.current_page),
        key="page_selector",
        label_visibility="collapsed"
    )
    if page != st.session_state.current_page:
        st.session_state.current_page = page

    st.markdown("---")
    st.markdown("Call Threshold")
    threshold = st.slider(
        "Min probability to call",
        min_value=0.10, max_value=0.90,
        value=float(model_data.get('threshold', 0.4)),
        step=0.05
    )
    st.caption(f"Calling retailers with >= {int(threshold*100)}% order probability")

    st.markdown("---")
    st.markdown("### Forecast Date")
    pred_date = st.date_input(
        "Select a forecast date",
        value=default_pred,
        min_value=forecast_min,
        max_value=forecast_max
    )

    st.markdown("---")

# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# GENERATE PREDICTIONS FOR SELECTED DATE
# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
@st.cache_data(show_spinner=False)
def get_predictions(pred_date_str, threshold):
    f = build_features_for_date(orders, profile, encoders, pred_date_str)
    X = np.nan_to_num(f[FEATURE_COLS].values, nan=0.0)
    f = f.copy()
    f['order_probability'] = model.predict_proba(X)[:, 1]
    f['will_order']        = (f['order_probability'] >= threshold).astype(int)
    return f

preds = None
if page != PAGE_OPTIONS["landing"]:
    with st.spinner(f"Scoring retailers for {pred_date}..."):
        preds = get_predictions(str(pred_date), threshold)

# PAGE: LANDING
if page == PAGE_OPTIONS["landing"]:
    st.title("O2R Order Prediction")
    st.markdown("A machine learning dashboard for retailer order propensity, call prioritization, and next-order planning.")
    st.markdown("---")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("What This Dashboard Does")
        st.markdown("""
- Scores each retailer for a selected forecast date using order probability.
- Generates a daily call priority list so the team can focus on the most likely buyers.
- Shows forecasted call reduction and business impact from selective outreach.
- Summarizes precomputed schedules and next-order timelines.
- Supports planning with both classification and regression outputs.
""")

        st.subheader("ML Concepts Used")
        st.markdown("""
- Supervised learning for retailer-level daily order prediction.
- Binary classification to estimate whether a retailer will order on a given day.
- Regression to estimate days until the next expected order.
- Time-aware validation using past data to predict future periods.
- Class imbalance handling with weighted learning and threshold-based decisions.
- Ranking retailers by predicted probability instead of using a manual rule.
""")

    with right:
        st.subheader("Feature Engineering Used")
        st.markdown("""
- Recency features such as days since last order.
- Frequency features such as orders in the last 3, 7, 14, and 30 days.
- Gap features such as average, median, and standard deviation of order gaps.
- Overdue features such as days overdue and overdue ratio.
- Calendar features such as day of week, month boundaries, and weekend flags.
- Retailer profile features such as hub, shop type, retailer type, tenure, and app-order ratio.
""")

        st.subheader("Models and Outputs")
        st.markdown("""
- XGBoost Classifier for order probability scoring.
- XGBoost Regressor for next-order date estimation.
- Threshold-based call decisions for operational use.
- Ranked call lists, schedule views, and performance summaries inside the engine.
""")

    st.markdown("---")
    if st.button("Proceed to Engine", type="primary", use_container_width=True):
        st.session_state.current_page = PAGE_OPTIONS["overview"]
        st.rerun()

# PAGE: OVERVIEW
if page == PAGE_OPTIONS["overview"]:
    st.title("O2R : Retailer Order Prediction")
    st.markdown(
        f"**Model:** Trained on {model_data.get('trained_on', 'historical data')} "
        f"&nbsp;|&nbsp; **Predicting for:** {pred_date.strftime('%B %d, %Y')}"
    )
    st.markdown("---")

    total     = len(preds)
    to_call   = int(preds['will_order'].sum())
    to_skip   = total - to_call
    reduction = (1 - to_call / total) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Retailers",   f"{total:,}")
    c2.metric("📞 Call Today",     f"{to_call:,}", delta=f"-{to_skip:,} skipped")
    c3.metric("📉 Call Reduction", f"{reduction:.1f}%")
    c4.metric("Avg Probability",   f"{preds['order_probability'].mean()*100:.1f}%")

    st.markdown("---")

    # Business impact
    st.subheader("Estimated Business Impact")
    col_a, col_b = st.columns(2)
    COST_PER_MIN  = 8
    AVG_CALL_MINS = 2

    with col_a:
        st.markdown("**Before (Calling Everyone)**")
        cur_cost = total * COST_PER_MIN * AVG_CALL_MINS
        st.markdown(f"- Daily calls: **{total:,}**")
        st.markdown(f"- AI voice cost @ ₹{COST_PER_MIN}/min × {AVG_CALL_MINS} min")
        st.markdown(f"- **Daily cost: ₹{cur_cost:,}**")
        st.markdown(f"- Monthly (26 days): **₹{cur_cost*26:,}**")

    with col_b:
        st.markdown("**After (With Model)**")
        new_cost = to_call * COST_PER_MIN * AVG_CALL_MINS
        savings  = cur_cost - new_cost
        st.markdown(f"- Daily calls: **{to_call:,}**")
        st.markdown(f"- AI voice cost @ ₹{COST_PER_MIN}/min × {AVG_CALL_MINS} min")
        st.markdown(f"- **Daily cost: ₹{new_cost:,}**")
        st.success(f"💸 Daily saving: ₹{savings:,}  |  Monthly: ₹{savings*26:,}")

    st.markdown("---")

    # Probability distribution
    st.subheader(f"Probability Distribution — {pred_date}")
    fig = px.histogram(preds, x='order_probability', nbins=50,
                       color_discrete_sequence=['#2196F3'],
                       labels={'order_probability':'Predicted Order Probability'})
    fig.add_vline(x=threshold, line_dash='dash', line_color='red',
                  annotation_text=f"Threshold ({int(threshold*100)}%)")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      xaxis_title='Order Probability', yaxis_title='Retailers', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Hub breakdown
    st.subheader("📍 Call Distribution by Hub")
    hub_df = preds.groupby('hubName').agg(
        total=('customerId','count'), call=('will_order','sum')
    ).reset_index()
    hub_df['skip'] = hub_df['total'] - hub_df['call']
    hub_df = hub_df.sort_values('call', ascending=False)

    fig2 = go.Figure([
        go.Bar(name='Call', x=hub_df['hubName'], y=hub_df['call'],   marker_color='#2196F3'),
        go.Bar(name='Skip', x=hub_df['hubName'], y=hub_df['skip'],   marker_color='#37474F')
    ])
    fig2.update_layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-35,
                       legend=dict(orientation='h', y=1.05))
    st.plotly_chart(fig2, use_container_width=True)

    # June monthly call volume (if pre-computed)
    if june_calls is not None:
        st.subheader("📆 June 2026 — Predicted Daily Call Volume")
        fig3 = px.bar(june_calls, x='date', y='calls_needed',
                      color='call_reduction', color_continuous_scale='Blues',
                      labels={'calls_needed':'Calls Needed','date':'Date',
                              'call_reduction':'Reduction %'},
                      title=f'Daily calls needed at {int(threshold*100)}% threshold')
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Avg Daily Calls (June)",    f"{june_calls['calls_needed'].mean():.0f}")
        col2.metric("Avg Call Reduction (June)", f"{june_calls['call_reduction'].mean():.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DAILY CALL LIST
# ─────────────────────────────────────────────────────────────────────────────
elif page == PAGE_OPTIONS["daily_call_list"]:
    st.title(f"Call Priority List — {pred_date.strftime('%B %d, %Y')}")
    st.markdown("Retailers ranked by order probability. Call top-down.")
    st.markdown("---")

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        hub_f  = st.multiselect("Hub",       sorted(preds['hubName'].dropna().unique()))
    with c2:
        shop_f = st.multiselect("Shop Type", sorted(preds['shopType'].dropna().unique()))
    with c3:
        show   = st.radio("Show", ["All","Call Only","Skip Only"], horizontal=True)

    filtered = preds.copy()
    if hub_f:   filtered = filtered[filtered['hubName'].isin(hub_f)]
    if shop_f:  filtered = filtered[filtered['shopType'].isin(shop_f)]
    if show == "Call Only": filtered = filtered[filtered['will_order']==1]
    if show == "Skip Only": filtered = filtered[filtered['will_order']==0]

    filtered = filtered.sort_values('order_probability', ascending=False).reset_index(drop=True)
    filtered.index += 1

    st.markdown(f"Showing **{len(filtered):,}** retailers &nbsp;|&nbsp; "
                f"Call: **{filtered['will_order'].sum():,}** &nbsp;|&nbsp; "
                f"Skip: **{(filtered['will_order']==0).sum():,}**")

    display = filtered[[
        'customerId','hubName','shopType','retailerType',
        'order_probability','will_order',
        'days_since_last_order','avg_gap_between_orders',
        'days_overdue','orders_last_7_days','total_orders_so_far','app_order_ratio'
    ]].copy()

    display['order_probability']      = (display['order_probability'] * 100).round(1)
    display['avg_gap_between_orders'] = display['avg_gap_between_orders'].round(1)
    display['days_overdue']           = display['days_overdue'].round(1)
    display['app_order_ratio']        = (display['app_order_ratio'] * 100).round(0).astype(int)
    display['Action']                 = display['will_order'].map({1:'📞 CALL', 0:'⏭️ SKIP'})

    display = display.drop(columns=['will_order']).rename(columns={
        'customerId':             'Retailer ID',
        'hubName':                'Hub',
        'shopType':               'Shop Type',
        'retailerType':           'Type',
        'order_probability':      'Prob %',
        'days_since_last_order':  'Days Since Last',
        'avg_gap_between_orders': 'Avg Gap',
        'days_overdue':           'Days Overdue',
        'orders_last_7_days':     'Orders (7d)',
        'total_orders_so_far':    'Total Orders',
        'app_order_ratio':        'App Usage %'
    })

    st.dataframe(display, use_container_width=True, height=580)

    csv = display.reset_index(names='Rank').to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Call List CSV", data=csv,
                       file_name=f"call_list_{pred_date}.csv", mime='text/csv')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: JUNE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
elif page == PAGE_OPTIONS["june_schedule"]:
    st.title("June 2026 Full Month Call Schedule")
    st.markdown("Pre-computed predictions for all 30 days of June.")
    st.markdown("---")

    if june_calls is not None:
        st.dataframe(june_calls.rename(columns={
            'date':'Date','day':'Day','calls_needed':'Calls Needed',
            'skipped':'Skipped','call_reduction':'Reduction %'
        }), use_container_width=True, hide_index=True)

        total_june_calls = june_calls['calls_needed'].sum()
        total_without    = june_calls['skipped'].sum() + total_june_calls
        june_savings     = int((total_without - total_june_calls) * 8 * 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Calls Saved (June)", f"{total_without-total_june_calls:,}")
        c2.metric("Avg Daily Reduction",       f"{june_calls['call_reduction'].mean():.1f}%")
        c3.metric("Estimated Monthly Savings", f"₹{june_savings:,}")

        csv = june_calls.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download June Schedule", data=csv,
                           file_name="june_2026_call_schedule.csv", mime='text/csv')
    else:
        st.warning("June schedule not found. Run Notebook 4 fully (Step 6 generates all-June CSV).")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == PAGE_OPTIONS["model_performance"]:
    st.title("Model Performance")
    st.markdown("Validated on May 2026 (held-out), trained on Jan–Apr 2026.")
    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance — XGBoost")
    feat_df = pd.DataFrame({
        'Feature':    FEATURE_COLS,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      coloraxis_showscale=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Threshold Decision Guide")
    st.markdown("""
    | Threshold | Strategy | Best For |
    |---|---|---|
    | **20–30%** | Cast wide net | Missing orders is costly |
    | **40%** | Balanced (recommended) | Good balance of calls vs capture |
    | **60–70%** | High precision | Call volume is the bottleneck |

    **Key metrics explained:**
    - **Precision** = Of all calls made, how many resulted in an order
    - **Recall** = Of all retailers who would order, how many did we reach
    - **Call Reduction** = % fewer calls vs calling every retailer
    """)

    st.markdown("---")
    st.subheader("Model Info")
    st.markdown(f"""
    - **Algorithm:** XGBoost Classifier
    - **Features:** {len(FEATURE_COLS)} engineered features
    - **Training data:** Jan 1 – May 31 2026 (full 151 days)
    - **Predicts for:** June 2026
    - **Class imbalance handling:** `scale_pos_weight` (neg:pos ratio)
    - **Top feature:** `days_since_last_order` (recency is strongest signal)
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NEXT ORDER DATES
# ─────────────────────────────────────────────────────────────────────────────
elif page == PAGE_OPTIONS["next_order_dates"]:
    st.title("Predicted Next Order Dates")
    st.markdown("When is each retailer expected to order next relative to the selected forecast date?")
    st.markdown("---")

    if june_sched is not None:
        anchor_date = pd.Timestamp(pred_date)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Due in 1-3 days",
            int(((june_sched['predicted_next_order_date'] >= anchor_date) &
                 (june_sched['predicted_next_order_date'] <= anchor_date + timedelta(days=3))).sum())
        )
        c2.metric(
            "Due in 4-7 days",
            int(((june_sched['predicted_next_order_date'] > anchor_date + timedelta(days=3)) &
                 (june_sched['predicted_next_order_date'] <= anchor_date + timedelta(days=7))).sum())
        )
        c3.metric(
            "Due after 7 days",
            int((june_sched['predicted_next_order_date'] > anchor_date + timedelta(days=7)).sum())
        )

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input(
                "Show retailers due on or before",
                value=min((anchor_date + timedelta(days=7)).date(), forecast_max.date())
            )
        with col2:
            hub_f2 = st.multiselect("Filter by Hub",
                                     sorted(june_sched['hubName'].dropna().unique()))

        filtered_s = june_sched[june_sched['predicted_next_order_date'].dt.date <= date_filter]
        if hub_f2:
            filtered_s = filtered_s[filtered_s['hubName'].isin(hub_f2)]
        filtered_s = filtered_s.sort_values('predicted_next_order_date')

        st.markdown(f"**{len(filtered_s):,} retailers** due by {date_filter}")

        disp_s = filtered_s[[
            'customerId','hubName','shopType',
            'last_order_date','historical_avg_gap_days',
            'predicted_days_until_next','predicted_next_order_date'
        ]].copy()
        disp_s['last_order_date']           = disp_s['last_order_date'].dt.date
        disp_s['predicted_next_order_date'] = disp_s['predicted_next_order_date'].dt.date
        disp_s['historical_avg_gap_days']   = disp_s['historical_avg_gap_days'].round(1)
        disp_s['predicted_days_until_next'] = disp_s['predicted_days_until_next'].round(1)
        disp_s = disp_s.rename(columns={
            'customerId':               'Retailer ID',
            'hubName':                  'Hub',
            'shopType':                 'Shop Type',
            'last_order_date':          'Last Order',
            'historical_avg_gap_days':  'Avg Gap (days)',
            'predicted_days_until_next':'Days Until Next',
            'predicted_next_order_date':'Predicted Order Date'
        })
        st.dataframe(disp_s, use_container_width=True, height=500, hide_index=True)

        csv_s = disp_s.to_csv(index=False).encode('utf-8')
        st.download_button("Download Schedule CSV", data=csv_s,
                           file_name="next_order_schedule_june2026.csv", mime='text/csv')

        st.markdown("---")
        st.subheader("Expected Orders Per Day")
        daily = june_sched.groupby(june_sched['predicted_next_order_date'].dt.date).size().reset_index()
        daily.columns = ['Date','Expected Orders']
        fig = px.bar(daily, x='Date', y='Expected Orders',
                     color_discrete_sequence=['#2196F3'])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Next order schedule not found. Run Notebook 5 first.")
        st.code("Run: 05_next_order_prediction.ipynb")

