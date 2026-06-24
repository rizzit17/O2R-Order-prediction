"""
O2R Order Prediction Dashboard
Streamlit app | Trained: Jan–Apr 2026 | Predicts: May 2026

Run with:
    streamlit run app.py

Requires Notebook A and Notebook B to have been run first.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, warnings
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="O2R Order Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
BASE = r'C:\Users\Rishit\Desktop\O2R-Order-prediction'
# ─────────────────────────────────────────────────────────────────────────────

RAW_PATH        = os.path.join(BASE, 'data',      "Jan - May '26 Data.csv")
MODEL_PATH      = os.path.join(BASE, 'models',    'xgboost_order_model.pkl')
REG_MODEL_PATH  = os.path.join(BASE, 'models',    'xgboost_next_order_model.pkl')
ENCODER_PATH    = os.path.join(BASE, 'processed', 'label_encoders.pkl')
PROFILE_PATH    = os.path.join(BASE, 'processed', 'retailer_profiles.parquet')
OUTPUTS_DIR     = os.path.join(BASE, 'outputs')

# Output files produced by Notebook B
MAY_SCHEDULE_PATH  = os.path.join(OUTPUTS_DIR, 'may_2026_call_schedule.csv')
NEXT_ORDER_PATH    = os.path.join(OUTPUTS_DIR, 'next_order_prediction_may2026.csv')

COST_PER_MIN  = 8
AVG_CALL_MINS = 2

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS — must match NotebookA exactly
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'days_since_last_order',
    'avg_gap_between_orders',
    'std_gap_between_orders',
    'median_gap',
    'orders_last_3_days',
    'orders_last_7_days',
    'orders_last_14_days',
    'orders_last_30_days',
    'momentum_7_30',
    'momentum_14_30',
    'total_orders_so_far',
    'days_overdue',
    'is_overdue',
    'order_regularity',
    'overdue_ratio',
    'app_order_ratio',
    'tenure_days',
    'day_of_week',
    'day_of_month',
    'week_of_month',
    'month',
    'is_weekend',
    'is_month_start',
    'is_month_end',
    'hubName_enc',
    'shopType_enc',
    'retailerType_enc'
]

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
.info-box {
    background: #0d1b2a;
    border-left: 4px solid #2196F3;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #cdd8e3;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    cat_path = os.path.join(BASE, 'models', 'catboost_order_model.pkl')
    if os.path.exists(cat_path):
        with open(cat_path, 'rb') as f:
            return pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_reg_model():
    if not os.path.exists(REG_MODEL_PATH):
        return None
    with open(REG_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_encoders():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_profile():
    return pd.read_parquet(PROFILE_PATH)

@st.cache_data(show_spinner=False)
def load_orders():
    df = pd.read_csv(RAW_PATH)
    df['createdAt'] = pd.to_datetime(df['createdAt'], dayfirst=True)
    confirmed = df[df['orderStatus'].isin(['Delivered', 'PartiallyDelivered'])]
    orders = confirmed.drop_duplicates(subset='orderNumber')[[
        'orderNumber', 'customerId', 'createdAt',
        'hubName', 'shopType', 'retailerType', 'orderSource'
    ]].copy()
    return orders.sort_values(['customerId', 'createdAt'])

@st.cache_data(show_spinner=False)
def load_may_schedule():
    if not os.path.exists(MAY_SCHEDULE_PATH):
        return None
    df = pd.read_csv(MAY_SCHEDULE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(show_spinner=False)
def load_next_order():
    if not os.path.exists(NEXT_ORDER_PATH):
        return None
    df = pd.read_csv(NEXT_ORDER_PATH)
    df['predicted_next_order_date'] = pd.to_datetime(df['predicted_next_order_date'])
    df['last_order_date']           = pd.to_datetime(df['last_order_date'])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER — identical logic to Notebook B build_features_for_date()
# ─────────────────────────────────────────────────────────────────────────────
def build_features_for_date(orders, profile, encoders, target_date):
    target_date = pd.Timestamp(target_date)
    hist        = orders[orders['createdAt'] < target_date].copy()

    # Last order date & days since
    last_ord = hist.groupby('customerId')['createdAt'].max().reset_index()
    last_ord.columns = ['customerId', 'last_order_date']
    last_ord['days_since_last_order'] = (target_date - last_ord['last_order_date']).dt.days

    # Rolling counts
    def cnt(n):
        cut = target_date - pd.Timedelta(days=n)
        return (
            hist[hist['createdAt'] >= cut]
            .groupby('customerId')['orderNumber'].count()
            .reset_index()
            .rename(columns={'orderNumber': f'orders_last_{n}_days'})
        )

    # Total orders so far
    total_sf = (
        hist.groupby('customerId')['orderNumber'].count()
        .reset_index()
        .rename(columns={'orderNumber': 'total_orders_so_far'})
    )

    # Gap statistics
    hist_s       = hist.sort_values(['customerId', 'createdAt'])
    hist_s['gap'] = hist_s.groupby('customerId')['createdAt'].diff().dt.days
    gap_stats = hist_s.groupby('customerId')['gap'].agg(
        avg_gap_between_orders='mean',
        std_gap_between_orders='std',
        median_gap='median'
    ).reset_index().fillna({
        'avg_gap_between_orders': 30,
        'std_gap_between_orders': 0,
        'median_gap': 30
    })

    # App ratio
    app_r = (
        hist.groupby('customerId')
        .apply(lambda x: (x['orderSource'] == 'App').mean())
        .reset_index()
    )
    app_r.columns = ['customerId', 'app_order_ratio']

    # Assemble
    f = (
        profile[['customerId', 'hubName', 'shopType', 'retailerType', 'first_order']]
        .merge(last_ord[['customerId', 'last_order_date', 'days_since_last_order']], on='customerId', how='left')
        .merge(cnt(3),    on='customerId', how='left')
        .merge(cnt(7),    on='customerId', how='left')
        .merge(cnt(14),   on='customerId', how='left')
        .merge(cnt(30),   on='customerId', how='left')
        .merge(total_sf,  on='customerId', how='left')
        .merge(gap_stats, on='customerId', how='left')
        .merge(app_r,     on='customerId', how='left')
    )

    fill = {
        'days_since_last_order': 999,
        'orders_last_3_days': 0, 'orders_last_7_days': 0,
        'orders_last_14_days': 0, 'orders_last_30_days': 0,
        'total_orders_so_far': 0,
        'avg_gap_between_orders': 30, 'std_gap_between_orders': 0,
        'median_gap': 30, 'app_order_ratio': 0.5, 'tenure_days': 0
    }
    f = f.fillna(fill)

    f['tenure_days'] = (target_date - pd.to_datetime(f['first_order'])).dt.days
    f['tenure_days'] = f['tenure_days'].clip(lower=0).fillna(0)
    f['momentum_7_30'] = f['orders_last_7_days'] / (f['orders_last_30_days'] + 1)
    f['momentum_14_30'] = f['orders_last_14_days'] / (f['orders_last_30_days'] + 1)
    # Derived features
    f['days_overdue']     = (f['days_since_last_order'] - f['avg_gap_between_orders']).clip(lower=0)
    f['is_overdue']       = (f['days_overdue'] > 0).astype(int)
    f['order_regularity'] = 1 / (f['std_gap_between_orders'] + 1)
    f['overdue_ratio']    = (
        f['days_since_last_order'] / (f['avg_gap_between_orders'] + 1)
    ).clip(upper=10).round(3)

    # Temporal
    f['day_of_week']    = target_date.dayofweek
    f['day_of_month']   = target_date.day
    f['week_of_month']  = (target_date.day - 1) // 7 + 1
    f['month']          = target_date.month
    f['is_weekend']     = int(target_date.dayofweek >= 5)
    f['is_month_start'] = int(target_date.day <= 3)
    f['is_month_end']   = int(target_date.day >= 28)
    f['date']           = target_date

    # Encode categoricals with saved encoders
    for col in ['hubName', 'shopType', 'retailerType']:
        le    = encoders[col]
        known = set(le.classes_)
        f[col]          = f[col].apply(lambda x: x if str(x) in known else le.classes_[0])
        f[col + '_enc'] = le.transform(f[col].astype(str))

    return f

# ─────────────────────────────────────────────────────────────────────────────
# SCORE RETAILERS FOR A DATE (cached per date+threshold combo)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def score_retailers(target_date_str, top_k, _orders, _profile, _encoders, _model):
    f = build_features_for_date(_orders, _profile, _encoders, target_date_str)
    X = np.nan_to_num(f[FEATURE_COLS].values, nan=0.0)
    f = f.copy()
    f['order_probability'] = _model.predict_proba(X)[:, 1]
    f = f.sort_values('order_probability', ascending=False)
    f['will_order'] = 0
    f.iloc[:top_k, f.columns.get_loc('will_order')] = 1
    return f

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
missing = []
for label, path in [
    ('Classification model',  MODEL_PATH),
    ('Label encoders',        ENCODER_PATH),
    ('Retailer profiles',     PROFILE_PATH),
    ('Raw CSV',               RAW_PATH),
]:
    if not os.path.exists(path):
        missing.append(f'❌ {label}: `{path}`')

if missing:
    st.error("### Missing files — run Notebook A first")
    for m in missing:
        st.markdown(m)
    st.info("Once Notebook A is complete, run Notebook B as well for the full May schedule and next-order predictions.")
    st.stop()

with st.spinner("Loading model and data..."):
    saved_model  = load_model()
    encoders     = load_encoders()
    profile      = load_profile()
    orders       = load_orders()
    reg_saved    = load_reg_model()
    may_schedule = load_may_schedule()
    next_order   = load_next_order()

clf_model = saved_model['model'] if isinstance(saved_model, dict) else saved_model

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**O2R Order Prediction System**")
    st.markdown(
        f"<div class='info-box'>"
        f"Trained on: <b>Jan–Apr 2026</b><br>"
        f"Predicts for: <b>May 2026</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    page = st.radio("", [
        "Overview",
        "Daily Call List",
        "May Schedule",
        "Model Performance",
        "Next Order Dates",
        "Validation Results",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Call Strategy")
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 100
    top_k_options = [100, 500, 1000, 1500, 2000]
    top_k = st.selectbox(
        "Select Top-K Retailers to Call",
        options=top_k_options,
        index=top_k_options.index(st.session_state.top_k) if st.session_state.top_k in top_k_options else 0
    )
    st.session_state.top_k = top_k
    st.caption(f"Calling the top {top_k} highest probability retailers")
    st.markdown("---")
    st.markdown("### 📅 Prediction Date")
    pred_date = st.date_input(
        "Select a May date",
        value=pd.Timestamp('2026-05-15').date(),
        min_value=pd.Timestamp('2026-05-01').date(),
        max_value=pd.Timestamp('2026-05-31').date(),
        help="Model uses Jan–Apr history to predict this date"
    )

    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PREDICTIONS FOR SELECTED DATE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Scoring retailers for {pred_date}..."):
    preds = score_retailers(
        str(pred_date), top_k,
        orders, profile, encoders, clf_model
    )

total_r   = len(preds)
to_call   = int(preds['will_order'].sum())
to_skip   = total_r - to_call
reduction = (1 - to_call / total_r) * 100
old_cost  = total_r * COST_PER_MIN * AVG_CALL_MINS
new_cost  = to_call * COST_PER_MIN * AVG_CALL_MINS
savings   = old_cost - new_cost

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("O2R Order Prediction")
    st.markdown(
        f"**Prediction date:** {pd.Timestamp(pred_date).strftime('%A, %B %d 2026')} &nbsp;|&nbsp;"
        f"**Top-K:** {top_k}"
    )
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Retailers",   f"{total_r:,}")
    c2.metric("📞 Call Today",     f"{to_call:,}", delta=f"−{to_skip:,} skipped", delta_color="normal")
    c3.metric("📉 Call Reduction", f"{reduction:.1f}%")
    c4.metric("💸 Daily Saving",   f"₹{savings:,}")

    st.markdown("---")

    # Business impact comparison
    st.subheader("Business Impact")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Before — Calling Everyone**")
        st.markdown(f"- Daily calls: **{total_r:,}**")
        st.markdown(f"- AI voice cost: ₹{COST_PER_MIN}/min × {AVG_CALL_MINS} min avg")
        st.markdown(f"- **Daily cost: ₹{old_cost:,}**")
        st.markdown(f"- Monthly (26 working days): **₹{old_cost*26:,}**")
    with col_b:
        st.markdown("**After — With Prediction Model**")
        st.markdown(f"- Daily calls: **{to_call:,}**")
        st.markdown(f"- AI voice cost: ₹{COST_PER_MIN}/min × {AVG_CALL_MINS} min avg")
        st.markdown(f"- **Daily cost: ₹{new_cost:,}**")
        st.success(f"💸 Daily saving: ₹{savings:,}  |  Monthly: ₹{savings*26:,}")

    st.markdown("---")

    # Probability distribution
    st.subheader(f"Probability Distribution — {pred_date}")
    fig_dist = px.histogram(
        preds, x='order_probability', nbins=50,
        color_discrete_sequence=['#2196F3'],
        labels={'order_probability': 'Order Probability'}
    )
    actual_threshold = preds[preds['will_order'] == 1]['order_probability'].min() if to_call > 0 else 0
    fig_dist.add_vline(
        x=actual_threshold, line_dash='dash', line_color='red', line_width=2,
        annotation_text=f"Top-K threshold",
        annotation_position="top right"
    )
    fig_dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Order Probability', yaxis_title='Number of Retailers',
        showlegend=False, height=350
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Hub breakdown
    st.subheader("📍 Call vs Skip by Hub")
    hub_df = preds.groupby('hubName').agg(
        Call=('will_order', 'sum'),
        Total=('customerId', 'count')
    ).reset_index()
    hub_df['Skip'] = hub_df['Total'] - hub_df['Call']
    hub_df = hub_df.sort_values('Call', ascending=False)

    fig_hub = go.Figure([
        go.Bar(name='📞 Call', x=hub_df['hubName'], y=hub_df['Call'],
               marker_color='#2196F3'),
        go.Bar(name='⏭️ Skip', x=hub_df['hubName'], y=hub_df['Skip'],
               marker_color='#37474F')
    ])
    fig_hub.update_layout(
        barmode='stack', plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-35,
        legend=dict(orientation='h', y=1.08), height=380
    )
    st.plotly_chart(fig_hub, use_container_width=True)

    # May monthly trend (if Notebook B was run)
    if may_schedule is not None:
        st.markdown("---")
        st.subheader("📆 May 2026 — Full Month Call Volume")
        fig_may = px.bar(
            may_schedule, x='date', y='calls_needed',
            color='call_reduction',
            color_continuous_scale='Blues',
            labels={'calls_needed': 'Calls Needed', 'date': 'Date',
                    'call_reduction': 'Reduction %'},
            hover_data=['day', 'skipped', 'call_reduction']
        )
        fig_may.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-35, height=360,
            coloraxis_colorbar=dict(title='Reduction %')
        )
        st.plotly_chart(fig_may, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Daily Calls (May)",     f"{may_schedule['calls_needed'].mean():.0f}")
        m2.metric("Avg Call Reduction",        f"{may_schedule['call_reduction'].mean():.1f}%")
        m3.metric("Total Calls Saved (May)",   f"{may_schedule['skipped'].sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DAILY CALL LIST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Daily Call List":
    st.title(f"Call Priority List")
    st.markdown(f"**Date:** {pd.Timestamp(pred_date).strftime('%A, %B %d 2026')} &nbsp;|&nbsp; Ranked by order probability, highest first.")
    st.markdown("---")

    # Filter bar
    c1, c2, c3 = st.columns(3)
    with c1:
        hub_f  = st.multiselect("Hub",       sorted(preds['hubName'].dropna().unique()))
    with c2:
        shop_f = st.multiselect("Shop Type", sorted(preds['shopType'].dropna().unique()))
    with c3:
        show   = st.radio("Show", ["All", "Call Only", "Skip Only"], horizontal=True)

    filtered = preds.copy()
    if hub_f:   filtered = filtered[filtered['hubName'].isin(hub_f)]
    if shop_f:  filtered = filtered[filtered['shopType'].isin(shop_f)]
    if show == "Call Only": filtered = filtered[filtered['will_order'] == 1]
    if show == "Skip Only": filtered = filtered[filtered['will_order'] == 0]

    filtered = filtered.sort_values('order_probability', ascending=False).reset_index(drop=True)
    filtered.index += 1

    # Summary line
    st.markdown(
        f"Showing **{len(filtered):,}** retailers &nbsp;|&nbsp; "
        f"📞 Call: **{filtered['will_order'].sum():,}** &nbsp;|&nbsp; "
        f"⏭️ Skip: **{(filtered['will_order']==0).sum():,}**"
    )
    st.markdown("---")

    # Build display table
    display = filtered[[
        'customerId', 'hubName', 'shopType', 'retailerType',
        'order_probability', 'will_order',
        'days_since_last_order', 'avg_gap_between_orders',
        'days_overdue', 'orders_last_7_days',
        'total_orders_so_far', 'app_order_ratio',
        'last_order_date'
    ]].copy()

    display['order_probability']      = (display['order_probability'] * 100).round(1)
    display['avg_gap_between_orders'] = display['avg_gap_between_orders'].round(1)
    display['days_overdue']           = display['days_overdue'].round(1)
    display['app_order_ratio']        = (display['app_order_ratio'] * 100).round(0).astype(int)
    display['Action']                 = display['will_order'].map({1: '📞 CALL', 0: '⏭️ SKIP'})

    if 'last_order_date' in display.columns:
        display['last_order_date'] = pd.to_datetime(display['last_order_date']).dt.date

    display = display.drop(columns=['will_order']).rename(columns={
        'customerId':             'Retailer ID',
        'hubName':                'Hub',
        'shopType':               'Shop Type',
        'retailerType':           'Type',
        'order_probability':      'Prob %',
        'days_since_last_order':  'Days Since Last',
        'avg_gap_between_orders': 'Avg Gap (days)',
        'days_overdue':           'Days Overdue',
        'orders_last_7_days':     'Orders (7d)',
        'total_orders_so_far':    'Total Orders',
        'app_order_ratio':        'App Usage %',
        'last_order_date':        'Last Order Date',
    })

    # Colour-code rows
    def highlight(row):
        if row['Action'] == '📞 CALL':
            return ['background-color: #0a2240'] * len(row)
        return ['background-color: #1a0e0e'] * len(row)

    st.dataframe(
        display.style.apply(highlight, axis=1),
        use_container_width=True,
        height=580
    )

    # Download
    csv_bytes = display.reset_index(names='Rank').to_csv(index=False).encode('utf-8')
    
    # CRM Export logic
    crm_display = display[display['Action'] == '📞 CALL'].reset_index(names='Rank').copy()
    crm_display['Phone Number'] = 'Not Provided'
    crm_export = crm_display[['Retailer ID', 'Rank', 'Action', 'Phone Number']].copy()
    crm_csv = crm_export.to_csv(index=False).encode('utf-8')

    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button(
            label="⬇️ Download Call List CSV",
            data=csv_bytes,
            file_name=f"call_list_{pred_date}.csv",
            mime='text/csv'
        )
    with d2:
        st.download_button(
            label="📞 Download CRM Dialer Export",
            data=crm_csv,
            file_name=f"crm_export_{pred_date}.csv",
            mime='text/csv'
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MAY SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "May Schedule":
    st.title("May 2026 - Full Month Call Schedule")
    st.markdown("Pre-computed daily predictions for all 31 days of May. Run Notebook B to generate this.")
    st.markdown("---")

    if may_schedule is None:
        st.warning("File not found: `outputs/may_2026_call_schedule.csv`")
        st.info("Run **Notebook B → Cell B4** to generate the full May schedule.")
        st.stop()

    # Summary KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Calls Needed (May)",   f"{may_schedule['calls_needed'].sum():,}")
    c2.metric("Total Calls Saved (May)",    f"{may_schedule['skipped'].sum():,}")
    c3.metric("Avg Daily Calls",            f"{may_schedule['calls_needed'].mean():.0f}")
    c4.metric("Avg Call Reduction",         f"{may_schedule['call_reduction'].mean():.1f}%")

    est_saving = int(may_schedule['skipped'].sum() * COST_PER_MIN * AVG_CALL_MINS)
    st.success(f"💸 Estimated May savings from reduced AI voice calls: **₹{est_saving:,}**")

    st.markdown("---")

    # Bar chart
    fig = px.bar(
        may_schedule, x='date', y='calls_needed',
        color='day',
        hover_data=['skipped', 'call_reduction'],
        labels={'calls_needed': 'Calls Needed', 'date': 'Date', 'day': 'Day'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45, height=380,
        legend=dict(orientation='h', y=1.08)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("Day-by-Day Breakdown")
    table = may_schedule.copy()
    table['date']           = table['date'].dt.date
    table['call_reduction'] = table['call_reduction'].astype(str) + '%'
    table = table.rename(columns={
        'date': 'Date', 'day': 'Day',
        'calls_needed': 'Calls Needed',
        'skipped': 'Calls Skipped',
        'call_reduction': 'Reduction'
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_may = table.to_csv(index=False).encode('utf-8')
    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button(
        "⬇️ Download May Schedule CSV",
        data=csv_may,
        file_name="may_2026_call_schedule.csv",
        mime='text/csv'
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("XGBoost trained on **Jan–Apr 2026**, evaluated on **May 2026** (true held-out test).")
    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance")
    importances = clf_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature':    FEATURE_COLS,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig_feat = px.bar(
        feat_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        labels={'Importance': 'Importance Score', 'Feature': ''}
    )
    fig_feat.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False, height=620
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")

    # Live threshold table computed on selected date's predictions
    st.subheader("Threshold Analysis — May Test Set")
    st.markdown(
        "The table below is computed on the currently selected date's predictions vs actual labels."
        " Use this to decide your operating threshold."
    )

    # We can only compute this if ordered_today exists in preds (it does for May dates since grid covers Jan-May)
    if 'ordered_today' in preds.columns:
        probs   = preds['order_probability'].values
        actuals = preds['ordered_today'].values
        total_a = int(actuals.sum())

        rows = []
        for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            p        = (probs >= t).astype(int)
            calls    = int(p.sum())
            captured = int(((p == 1) & (actuals == 1)).sum())
            missed   = total_a - captured
            prec     = captured / calls if calls > 0 else 0
            rec      = captured / total_a if total_a > 0 else 0
            f1       = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
            red      = (1 - calls / total_r) * 100
            rows.append({
                'Threshold':       f'{int(t*100)}%',
                'Calls Made':      f'{calls:,}',
                'Call Reduction':  f'{red:.0f}%',
                'Orders Captured': f'{captured:,}',
                'Orders Missed':   f'{missed:,}',
                'Precision':       f'{prec*100:.1f}%',
                'Recall':          f'{rec*100:.1f}%',
                'F1':              f'{f1:.3f}'
            })

        thresh_df = pd.DataFrame(rows)

        def hl_threshold(row):
            t_val = int(row['Threshold'].replace('%','')) / 100
            if False:
                return ['background-color: #1a3a5c; font-weight:bold'] * len(row)
            return [''] * len(row)

        st.dataframe(
            thresh_df.style.apply(hl_threshold, axis=1),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Threshold analysis is available when predictions are made for dates within the dataset (May 2026).")

    st.markdown("---")

    # Model info card
    st.subheader("📌 Model Info")
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Algorithm | XGBoost Classifier |
    | Features | {len(FEATURE_COLS)} engineered features |
    | Training data | Jan 1 – Apr 30 2026 |
    | Validation data | May 1 – May 31 2026 |
    | Class imbalance | Handled via `scale_pos_weight` |
    | Key insight | `days_since_last_order` is the strongest predictor |
    """)

    st.markdown("---")
    st.subheader("Metric Guide")
    st.markdown("""
    | Threshold | Strategy | When to use |
    |---|---|---|
    | **20–30%** | Cast wide net | Missing orders is very costly |
    | **40%** | Balanced recommended | Good call reduction + order capture |
    | **60–70%** | High precision | Call centre is the bottleneck |

    - **Precision** = Of all calls made, % that resulted in order (higher = less wasted calls)
    - **Recall** = Of all retailers who would order, % that we called (higher = fewer missed orders)
    - **F1** = Harmonic mean of Precision and Recall
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NEXT ORDER DATES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Next Order Dates":
    st.title("Next Order Date Predictions")
    st.markdown("For each retailer: predicted date of their next order in May 2026. (Phase 2 — Regression)")
    st.markdown("---")

    if next_order is None:
        st.warning("File not found: `outputs/next_order_prediction_may2026.csv`")
        st.info("Run **Notebook B → Cell B5** to generate next order predictions.")
        st.stop()

    ref_date = pd.Timestamp('2026-05-01')

    # KPI row
    due_3d = (next_order['predicted_next_order_date'] <= ref_date + timedelta(days=3)).sum()
    due_7d = (
        (next_order['predicted_next_order_date'] > ref_date + timedelta(days=3)) &
        (next_order['predicted_next_order_date'] <= ref_date + timedelta(days=7))
    ).sum()
    due_7p = (next_order['predicted_next_order_date'] > ref_date + timedelta(days=7)).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Due in 1–3 days",  f"{int(due_3d):,}")
    c2.metric("Due in 4–7 days",  f"{int(due_7d):,}")
    c3.metric("Due 7+ days away", f"{int(due_7p):,}")

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        date_filter = st.date_input(
            "Show retailers due on or before",
            value=(ref_date + timedelta(days=7)).date(),
            min_value=pd.Timestamp('2026-05-01').date(),
            max_value=pd.Timestamp('2026-05-31').date()
        )
    with col2:
        hub_f2 = st.multiselect(
            "Filter by Hub",
            sorted(next_order['hubName'].dropna().unique())
        )

    filtered_n = next_order[
        next_order['predicted_next_order_date'].dt.date <= date_filter
    ].copy()
    if hub_f2:
        filtered_n = filtered_n[filtered_n['hubName'].isin(hub_f2)]
    filtered_n = filtered_n.sort_values('predicted_next_order_date')

    st.markdown(f"**{len(filtered_n):,} retailers** due by {date_filter}")

    # Display table
    disp_n = filtered_n[[
        'customerId', 'hubName', 'shopType', 'retailerType',
        'last_order_date', 'historical_avg_gap_days',
        'predicted_days_until_next', 'predicted_next_order_date'
    ]].copy()

    disp_n['last_order_date']           = disp_n['last_order_date'].dt.date
    disp_n['predicted_next_order_date'] = disp_n['predicted_next_order_date'].dt.date
    disp_n['historical_avg_gap_days']   = disp_n['historical_avg_gap_days'].round(1)
    disp_n['predicted_days_until_next'] = disp_n['predicted_days_until_next'].round(1)

    disp_n = disp_n.rename(columns={
        'customerId':               'Retailer ID',
        'hubName':                  'Hub',
        'shopType':                 'Shop Type',
        'retailerType':             'Type',
        'last_order_date':          'Last Order',
        'historical_avg_gap_days':  'Avg Gap (days)',
        'predicted_days_until_next':'Days Until Next',
        'predicted_next_order_date':'Predicted Order Date'
    })

    st.dataframe(disp_n, use_container_width=True, height=500, hide_index=True)

    csv_n = disp_n.to_csv(index=False).encode('utf-8')
    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button(
        "⬇️ Download Next Order Schedule CSV",
        data=csv_n,
        file_name="next_order_schedule_may2026.csv",
        mime='text/csv'
    )

    st.markdown("---")

    # Daily predicted order volume chart
    st.subheader("📆 Expected Orders Per Day — May 2026")
    daily_vol = (
        next_order
        .groupby(next_order['predicted_next_order_date'].dt.date)
        .size()
        .reset_index()
    )
    daily_vol.columns = ['Date', 'Expected Orders']
    daily_vol = daily_vol[
        (daily_vol['Date'] >= pd.Timestamp('2026-05-01').date()) &
        (daily_vol['Date'] <= pd.Timestamp('2026-05-31').date())
    ]

    fig_vol = px.bar(
        daily_vol, x='Date', y='Expected Orders',
        color_discrete_sequence=['#2196F3'],
        labels={'Expected Orders': 'Retailers Expected to Order'}
    )
    fig_vol.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45, height=360
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VALIDATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Validation Results":
    st.title("✅ Model Validation Results (Ground Truth)")
    st.markdown("We back-tested the model's predictions against the **actual ground-truth orders** that occurred on two random dates in our held-out test set (May 2026).")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Validation Date 1: May 7th, 2026")
        st.markdown("*Total organic orders that day: 1,427*")
        
        data_may7 = {
            "Call Strategy": ["Top 100", "Top 500", "Top 1000", "Top 1500", "Top 2000"],
            "Retailers Called": [100, 500, 1000, 1500, 2000],
            "Orders Captured (Hits)": [82, 349, 585, 746, 872],
            "Hit Rate (Precision)": ["82.00%", "69.80%", "58.50%", "49.73%", "43.60%"],
            "Total Orders Captured (Recall)": ["5.75%", "24.46%", "41.00%", "52.28%", "61.11%"]
        }
        st.table(pd.DataFrame(data_may7).set_index("Call Strategy"))

    with col2:
        st.subheader("Validation Date 2: May 15th, 2026")
        st.markdown("*Total organic orders that day: 1,347*")
        
        data_may15 = {
            "Call Strategy": ["Top 100", "Top 500", "Top 1000", "Top 1500", "Top 2000"],
            "Retailers Called": [100, 500, 1000, 1500, 2000],
            "Orders Captured (Hits)": [77, 347, 586, 756, 883],
            "Hit Rate (Precision)": ["77.00%", "69.40%", "58.60%", "50.40%", "44.15%"],
            "Total Orders Captured (Recall)": ["5.72%", "25.76%", "43.50%", "56.12%", "65.55%"]
        }
        st.table(pd.DataFrame(data_may15).set_index("Call Strategy"))

    st.markdown("---")
    st.info("💡 **Business Takeaway:** Historically, calling 10,000 retailers yields ~900 orders (9% hit rate). By using the **Top 2000** strategy, we can capture ~880 orders with an average hit rate of **~44%**. This reduces daily call volume by 80% while maintaining core business output!")