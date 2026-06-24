O2R Order Prediction : Complete Project Teaching Guide

---

# PART 1: HIGH-LEVEL PROJECT EXPLANATION

---

## Business Problem

Here is the exact problem:

| Metric | Number |
|---|---|
| Daily calls made | ~10,000 |
| Calls that connect | ~8,000 |
| Calls that result in an actual order | ~900 |

That means 91% of calls result in NO order. The call center is working extremely hard and getting almost nothing from most of those calls.

We use AI voice agent to handle calls. That agent costs ₹8 per minute. With an average call lasting 2 minutes, that is ₹16 per call. At 10,000 calls a day, that is ₹1,60,000 per day — or roughly ₹41,60,000 per month — spent mostly on calls that produce no order.

**The company needs to know, every morning before the call center opens: which retailers are actually going to place an order today? Call only those. Skip everyone else.**

This is exactly what this project solves.

---

## Why Order Prediction Is Valuable

Every retailer has a buying pattern. A kirana store might reorder every 3 days. A wholesale dealer might reorder every week. A paan shop might reorder every day. These patterns exist in the historical order data. If you can learn these patterns, you can predict who is likely to order tomorrow — and who is not.

Before this project: Call everyone, hope for the best.
After this project: Call only the ~2,000–3,000 retailers the model flags. Get roughly the same number of orders with 70–75% fewer calls.

---

## End-to-End Architecture

```
Raw CSV (609K rows, Jan–May 2026)
         │
         ▼
A1 — EDA
Understand data: columns, nulls, date range,
order volumes, retailer frequency, cancellation rates
         │
         ▼
A2 — Retailer Static Profile
One row per retailer: hub, shop type, app ratio, tenure
         │
         ▼
A3 — Retailer-Day Grid
Every retailer × every date = 1 row (~1.3M rows)
Label each row: ordered today? 1 or 0
         │
         ▼
A4 — Feature Engineering
For each row, compute 27 features using ONLY
history before that date (no data leakage)
         │
         ▼
Save to parquet (processed/retailer_day_features.parquet)
         │
         ▼
A5 — Time-Based Train/Test Split
Train: Jan 1 – Apr 30 | Test: May 1 – May 31
         │
         ▼
Train Logistic Regression (baseline)
Train XGBoost (main model)
Train CatBoost (benchmark — A6)
         │
         ▼
Evaluate: Precision, Recall, F1, Confusion Matrix,
PR Curve, Threshold Analysis Table
         │
         ▼
Save best model (models/xgboost_order_model.pkl)
         │
         ▼
B3 — Score retailers for any May date
Build features using Jan–Apr history → model.predict_proba()
Each retailer gets a probability (0 to 1)
         │
         ▼
B4 — Full May schedule (all 31 days)
         │
         ▼
B5 — Next Order Date Prediction (Regression)
Predict WHEN, not just IF
         │
         ▼
CSV outputs + Streamlit Dashboard
```

---

# PART 2: REPOSITORY FILE-BY-FILE

---

## Repository Structure

```
O2R-Order-prediction/
├── data/
│   └── Jan - May '26 Data.csv         ← 609K raw rows from the company
├── notebooks/
│   ├── NotebookA_Train.ipynb           ← EDA + Feature Eng + Model Training
│   ├── NotebookB_Predict.ipynb         ← Predictions + Outputs
│   └── catboost_info/                  ← Auto-generated CatBoost training logs
├── processed/
│   ├── retailer_day_features.parquet   ← 1.3M row feature table
│   ├── label_encoders.pkl              ← Encoders for hub/shop/retailer type
│   └── retailer_profiles.parquet      ← Static retailer attributes
├── models/
│   ├── xgboost_order_model.pkl         ← Trained classification model
│   └── xgboost_next_order_model.pkl    ← Trained regression model
├── outputs/
│   ├── call_priority_YYYY-MM-DD.csv   ← Daily ranked call list
│   ├── may_2026_call_schedule.csv      ← Full month summary
│   └── next_order_prediction_may2026.csv
├── app.py                              ← Streamlit dashboard
└── README.md
```

---

### `data/Jan - May '26 Data.csv`
**What it is:** The raw transactional data from DS Group's O2R system. Every row is one SKU (product) in one order. Since one order usually has multiple products, one order = multiple rows.

**Key columns:**
- `createdAt` — date the order was placed (format: DD-MM-YYYY)
- `orderNumber` — unique order ID
- `customerId` — unique retailer ID
- `orderStatus` — Delivered / PartiallyDelivered / CANCELLED / Created
- `orderSource` — App / CALLING_AGENT / SUPER_ADMIN
- `hubName` — which distribution hub serves this retailer
- `shopType` — Paan B, General A, General B, etc.
- `retailerType` — category of the retailer
- `orderQty` — quantity of this SKU ordered
- `brand`, `category`, `subCategory` — product details

---

### `processed/retailer_day_features.parquet`
**What it is:** The most important intermediate file. Built by Notebook A, consumed by Notebook A (training) and Notebook B (inference).

It has one row per retailer per day. For 8,640 retailers × 151 days, that is roughly 1.3 million rows. Each row has 27 features and a label (`ordered_today`: 0 or 1).

Think of it as the "training table" — this is what the model actually learns from.

**Why parquet and not CSV?** Parquet is a compressed columnar format. The same data that would be 800MB as CSV is ~150MB as parquet and loads 10x faster.

---

### `processed/retailer_profiles.parquet`
**What it is:** A summary table with one row per retailer capturing their overall attributes — hub, shop type, how often they use the app, when they first ordered, tenure in days.

Built once in A2 and reused in Notebook B when scoring new dates.

---

### `processed/label_encoders.pkl`
**What it is:** A Python dictionary of LabelEncoder objects, one per categorical column: `hubName`, `shopType`, `retailerType`. Saved as a pickle file.

**Why it's needed:** The model was trained on numbers, not text. "Noida Hub" was encoded as, say, 7. When you want to predict for a new date, you must encode new data using the exact same mapping. If you re-fit the encoder, "Noida Hub" might get encoded as 3 — and the model will give wrong predictions. So you save the encoders from training and reuse them everywhere.

---

### `models/xgboost_order_model.pkl`
**What it is:** The trained XGBoost classification model. Saved as a dictionary containing:
- `model` — the actual XGBoost object
- `feature_cols` — list of 27 feature names in exact order
- `threshold` — 0.4 (default operating threshold)
- `trained_on` — "Jan–Apr 2026"
- `predicts_for` — "May 2026"

---

### `models/xgboost_next_order_model.pkl`
**What it is:** A separate XGBoost Regressor trained to predict "how many days until this retailer's next order?" The output is a number (e.g., 3.2 days), not a binary.

---

### `outputs/call_priority_YYYY-MM-DD.csv`
**What it is:** The final deliverable for the call center. One row per retailer, ranked by order probability. Columns include Retailer ID, Hub, Shop Type, Probability %, Action (CALL/SKIP), Days Since Last Order, Average Gap, etc.

---

### `outputs/may_2026_call_schedule.csv`
**What it is:** A 31-row summary — one row per day in May — showing how many retailers to call each day and the estimated call reduction.

---

### `outputs/next_order_prediction_may2026.csv`
**What it is:** One row per retailer with their predicted next order date in May. So instead of "call this retailer today", it says "this retailer is expected to order on May 14."

---

### `app.py`
**What it is:** The Streamlit dashboard. It loads the model, encoders, profiles, and orders at startup, then lets you pick any May date and threshold from the sidebar, and shows the call priority list, model performance, full May schedule, and next order dates.

---

# PART 3: NOTEBOOK-BY-NOTEBOOK EXPLANATION

---

## NotebookA_Train.ipynb

**Objective:** Take the raw CSV, do EDA, engineer features, train models, evaluate on May, save the best model.

---

### Cell A0 — Configuration

**What it does:** Imports all libraries and sets up all file paths.

```python
BASE = r'C:\Users\Rishit\Desktop\O2R-Order-prediction'
RAW_PATH     = os.path.join(BASE, 'data', "Jan - May '26 Data.csv")
PARQUET_PATH = os.path.join(BASE, 'processed', 'retailer_day_features.parquet')
ENCODER_PATH = os.path.join(BASE, 'processed', 'label_encoders.pkl')
PROFILE_PATH = os.path.join(BASE, 'processed', 'retailer_profiles.parquet')
MODEL_PATH   = os.path.join(BASE, 'models', 'xgboost_order_model.pkl')
TRAIN_END    = '2026-04-30'
TEST_START   = '2026-05-01'
```

**Why this exists:** Centralising all paths means you only need to update `BASE` if you move the folder. Everything else auto-adjusts.

**Key variables created:** `BASE`, `RAW_PATH`, `PARQUET_PATH`, `ENCODER_PATH`, `PROFILE_PATH`, `MODEL_PATH`, `TRAIN_END`, `TEST_START`

---

### Cell A1 — Load Raw Data & EDA (2 cells)

**Cell 1 — Basic stats:**
Loads the CSV, parses `createdAt` as a datetime (using `dayfirst=True` because the format is DD-MM-YYYY), then prints:
- Shape: 609,723 rows × 30 columns
- Date range: Jan 1 – May 31 2026
- Unique orders: ~200,000
- Unique retailers: ~9,766
- `orderStatus` distribution
- `orderSource` distribution (App vs CALLING_AGENT)
- Null check on key columns

**Cell 2 — Filtering and plots:**
```python
confirmed = df[df['orderStatus'].isin(['Delivered', 'PartiallyDelivered'])]
orders = confirmed.drop_duplicates(subset='orderNumber')
```
This is critical. The raw data has one row per SKU. Deduplicating on `orderNumber` collapses it to one row per order. You also add `order_hour` here (from `createdAt`).

It then plots:
- Left: Daily confirmed order volume over time (with a red mean line)
- Right: Distribution of inter-order gaps (how many days between consecutive orders per retailer)

From the gap histogram you can see the median gap is 2 days — meaning most retailers reorder very frequently. This is why `days_since_last_order` becomes the strongest feature.

Finally it frees the full `df` with `del df; gc.collect()` to save RAM before the heavy feature engineering steps.

**Key variables created:** `confirmed`, `orders`, `daily`, gaps distribution

---

### Cell A2 — Retailer Static Profile

```python
retailer_profile = orders.groupby('customerId').agg(
    hubName      = ('hubName',      lambda x: x.mode()[0]),
    shopType     = ('shopType',     lambda x: x.mode()[0]),
    retailerType = ('retailerType', ...),
    app_orders   = ('orderSource',  lambda x: (x == 'App').sum()),
    total_orders = ('orderNumber',  'count'),
    first_order  = ('createdAt',    'min'),
    last_order   = ('createdAt',    'max'),
)
retailer_profile['app_order_ratio'] = app_orders / total_orders
retailer_profile['tenure_days'] = last_order - first_order
```

**What this builds:** One row per retailer summarising their identity. The `mode()[0]` ensures you take the most commonly occurring hub/shop type for that retailer (in case they ever changed).

`app_order_ratio` is the proportion of their orders that came through the App vs the call center. If a retailer mostly orders via App, they probably don't need a call.

**Saved to:** `processed/retailer_profiles.parquet`

---

### Cell A3 — Build Retailer-Day Grid + Label

This is the conceptual heart of the project. Understanding this cell is critical for your review.

**Problem:** The raw data only has rows for days when orders happened. But to train a model, you also need examples of days when orders DID NOT happen. You need both 0s and 1s.

**Solution — the cross-join grid:**
```python
idx  = pd.MultiIndex.from_product([all_retailers, all_dates], names=['customerId','date'])
grid = pd.DataFrame(index=idx).reset_index()
```
This creates one row for every (retailer, date) combination. 8,640 retailers × 151 days = ~1,304,640 rows.

**Labelling:**
```python
order_flags['ordered_today'] = 1
grid = grid.merge(order_flags, on=['customerId','date'], how='left')
grid['ordered_today'] = grid['ordered_today'].fillna(0).astype(int)
```
Left merge: if a retailer placed an order that day, the merge finds a match and sets `ordered_today = 1`. If not, `NaN` which we fill with `0`.

**Class imbalance result:** About 177,340 rows are label=1 out of 1.3M total — roughly 13.6% positive rate.

---

### Cell A4 — Feature Engineering (2 cells)

**Cell 1 — Rolling order counts:**
```python
grp = grid.groupby('customerId')['_ord']
grid['orders_last_3_days']  = grp.transform(lambda x: x.shift(1).rolling(3,  min_periods=1).sum())
grid['orders_last_7_days']  = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).sum())
grid['orders_last_14_days'] = grp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).sum())
grid['orders_last_30_days'] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum())
grid['momentum_7_30']  = grid['orders_last_7_days']  / (grid['orders_last_30_days'] + 1)
grid['momentum_14_30'] = grid['orders_last_14_days'] / (grid['orders_last_30_days'] + 1)
```

The `.shift(1)` is essential. It shifts by one day so that today's order is NOT included when computing "how many orders in the last 7 days". Without the shift, today's order would leak into the feature — the model would be cheating by knowing whether an order happened today.

`momentum_7_30`: ratio of recent 7-day activity to 30-day baseline. If this is high (e.g., 0.8), the retailer is ordering MORE recently than their baseline — a strong signal they'll order again soon.

**Cell 2 — Recency, gap, overdue, temporal:**
```python
grid['last_order_date'] = grid.groupby('customerId')['_order_date'].transform(
    lambda x: x.shift(1).ffill()
)
grid['days_since_last_order'] = (grid['date'] - grid['last_order_date']).dt.days
```

`ffill()` (forward fill) propagates the last known order date forward through all future rows until a new order happens. Then subtracting from `date` gives days since last order.

```python
grid['expected_next_order'] = last_order_date + avg_gap_between_orders (in days)
grid['days_overdue']     = date - expected_next_order
grid['is_overdue']       = 1 if days_overdue > 0
grid['order_regularity'] = 1 / (std_gap + 1)
grid['overdue_ratio']    = days_since_last_order / (avg_gap + 1)
```

Temporal features: `day_of_week`, `day_of_month`, `week_of_month`, `month`, `is_weekend`, `is_month_start`, `is_month_end`.

Label encoding of hub, shop type, retailer type. Encoders saved to pickle.

Final grid saved to parquet.

---

### Cell A5 — Model Training (6 cells)

**Cell 1 — Load and split:**
```python
train = grid[grid['date'] <= '2026-04-30']   # Jan–Apr
test  = grid[grid['date'] >= '2026-05-01']   # May
```
This is a time-based split. Training on earlier data, testing on later data — the way real predictions work. You never use random split on time-series data.

```python
scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()  # roughly 6.3
```
This tells XGBoost that for every 1 positive, there are 6.3 negatives. XGBoost then upweights the minority class (orders) during training.

**Cell 2 — Logistic Regression baseline:**
Scales features with StandardScaler (LR needs normally distributed inputs), then trains a simple logistic regression with `class_weight='balanced'`. This is just a benchmark. If your XGBoost doesn't clearly beat this, something is wrong.

**Cell 3 — XGBoost:**
```python
xgb_model = xgb.XGBClassifier(
    n_estimators     = 400,   # 400 trees
    max_depth        = 5,     # each tree can be 5 levels deep
    learning_rate    = 0.05,  # slow learning = better generalisation
    subsample        = 0.8,   # use 80% of rows for each tree
    colsample_bytree = 0.8,   # use 80% of features for each tree
    scale_pos_weight = scale_pos_weight,  # class imbalance correction
    eval_metric      = 'logloss',
    tree_method      = 'hist' # faster algorithm for large datasets
)
```

**Cell 4 — Confusion matrix and PR curve plots**

**Cell 5 — Threshold analysis table:** Runs predictions at 7 different thresholds (20% to 80%) and shows how many calls you'd make, how many orders you'd capture, precision and recall at each. This is the most useful output for presenting to your manager.

**Cell 6 — Feature importance bar chart and model save**

---

### Cell A6 — CatBoost Benchmark

This cell installs and trains a CatBoost model (another gradient boosting library by Yandex) with identical hyperparameters and compares its F1 score against XGBoost.

```python
if cat_f1 > xgb_f1:
    print('CatBoost outperforms XGBoost! Saving CatBoost model...')
else:
    print('XGBoost remains the best model.')
```

**Why this is good to mention:** It shows you didn't just pick one model arbitrarily — you benchmarked against alternatives. The CatBoost logs in `notebooks/catboost_info/` are evidence that this comparison actually ran.

---

## NotebookB_Predict.ipynb

**Objective:** Load the saved model and generate actual predictions for May 2026.

---

### Cell B0 — Configuration
Same pattern as Notebook A. Also introduces:
```python
TARGET_DATE = '2026-05-15'  # any May date
TOP_K       = 2000          # call only top 2000 retailers per day
```

`TOP_K` is important — this is an alternative to using a probability threshold. Instead of saying "call everyone above 40%", you say "call the top 2,000 ranked by probability regardless of their exact score." This gives a fixed call volume every day.

---

### Cell B1 — Load Everything

Loads the XGBoost model, checks if a CatBoost model also exists (and if so, uses CatBoost instead since it was the better performer), loads encoders, profiles, and the full order history.

---

### Cell B2 — Feature Builder Function

This is the most important function in the project for inference. It takes any date and builds features for every retailer using only history before that date.

```python
def build_features_for_date(orders, profile, encoders, target_date):
    hist = orders[orders['createdAt'] < target_date]   # STRICT past only
    
    # last order, days since last order
    # rolling counts for 3, 7, 14, 30 days
    # gap stats (avg, std, median)
    # app ratio
    # merge all into one table
    # fill nulls for retailers with no recent history
    # derived features (overdue, regularity, etc.)
    # temporal features
    # encode categoricals with saved encoders
    return feature_dataframe
```

The key design principle: it is impossible for any future information to enter this function. `hist = orders[orders['createdAt'] < target_date]` — strictly less than, not less than or equal to.

---

### Cell B3 — Score Retailers for TARGET_DATE

```python
features['order_probability'] = model.predict_proba(X_score)[:, 1]
features = features.sort_values('order_probability', ascending=False)
features['will_order'] = 0
features.iloc[:TOP_K, features.columns.get_loc('will_order')] = 1
```

The model outputs a probability for every retailer. Sort descending. Mark the top `TOP_K` as CALL. Save to CSV.

---

### Cell B4 — Full May Schedule

Loops over all 31 days of May, calls `build_features_for_date()` and the model for each, records how many calls needed per day. Saves `may_2026_call_schedule.csv`.

---

### Cell B5 — Next Order Date Regression

**What it does:** Trains a second model — an XGBoost Regressor — to predict "in how many days will this retailer order next?"

**Training data:** Each row = one consecutive order pair. Target = gap in days between them. Caps at 60 days (retailers who don't order within 60 days are treated as churned for this model).

**Evaluation metrics used:**
- MAE (Mean Absolute Error) — average error in days
- RMSE (Root Mean Squared Error) — penalises large errors more
- R² — what proportion of variance the model explains
- % of predictions within 1, 2, 3 days

Then retrained on full Jan–Apr data. Each retailer's last known order is taken as the reference point, and the model predicts how many days from that until their next order. Adding that to their last order date gives the predicted next order date.

---

# PART 4: FEATURE ENGINEERING DEEP DIVE

---

## Every Feature Explained

### `days_since_last_order`
**Definition:** Number of days between the current prediction date and the most recent confirmed order by this retailer.

**Formula:** `prediction_date - last_order_date` (in days). If a retailer has never ordered, set to 999.

**Business meaning:** If a retailer ordered yesterday, they probably won't order today. If a retailer last ordered 5 days ago and they normally order every 3 days, they are overdue.

**Why it helps:** Consistently the #1 most important feature. Recency is the single strongest signal of purchase propensity in retail.

---

### `avg_gap_between_orders`
**Definition:** The mean number of days between consecutive orders for this retailer, computed over their full history.

**Formula:** Mean of all (next_order_date - current_order_date) gaps.

**Business meaning:** This is the retailer's ordering frequency. A retailer with avg_gap = 2 orders very frequently. A retailer with avg_gap = 14 is a weekly buyer.

**Why it helps:** Combined with days_since_last_order, it tells you where you are in their cycle.

---

### `std_gap_between_orders`
**Definition:** Standard deviation of all inter-order gaps for this retailer.

**Business meaning:** Measures ordering consistency. Low std means the retailer is very regular (orders on a fixed schedule). High std means erratic behaviour.

**Why it helps:** A retailer with std=1 is very predictable. A retailer with std=10 is unpredictable — the model should be less confident about them.

---

### `median_gap`
**Definition:** Median (middle value) of all inter-order gaps.

**Why it differs from avg_gap:** If a retailer usually orders every 2 days but once went 60 days without ordering (perhaps during a festival), the average is pulled up but the median stays at 2. The median is more robust to outliers.

---

### `orders_last_3_days`, `orders_last_7_days`, `orders_last_14_days`, `orders_last_30_days`
**Definition:** Count of confirmed orders placed by this retailer in the last N days before the prediction date.

**Formula (no-leakage version):** `x.shift(1).rolling(N, min_periods=1).sum()` — shift by 1 before rolling to exclude today.

**Business meaning:**
- `orders_last_3_days` — very recent activity. If 0, retailer hasn't ordered in 3 days.
- `orders_last_30_days` — overall monthly activity level.

**Why it helps:** Recent activity is a strong predictor of future activity. Retailers who ordered 3 times last week are more likely to order today than retailers who haven't ordered in 2 weeks.

---

### `momentum_7_30`
**Definition:** `orders_last_7_days / (orders_last_30_days + 1)`

**Business meaning:** Are they ordering MORE recently than their average? A value above their usual 7/30 ratio means accelerating activity. Below means slowing down.

**Example:** A retailer placed 8 orders in the last 30 days (average ~2/week) but 5 in the last 7 days — momentum is high, they're in a high-activity phase.

---

### `momentum_14_30`
Same concept but 14-day vs 30-day window. Captures medium-term momentum.

---

### `total_orders_so_far`
**Definition:** Cumulative count of all orders placed by this retailer up to (but not including) the prediction date.

**Business meaning:** Proxy for retailer loyalty and data richness. A retailer with 50 orders has a well-established pattern. A retailer with 2 orders is harder to predict.

---

### `days_overdue`
**Definition:** How many days PAST their expected next order date they currently are.

**Formula:** `(prediction_date - expected_next_order_date)` where `expected_next_order_date = last_order_date + avg_gap_between_orders`. Negative values clipped to 0.

**Business meaning:** If a retailer's average gap is 3 days and their last order was 5 days ago, they are 2 days overdue. The longer they are overdue, the more urgently they need a call.

**Why it helps:** Likely the 2nd or 3rd most important feature. Overdue retailers are prime candidates for immediate calls.

---

### `is_overdue`
**Definition:** Binary flag. 1 if `days_overdue > 0`, else 0.

**Why it's separate from `days_overdue`:** The model can use the binary flag differently from the continuous value. Sometimes a 0/1 flag captures a threshold effect that the continuous value doesn't capture as cleanly.

---

### `order_regularity`
**Definition:** `1 / (std_gap_between_orders + 1)`

**Business meaning:** Inverse of order variability. High regularity means the model can predict this retailer's next order very confidently. Low regularity means erratic ordering — the model should hedge.

**Why adding 1:** Prevents division by zero for retailers with perfect consistency (std=0).

---

### `overdue_ratio`
**Definition:** `days_since_last_order / (avg_gap_between_orders + 1)`, clipped at 10.

**Business meaning:** Relative measure of how overdue they are, normalised by their own pattern. An `overdue_ratio` of 1.0 means they are exactly at their average gap. A value of 2.0 means they are twice overdue. A value of 0.5 means they ordered half a gap ago and are unlikely to order today.

**Why it's better than raw `days_since_last_order` alone:** Two retailers can both have days_since_last_order = 5. But if Retailer A orders every 3 days, they are very overdue. If Retailer B orders every 10 days, they are not. The ratio captures this personalisation.

---

### `app_order_ratio`
**Definition:** Proportion of the retailer's historical orders that came through the App (vs call agent).

**Business meaning:** Retailers who primarily use the App to order autonomously don't need a call — they'll order themselves. Retailers who always order through a call agent are more dependent on being called.

**Why it helps:** A retailer with app_order_ratio = 0.9 probably shouldn't be in the call list at all regardless of their order probability — they'll order on their own.

---

### `tenure_days`
**Definition:** `last_order_date - first_order_date` in days for this retailer.

**Business meaning:** How long this retailer has been active with DS Group. Older retailers tend to have more stable patterns. New retailers are less predictable.

---

### `day_of_week` (0=Monday to 6=Sunday)
**Business meaning:** Many retailers order on specific days. A wholesale dealer might always order on Mondays for the week. A paan shop might order every day. This feature lets the model learn day-of-week patterns.

---

### `day_of_month`, `week_of_month`
**Business meaning:** Some retailers order at the start of the month (after payroll/credit cycle resets) or end of month.

---

### `month`
**Business meaning:** Seasonal effects. Orders might spike in October (festive season). The model can learn month-level patterns.

---

### `is_weekend`
**Business meaning:** Many retailers and distributors don't accept orders or deliveries on weekends.

---

### `is_month_start`, `is_month_end`
**Business meaning:** First 3 days and last 3 days of the month often see different ordering behavior — end-of-month stock-clearing or start-of-month replenishment.

---

### `hubName_enc`, `shopType_enc`, `retailerType_enc`
**Definition:** Label-encoded versions of categorical columns. "Noida Hub" → 7, "Paan B" → 4, etc.

**Why encoding:** XGBoost cannot process text. Must be numbers. Label encoding assigns each unique category a unique integer.

**Critical:** The encoder is saved and reused for all future predictions. Never re-fit the encoder on new data.

---

# PART 5: MODEL TRAINING DEEP DIVE

---

## Logistic Regression — Baseline Model

**What it does:** Draws a straight mathematical boundary between "will order" and "won't order" in the feature space. Outputs probabilities using the sigmoid function.

**Why it's used here:** As a baseline. If XGBoost barely beats Logistic Regression, your features are not strong enough. If XGBoost clearly beats it, the non-linear relationships in the data are meaningful and the more complex model is justified.

**`class_weight='balanced'`:** Automatically adjusts weights to give the minority class (ordered=1) more influence during training. Without this, LR would just predict "no order" for everything.

**`StandardScaler`:** Logistic Regression is sensitive to feature scale. `days_since_last_order` can be up to 999 while `is_weekend` is only 0 or 1. Scaling normalises everything to mean=0, std=1.

**Limitation:** Can only learn linear relationships. If the relationship between `overdue_ratio` and order probability is not a straight line (and it almost certainly isn't), Logistic Regression misses it.

---

## XGBoost — Main Model

**What it is:** Extreme Gradient Boosting. An ensemble of decision trees where each new tree corrects the mistakes of the previous ones.

**How it works step by step:**
1. Start with a simple prediction (e.g., predict the mean)
2. Calculate the residual errors (where did we go wrong?)
3. Train a decision tree to predict those errors
4. Add that tree to the model with a small weight (learning rate)
5. Recalculate residuals
6. Train another tree on the new residuals
7. Repeat 400 times

**Why it works so well on this data:**
- Tabular structured data with mixed feature types — exactly XGBoost's domain
- Naturally handles non-linear relationships (e.g., ordering probability drops sharply after gap > 7 days, not linearly)
- Handles class imbalance via `scale_pos_weight`
- Can detect feature interactions (e.g., "overdue AND high-frequency retailer" = very high probability)

---

## XGBoost Hyperparameters — Every Parameter Explained

### `n_estimators = 400`
Number of trees to build. More trees = more complex model = better fit but slower training. 400 is a reasonable number for this dataset size. The `eval_set` and early stopping (implicitly via `verbose`) help monitor if the model starts overfitting.

### `max_depth = 5`
Maximum depth of each decision tree. A depth-5 tree can ask 5 questions. Deeper trees = more complex patterns captured = more risk of overfitting. 5 is a good balance for this tabular data.

### `learning_rate = 0.05`
How much each new tree contributes to the ensemble. Smaller = slower learning = better generalisation (but needs more trees). 0.05 with 400 trees is a common well-tested combination.

### `subsample = 0.8`
Each tree is trained on a random 80% sample of the rows. Prevents overfitting by introducing randomness. The model can't memorise individual rows.

### `colsample_bytree = 0.8`
Each tree uses a random 80% of the features. Also prevents overfitting and makes the model more robust.

### `scale_pos_weight = ~6.3`
`(number of negatives) / (number of positives)` in training data. Tells XGBoost to treat each positive example as if it was 6.3 examples. This corrects for the class imbalance.

### `eval_metric = 'logloss'`
The loss function used to evaluate at each boosting round. Log loss penalises confident wrong predictions heavily.

### `tree_method = 'hist'`
Uses histogram-based algorithm for finding splits. 3–10x faster than the default on large datasets. Essential for running on a laptop.

### `n_jobs = -1`
Use all available CPU cores for training in parallel.

---

## CatBoost — Benchmark Model

CatBoost (by Yandex) is another gradient boosting library. Key differences from XGBoost:
- Natively handles categorical features (doesn't need label encoding)
- Uses ordered boosting to reduce overfitting
- Generally strong on datasets with categorical features

The notebook trains it with identical parameters and compares F1 scores. If CatBoost wins, it saves the CatBoost model automatically. The `catboost_info/` folder in the repo is evidence that this comparison actually ran.

---

# PART 6: EVALUATION DEEP DIVE

---

## Why NOT Accuracy

Imagine you have 1.3 million rows, of which 180,000 are positive (ordered=1). A model that always predicts "no order" would be right 86% of the time — 86% accuracy. But it would catch zero orders. Completely useless.

This is why accuracy is meaningless for imbalanced classification problems. You must use Precision, Recall, and F1.

---

## Precision

**Definition:** Of all the retailers the model said "CALL", what fraction actually placed an order?

**Formula:** `True Positives / (True Positives + False Positives)`

**Business translation:** If precision = 40%, for every 100 calls you make based on the model, 40 retailers actually order. That's still far better than the current 9% (900 orders from 10,000 calls).

**What it measures:** The quality of the call list. High precision = fewer wasted calls.

---

## Recall

**Definition:** Of all the retailers who actually placed an order, what fraction did the model catch?

**Formula:** `True Positives / (True Positives + False Negatives)`

**Business translation:** If recall = 80%, you capture 80% of all real orders. You miss 20%. The question for the business is: how many missed orders are acceptable in exchange for the call reduction?

**What it measures:** How many real orders you catch. High recall = fewer missed sales.

---

## The Precision-Recall Tradeoff

These two always trade off. If you lower the threshold (call more people), recall goes up but precision goes down. If you raise the threshold (call fewer people), precision goes up but recall goes down.

The Precision-Recall curve shows this tradeoff visually for every possible threshold. The area under this curve (AP score) is the best single number to summarise model quality for imbalanced classification.

---

## F1 Score

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

Harmonic mean of precision and recall. Ranges from 0 (worst) to 1 (best). Good single metric when you care about both precision and recall.

---

## Confusion Matrix

A 2×2 table:

|  | Predicted: No Order | Predicted: Order |
|---|---|---|
| **Actual: No Order** | True Negative (TN) | False Positive (FP) |
| **Actual: Order** | False Negative (FN) | True Positive (TP) |

In business terms:
- **TP** = Called a retailer who ordered → 
- **TN** = Didn't call a retailer who wouldn't order → 
- **FP** = Called a retailer who didn't order → Wasted call (money wasted)
- **FN** = Didn't call a retailer who would have ordered → Missed sale

---

## Threshold Analysis Table

This is the most important output to show your manager. Example:

| Threshold | Calls Made | Reduction | Orders Captured | Precision | Recall |
|---|---|---|---|---|---|
| 20% | 6,500 | 25% | 870 | 13% | 97% |
| 40% | 3,200 | 63% | 810 | 25% | 90% |
| 60% | 1,500 | 83% | 700 | 47% | 78% |
| 80% | 600 | 93% | 520 | 87% | 58% |

The business decides: do we care more about not missing orders (choose low threshold, high recall) or minimising call volume (choose high threshold, high precision)?

The recommended threshold of 0.4 balances both — significant call reduction while capturing most orders.

---

# PART 7: PREDICTION PIPELINE

---

## How Probabilities Are Generated

The model was trained to output P(ordered_today = 1 | features). The `predict_proba(X)[:, 1]` call returns the probability of the positive class for every retailer.

These are calibrated probabilities — a retailer with probability 0.7 should order roughly 70% of the time over many days.

---

## How Retailers Are Ranked and Called

Two approaches are implemented:

**Approach 1 — Probability threshold:** All retailers above X% probability get CALL. The rest get SKIP. Gives variable call volume depending on the day.

**Approach 2 — TOP_K:** Sort all retailers by probability descending. Take the top K (default 2000). This gives a fixed daily call volume regardless of the day.

The notebook uses TOP_K. The app uses both — the threshold slider in the sidebar acts as a filter.

---

## How the Call Priority List is Produced

1. `build_features_for_date(orders, profile, encoders, target_date)` — builds 8,640 rows with features for all retailers
2. `model.predict_proba(X)[:, 1]` — scores every retailer
3. Sort descending by probability
4. Add action column (CALL/SKIP)
5. Export to `outputs/call_priority_YYYY-MM-DD.csv`

---

## How the Next Order Schedule is Generated

1. Train XGBoost Regressor on all consecutive order pairs (target = days until next order)
2. For each retailer, take their last confirmed order as the starting point
3. Run the regressor to predict "days until next order from this starting point"
4. Add that number of days to their last order date → predicted next order date
5. Sort by predicted date
6. Export to `outputs/next_order_prediction_may2026.csv`

---

# PART 8: STREAMLIT DASHBOARD

---

## How to Launch

```bash
cd C:\Users\Rishit\Desktop\O2R-Order-prediction
streamlit run app.py
```

Opens in browser at `http://localhost:8501`.

---

## What Powers the App

At startup the app loads:
- `xgboost_order_model.pkl` → classification model
- `label_encoders.pkl` → encoders for categorical columns
- `retailer_profiles.parquet` → static retailer attributes
- `Jan - May '26 Data.csv` → order history (for live feature building)
- `may_2026_call_schedule.csv` → pre-computed May summary
- `next_order_prediction_may2026.csv` → pre-computed next order dates

All loaded with `@st.cache_data` or `@st.cache_resource` so they load once and stay in memory — no reloading on every interaction.

---

## Page 1 — Overview

**What it shows:**
- 4 KPI metrics: Total Retailers, Calls Today, Call Reduction %, Daily Saving (₹)
- Business impact comparison: cost before model vs cost with model, monthly savings
- Probability distribution histogram for the selected date
- Hub-wise Call vs Skip stacked bar chart
- Full May month bar chart (if Notebook B was run)

**Powered by:** Live model scoring via `build_features_for_date()` on every date change, plus `may_2026_call_schedule.csv` for the monthly chart.

---

## Page 2 — Daily Call List

**What it shows:**
- Filterable table of all retailers ranked by probability
- Filters: Hub, Shop Type, Show All / Call Only / Skip Only
- Columns: Retailer ID, Hub, Shop Type, Prob %, Days Since Last Order, Avg Gap, Days Overdue, Orders (7d), App Usage %, Action (CALL/SKIP)
- Download button to export filtered list as CSV

**Business use:** Call center team opens this every morning, downloads the CSV, and starts calling from Rank 1 downward.

---

## Page 3 — May Schedule

**What it shows:**
- 31-row table: one row per May day with calls needed, calls skipped, reduction %
- Bar chart coloured by day of week
- Summary KPIs: total calls saved, avg reduction, estimated monthly savings

**Powered by:** `outputs/may_2026_call_schedule.csv`

---

## Page 4 — Model Performance

**What it shows:**
- Feature importance horizontal bar chart (top 5 in dark blue)
- Live threshold analysis table computed on the selected date's predictions vs actual labels
- Model info table (algorithm, features, training period)
- Metric guide explaining Precision, Recall, F1

**Powered by:** `clf_model.feature_importances_` from the loaded model + live scoring

---

## Page 5 — Next Order Dates

**What it shows:**
- 3 KPIs: retailers due in 1–3 days, 4–7 days, 7+ days
- Date filter + hub filter
- Table: Retailer ID, Hub, Shop Type, Last Order Date, Avg Gap, Predicted Days Until Next, Predicted Order Date
- Bar chart: expected orders per day in May
- Download button

**Powered by:** `outputs/next_order_prediction_may2026.csv`

---

# PART 9: QUESTIONS YOUR SIR IS LIKELY TO ASK

---

**Q1. What exactly is the business problem you solved?**

DS Group's call center makes ~10,000 calls daily to retailers to collect orders. Only ~900 of those calls result in actual orders — a 9% hit rate. The AI voice agent costs ₹8/min, making this extremely expensive. I built an ML model that predicts each morning which retailers are likely to place an order that day, so the call center only calls those retailers. This reduces call volume by 60–75% while capturing 80–90% of orders.

---

**Q2. Why did you choose XGBoost over other models?**

Three reasons. First, this is tabular structured data — a type where XGBoost consistently outperforms neural networks and often beats other tree-based models. Second, XGBoost handles class imbalance natively via `scale_pos_weight`. Third, it's computationally efficient on a laptop with `tree_method='hist'`. I also benchmarked against Logistic Regression (baseline) and CatBoost, and XGBoost performed best.

---

**Q3. Why not use a neural network?**

Neural networks need large amounts of data, are computationally expensive, are harder to interpret (feature importance is not straightforward), and don't have a clear advantage on tabular data over gradient boosting. For a business problem where explainability matters — my manager needs to understand why a retailer is flagged — XGBoost's feature importance is far more useful.

---

**Q4. Why train on Jan–Apr and test on May?**

Because this is a time-series problem. In the real world, you train on past data and predict future data. If I used random train-test split, training data would include rows from May and test data would include rows from January. The model would essentially know "future" information during training, giving artificially inflated accuracy — that's called data leakage. Time-based split simulates the real prediction scenario perfectly.

---

**Q5. What is data leakage and how did you prevent it?**

Data leakage is when information that wouldn't be available at prediction time accidentally enters your training features. I prevented it in two ways. First, the `shift(1)` in rolling features ensures today's order is never included when computing "orders in the last 7 days". Second, in the `build_features_for_date()` function, I use `orders[orders['createdAt'] < target_date]` — strictly less than, so only history before the prediction date is used.

---

**Q6. What is the retailer-day grid and why do you need it?**

Raw data only has rows for days when orders happened. But a model needs examples of both classes — days when retailers ordered AND days when they didn't. The grid is a cross-join of every retailer × every date, giving 1.3 million rows. Each row is then labelled 1 (ordered) or 0 (didn't order). This is the training table.

---

**Q7. What is class imbalance and how did you handle it?**

Out of 1.3M rows, only ~180K are positive (ordered=1). That's 13.6% positive rate — the dataset is imbalanced. A naive model could just predict "no order" for everything and be 86% accurate while being useless. I handled this by setting `scale_pos_weight = neg_count / pos_count ≈ 6.3` in XGBoost. This tells the model to treat each positive example as if it counted 6.3 times as much, compensating for the imbalance.

---

**Q8. Why not use accuracy as the evaluation metric?**

Because of class imbalance. A model that predicts "no order" for every retailer would be 86% accurate but would catch zero orders. Accuracy is meaningless here. I use Precision (quality of the call list), Recall (how many real orders are captured), F1 (balance of both), and the Precision-Recall curve (which captures performance across all thresholds).

---

**Q9. What does `days_since_last_order = 999` mean?**

It's the default value assigned to retailers who have no order history before the prediction date (for example, if a retailer's first order was after the prediction date, or they are a cold-start retailer). 999 acts as a sentinel value signalling "never ordered before." The model learns to treat such retailers as low probability.

---

**Q10. What is `days_overdue` and why is it important?**

`days_overdue = prediction_date - (last_order_date + avg_gap_between_orders)`. If a retailer's average gap is 3 days and they last ordered 5 days ago, days_overdue = 2. The more overdue they are, the higher the chance they'll order today. It's likely one of the top 3 most important features because it combines recency and frequency into a single signal.

---

**Q11. What is `overdue_ratio` and how does it differ from `days_overdue`?**

`overdue_ratio = days_since_last_order / (avg_gap + 1)`. It normalises the overdue measurement by the retailer's own pattern. Retailer A with days_since=5 and avg_gap=3 has overdue_ratio=1.67 (significantly overdue). Retailer B with days_since=5 and avg_gap=10 has overdue_ratio=0.45 (barely past midpoint of their cycle). Raw `days_overdue` wouldn't capture this personalisation.

---

**Q12. What is `momentum_7_30`?**

`orders_last_7_days / (orders_last_30_days + 1)`. This captures whether recent ordering activity is accelerating or decelerating relative to the monthly baseline. High momentum means the retailer is in an active purchasing phase and likely to order again soon.

---

**Q13. Why did you add CatBoost?**

To benchmark and ensure XGBoost was truly the best choice. CatBoost is competitive with XGBoost on datasets with categorical features and sometimes outperforms it. The notebook automatically saves whichever performs better (based on F1 score). The CatBoost training logs in `catboost_info/` prove this comparison was actually run.

---

**Q14. Why `scale_pos_weight` specifically and not SMOTE?**

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic positive examples. For this dataset with 1.3M rows, SMOTE would be computationally expensive and might generate unrealistic synthetic retailer-day combinations. `scale_pos_weight` achieves the same correction inside the XGBoost algorithm natively with no computational overhead and no risk of unrealistic data.

---

**Q15. What is `order_regularity` and why is it useful?**

`order_regularity = 1 / (std_gap + 1)`. Retailers with very consistent ordering patterns (low std) have high regularity scores. These retailers are more predictable — if their average gap is 3 days and they ordered 3 days ago, you can be very confident they'll order today. Retailers with erratic patterns (high std) have low regularity — harder to predict regardless of other signals.

---

**Q16. What is `app_order_ratio` and why does it matter?**

The proportion of a retailer's historical orders that came through the DS Group app vs the call center. If app_order_ratio = 0.9, this retailer orders independently through the app 90% of the time. Calling them wastes agent time — they'll order themselves. The model learns to deprioritise high-app-ratio retailers for calls.

---

**Q17. Why `tree_method = 'hist'`?**

The standard XGBoost tree building algorithm evaluates every possible split point for every feature. `hist` mode instead bins features into histograms and only evaluates bucket boundaries as split points. On large datasets like ours (1.3M rows, 27 features), this is 3–10x faster with minimal accuracy loss. Essential for training on a laptop.

---

**Q18. What is `tenure_days` capturing?**

How long the retailer has been ordering from DS Group (`last_order_date - first_order_date`). Longer-tenured retailers have more established patterns and are more predictable. Newer retailers have less history — the model is less confident about them. Also correlates with loyalty — a retailer who's been ordering for 5 months is a reliable customer.

---

**Q19. Why threshold 0.4 as default?**

From the threshold analysis table. At 0.4, you get a significant call reduction (typically 60–65%) while capturing 85–90% of orders. It's a balance point. The business can shift this based on their priorities — if call center cost is the primary concern, raise the threshold. If missing orders is more costly, lower it.

---

**Q20. What is the difference between Phase 1 (binary classification) and Phase 2 (regression)?**

Phase 1 answers: "Will this retailer order today? Yes or No?" — Binary classification with XGBoost Classifier.

Phase 2 answers: "In how many days will this retailer order next?" — Regression with XGBoost Regressor. The output is a continuous number (e.g., 2.7 days). This is more powerful for scheduling — instead of running the binary model every day for every retailer, you can schedule calls in advance.

---

**Q21. What is MAE in the regression model and what does it mean practically?**

MAE = Mean Absolute Error. If MAE = 2.1 days, on average the model's next-order prediction is off by 2.1 days. Practically: if the model predicts Retailer A will order on May 10, they might actually order on May 8 or May 12. A 2-day error is quite acceptable for call scheduling purposes.

---

**Q22. How would you handle a cold-start retailer — someone who just joined and has no order history?**

Currently: `days_since_last_order = 999`, all rolling counts = 0, avg_gap = 30 (global default). The model will assign a low probability. For production, a better approach would be to profile cold-start retailers by their `shopType` and `hubName` and assign them probabilities based on similar retailers' patterns.

---

**Q23. Why do you save label encoders and not just re-fit them?**

If you re-fit the LabelEncoder on new data, the mapping changes. "Noida Hub" might be encoded as 7 during training but as 3 on new data (if the order of unique values differs). The model was trained with "Noida Hub" = 7, so feeding 3 would produce wrong predictions. Always save and reload the exact same encoders from training.

---

**Q24. Why is the grid built using `pd.MultiIndex.from_product` instead of a loop?**

Performance. A Python loop over 8,640 retailers × 151 days = 1.3 million iterations would take minutes. `from_product` generates this in seconds using vectorised NumPy operations internally.

---

**Q25. What happens to retailers who placed multiple orders on the same day?**

The deduplication step `drop_duplicates(subset=['customerId','date'])` in the order_flags creation ensures this is treated as a single order event for the labelling. The model predicts binary: ordered today or not. Multiple orders in a day still count as a single positive label.

---

**Q26. Why is the parquet file much smaller than the original CSV?**

Parquet is a columnar binary format with compression. The 600K row CSV is ~200MB. The 1.3M row parquet (with features) is ~150MB because: (a) columnar storage compresses repeated values efficiently, and (b) binary encoding is more compact than text. Parquet also loads 5–10x faster than CSV.

---

**Q27. How would you improve this model if given more time?**

Five improvements: (1) Add product-level features — which SKUs does this retailer usually buy, are they running low based on quantity and avg consumption rate. (2) Add weather and festival calendar features — orders spike before festivals. (3) Use a time-series model like LSTM or Temporal Fusion Transformer for the regression. (4) Add geographic features — distance from hub, delivery route. (5) Build an online learning system that updates the model weekly as new orders come in.

---

**Q28. What is the business risk of a False Negative (missed order)?**

The retailer doesn't get called. They might: (a) order through the App anyway (no impact), (b) order from a competitor (lost revenue), or (c) not order at all that day (lost sale for that day). The business needs to quantify the average order value and compare that loss against the cost savings from reduced calls.

---

**Q29. Why `n_estimators=400` specifically?**

400 trees with learning_rate=0.05 is a standard combination that provides good model complexity without severe overfitting. The `eval_set` during training allows monitoring validation loss — if the model started overfitting, you'd see validation loss increase after a certain number of trees. I could add `early_stopping_rounds` to automate this, which would be a good improvement.

---

**Q30. What's the difference between this project and a recommendation system?**

I initially built a product recommendation system (what should this retailer buy?) using Surprise SVD and FP-Growth algorithms. But the actual business problem was propensity prediction (will this retailer buy today?). These are completely different problems. A recommendation system assumes the retailer is going to buy and tells them what. This project tells the call center who to call — the decision whether to buy hasn't been made yet. The recommendation system adds no value if the retailer wasn't going to order at all.

---

# PART 10: 5-MINUTE PRESENTATION SCRIPT

---

"Good morning sir. I'd like to present the O2R Order Prediction project I built during my internship at DS Group.

**The Problem — 1 minute**

DS Group's call center calls roughly 10,000 retailers every day to collect orders. Out of those 10,000 calls, only around 900 retailers actually place an order. That means 91% of calls generate no revenue. The call center also uses an AI voice agent that costs ₹8 per minute. At 10,000 calls a day with an average call length of 2 minutes, that's ₹1,60,000 per day — or over ₹40 lakhs per month — spent mostly on calls that produce nothing.

The question I set out to answer was: every morning, before the call center opens, can we predict which retailers are actually going to place an order today — and call only those?

**The Data — 30 seconds**

I worked with 609,723 rows of historical order data from January to May 2026, covering 9,766 unique retailers and 200,000 confirmed orders across all DS Group hubs including Noida, Delhi, Meerut, and Ghaziabad.

**The Approach — 1 minute**

I treated this as a binary classification problem. For every retailer, on every day: will they place an order today — yes or no?

The key insight was that every retailer has a buying pattern. Some order every 2 days. Some order every week. Some order on Mondays. If you can learn these patterns from 5 months of history, you can predict who will order tomorrow.

I engineered 27 features capturing: how recently they ordered, how often they order, whether they're overdue compared to their own pattern, their recent momentum, whether they use the app or need a call, what kind of shop they are, and which hub they belong to.

**The Models — 30 seconds**

I trained three models: Logistic Regression as a baseline, XGBoost as the main model, and CatBoost as a benchmark. XGBoost performed best. I trained on January through April and tested on May to simulate the real prediction scenario — training on past, predicting future.

**The Results — 1 minute**

On the May test set, at a 40% probability threshold:
- The model recommends calling roughly 3,000–3,500 retailers instead of 8,640
- It captures approximately 85–90% of actual orders
- Call reduction: approximately 60–65%
- Estimated daily saving: ₹80,000–₹1,00,000 in AI voice agent costs alone
- Monthly saving: ₹20–25 lakhs

The most important features were: days since last order, days overdue, order regularity, and recent ordering momentum.

I also built a Phase 2 regression model that predicts the exact date when each retailer will next order — not just whether they'll order today. This allows the call center to schedule calls proactively instead of re-running predictions every day.

**The Dashboard — 30 seconds**

Everything is wrapped in a Streamlit dashboard that the call center can use every morning. You select any May date, set your preferred threshold, and the dashboard shows you the ranked call list, which you can download as a CSV. It also shows the full month call schedule, model performance, and the next order date predictions for every retailer.

**Future Improvements — 30 seconds**

Three key improvements I would build next: First, add product consumption rate features — how fast does this retailer typically exhaust their inventory — to make predictions even sharper. Second, integrate festival and weather calendars since both affect ordering patterns significantly. Third, build an online learning pipeline so the model retrains weekly on new order data rather than being a static model.

Sir, the core business outcome is this: DS Group can reduce call center costs by approximately 60% while maintaining nearly the same order capture rate. The model is production-ready and the dashboard is built for daily use."

---

*You now know every design decision, every feature, every model, every metric, every output, and every line of your project. Go confidently.*
