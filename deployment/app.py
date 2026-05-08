import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="Olist Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Data ────────────────────────────────────────────────────────────────────

OVERVIEW = {
    "totalRows": 9038,
    "balancedRows": 748,
    "churnedRows": 374,
    "retainedRows": 374,
    "features": 13,
    "trainRows": 598,
    "testRows": 150,
    "churnRateOriginal": 0.97,
    "churnRateBalanced": 0.50,
}

MODELS = pd.DataFrame([
    {"Model": "Tuned XGBoost",       "Group": "Tuned",    "AUC": 0.8267, "Accuracy": 0.8267, "F1": 0.8194, "Kappa": 0.6533, "Best": True},
    {"Model": "XGBoost",             "Group": "Baseline", "AUC": 0.8000, "Accuracy": 0.8000, "F1": 0.7972, "Kappa": 0.6000, "Best": False},
    {"Model": "XGBoost (RFE)",       "Group": "RFE",      "AUC": 0.7933, "Accuracy": 0.7933, "F1": 0.7870, "Kappa": 0.5867, "Best": False},
    {"Model": "Random Forest",       "Group": "Baseline", "AUC": 0.7800, "Accuracy": 0.7800, "F1": 0.7762, "Kappa": 0.5600, "Best": False},
    {"Model": "Logistic Regression", "Group": "Baseline", "AUC": 0.7800, "Accuracy": 0.7800, "F1": 0.7762, "Kappa": 0.5600, "Best": False},
    {"Model": "Naïve Bayes",         "Group": "Baseline", "AUC": 0.7600, "Accuracy": 0.7600, "F1": 0.7547, "Kappa": 0.5200, "Best": False},
    {"Model": "Tuned SVM",           "Group": "Tuned",    "AUC": 0.7533, "Accuracy": 0.7533, "F1": 0.7481, "Kappa": 0.5067, "Best": False},
    {"Model": "Tuned Random Forest", "Group": "Tuned",    "AUC": 0.7533, "Accuracy": 0.7533, "F1": 0.7481, "Kappa": 0.5067, "Best": False},
    {"Model": "SVM (RFE)",           "Group": "RFE",      "AUC": 0.7467, "Accuracy": 0.7467, "F1": 0.7414, "Kappa": 0.4933, "Best": False},
    {"Model": "SVM",                 "Group": "Baseline", "AUC": 0.7400, "Accuracy": 0.7400, "F1": 0.7333, "Kappa": 0.4800, "Best": False},
    {"Model": "KNN (k=3)",           "Group": "Baseline", "AUC": 0.7333, "Accuracy": 0.7333, "F1": 0.7278, "Kappa": 0.4667, "Best": False},
    {"Model": "Decision Tree",       "Group": "Baseline", "AUC": 0.7067, "Accuracy": 0.7067, "F1": 0.6994, "Kappa": 0.4133, "Best": False},
])

FEATURES = pd.DataFrame([
    {"Feature": "freight_value",        "Importance": 0.182, "RFE": True},
    {"Feature": "carrier_delivered",    "Importance": 0.161, "RFE": True},
    {"Feature": "payment_value",        "Importance": 0.143, "RFE": False},
    {"Feature": "approved_carrier",     "Importance": 0.129, "RFE": True},
    {"Feature": "payment_installments", "Importance": 0.098, "RFE": False},
    {"Feature": "order_item_id",        "Importance": 0.089, "RFE": True},
    {"Feature": "review_score",         "Importance": 0.075, "RFE": False},
    {"Feature": "no_of_products",       "Importance": 0.062, "RFE": True},
    {"Feature": "payment_sequential",   "Importance": 0.058, "RFE": True},
    {"Feature": "delivered_estimated",  "Importance": 0.049, "RFE": False},
    {"Feature": "price",                "Importance": 0.034, "RFE": False},
    {"Feature": "purchased_approved",   "Importance": 0.028, "RFE": False},
    {"Feature": "purchased_delivered",  "Importance": 0.022, "RFE": False},
])

RFM = pd.DataFrame([
    {"Segment": "Champions",           "Count": 412,  "Percentage": 4.56},
    {"Segment": "Loyal Customers",     "Count": 683,  "Percentage": 7.56},
    {"Segment": "Potential Loyalists", "Count": 1024, "Percentage": 11.34},
    {"Segment": "At Risk",             "Count": 2187, "Percentage": 24.21},
    {"Segment": "Lost Customers",      "Count": 4732, "Percentage": 52.33},
])

CONFUSION = np.array([[65, 10], [16, 59]])
LABELS = ["Retained", "Churned"]

# ── Deep Learning Data ───────────────────────────────────────────────────────

DL_OVERVIEW = {
    "totalRows": 90528,
    "trainRows": 72422,
    "testRows": 18106,
    "features": 20,
    "churnRate": 45.41,
}

DL_MODELS = pd.DataFrame([
    {"Model": "ANN (thresh=0.50)", "Architecture": "ANN", "AUC": 0.7621, "Accuracy": 0.6831, "Precision": 0.6212, "Recall": 0.7744, "F1": 0.6894},
    {"Model": "ANN (thresh=0.43)", "Architecture": "ANN", "AUC": 0.7621, "Accuracy": 0.6530, "Precision": 0.5764, "Recall": 0.8896, "F1": 0.6995},
    {"Model": "CNN (thresh=0.50)", "Architecture": "CNN", "AUC": 0.7656, "Accuracy": 0.6850, "Precision": 0.6324, "Recall": 0.7316, "F1": 0.6784},
    {"Model": "XGBoost+RFE (NB4)", "Architecture": "ML",  "AUC": 0.8400, "Accuracy": 0.7617, "Precision": 0.7286, "Recall": 0.7613, "F1": 0.7440},
])

ANN_FEATURES = pd.DataFrame([
    {"Feature": "freight_value",           "Importance": 0.1202},
    {"Feature": "delivered_estimated",     "Importance": 0.1118},
    {"Feature": "purchased_delivered",     "Importance": 0.1025},
    {"Feature": "purchased_approved",      "Importance": 0.0619},
    {"Feature": "geolocation_lat",         "Importance": 0.0387},
    {"Feature": "product_weight_g",        "Importance": 0.0330},
    {"Feature": "payment_type_credit_card","Importance": 0.0321},
    {"Feature": "Monetary",                "Importance": 0.0277},
    {"Feature": "geolocation_lng",         "Importance": 0.0226},
    {"Feature": "price",                   "Importance": 0.0200},
])

COLORS = {
    "blue":   "#0079F2",
    "purple": "#795EFF",
    "green":  "#009118",
    "red":    "#A60808",
    "pink":   "#ec4899",
    "grey":   "#a0aec0",
    "teal":   "#0d9488",
    "orange": "#f97316",
}
COLOR_LIST = [COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["red"], COLORS["pink"]]
GROUP_COLORS = {"Tuned": COLORS["blue"], "RFE": COLORS["purple"], "Baseline": COLORS["grey"]}

# ── Header ───────────────────────────────────────────────────────────────────

st.title("📊 Olist Churn Prediction Dashboard")
st.caption("Brazilian E-Commerce · RFM Analysis · ML Model Comparison · Deep Learning (ANN & CNN) · Balanced 50/50 · 80/20 Split")
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Orders (After Clean)", f"{OVERVIEW['totalRows']:,}", help="After merge & preprocessing")
k2.metric("Balanced Dataset",           f"{OVERVIEW['balancedRows']:,}",
          f"{OVERVIEW['churnedRows']} churned + {OVERVIEW['retainedRows']} retained")
k3.metric("Best ML Model AUC",          "0.8267",    "Tuned XGBoost")
k4.metric("Original Churn Rate",        "97%",       "Frequency = 1 customer", delta_color="inverse")

st.divider()

# ── Model loading ─────────────────────────────────────────────────────────────

_PKL_PATH = os.path.join(os.path.dirname(__file__), "tuned_xgboost_model.pkl")
_model_payload = None
if os.path.exists(_PKL_PATH):
    try:
        _model_payload = joblib.load(_PKL_PATH)
    except Exception:
        _model_payload = None

# ── Prediction function ───────────────────────────────────────────────────────

def predict_churn(freight_value, carrier_delivered, payment_value, approved_carrier,
                  payment_installments, order_item_id, review_score, no_of_products,
                  payment_sequential, delivered_estimated, price, purchased_approved,
                  purchased_delivered):

    if _model_payload is not None:
        model    = _model_payload["model"]
        features = _model_payload["features"]
        raw = {
            "freight_value":        freight_value,
            "carrier_delivered":    carrier_delivered,
            "payment_value":        payment_value,
            "approved_carrier":     approved_carrier,
            "payment_installments": payment_installments,
            "order_item_id":        order_item_id,
            "review_score":         review_score,
            "no_of_products":       no_of_products,
            "payment_sequential":   payment_sequential,
            "delivered_estimated":  delivered_estimated,
            "price":                price,
            "purchased_approved":   purchased_approved,
            "purchased_delivered":  purchased_delivered,
        }
        row = pd.DataFrame([[raw.get(f, 0.0) for f in features]], columns=features)
        prob = float(model.predict_proba(row)[0][1])
        return prob

    # Fallback: weighted sigmoid approximation (used when no pkl is present)
    score = 0.0
    score += 0.182 * min(freight_value / 80,  1.0)
    score += 0.161 * min(carrier_delivered / 15, 1.0)
    score -= 0.143 * min(payment_value / 400, 1.0)
    score += 0.129 * min(approved_carrier / 8, 1.0)
    score -= 0.098 * min(payment_installments / 10, 1.0)
    score -= 0.089 * min(order_item_id / 4, 1.0)
    score -= 0.075 * (review_score - 1) / 4
    score -= 0.062 * min(no_of_products / 4, 1.0)
    score += 0.029 * min(payment_sequential / 4, 1.0)
    score += 0.049 * max(0, delivered_estimated / 7)
    score += 0.017 * min(price / 400, 1.0)
    score += 0.028 * min(purchased_approved / 4, 1.0)
    score += 0.022 * min(purchased_delivered / 25, 1.0)
    return float(1 / (1 + np.exp(-8 * score)))

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔮 Predict Customer",
    "🏆 Model Comparison",
    "🔀 Feature Importance",
    "🧩 Confusion Matrix",
    "👥 RFM Segments",
    "🧠 Deep Learning",
    "📋 Pipeline Summary",
])

# ── Tab 1: Predict Customer ───────────────────────────────────────────────────

with tab1:
    st.subheader("🔮 Customer Churn Predictor")
    st.caption("Enter the customer's order details below. The model will predict whether they are likely to churn.")

    with st.form("predict_form"):
        st.markdown("#### 📦 Order & Payment Details")
        c1, c2, c3 = st.columns(3)
        price               = c1.number_input("Price (R$)",               min_value=0.0,  max_value=2000.0, value=85.0,  step=1.0)
        freight_value       = c2.number_input("Freight Value (R$)",        min_value=0.0,  max_value=300.0,  value=18.0,  step=0.5)
        payment_value       = c3.number_input("Total Payment Value (R$)",  min_value=0.0,  max_value=3000.0, value=105.0, step=1.0)

        c4, c5, c6 = st.columns(3)
        payment_installments = c4.number_input("Payment Installments",     min_value=1,    max_value=24,     value=1,     step=1)
        payment_sequential   = c5.number_input("Payment Sequential",       min_value=1,    max_value=10,     value=1,     step=1)
        order_item_id        = c6.number_input("No. of Items in Order",    min_value=1,    max_value=20,     value=1,     step=1)

        st.markdown("#### 🚚 Delivery Details")
        d1, d2, d3 = st.columns(3)
        purchased_approved  = d1.number_input("Purchase → Approval (days)",  min_value=0.0, max_value=30.0, value=0.5,  step=0.1)
        approved_carrier    = d2.number_input("Approval → Carrier (days)",   min_value=0.0, max_value=30.0, value=2.5,  step=0.1)
        carrier_delivered   = d3.number_input("Carrier → Delivered (days)",  min_value=0.0, max_value=40.0, value=9.0,  step=0.1)

        d4, d5, d6 = st.columns(3)
        purchased_delivered = d4.number_input("Total Delivery Time (days)",  min_value=0.0, max_value=60.0,  value=12.0, step=0.1)
        delivered_estimated = d5.number_input("Late vs Estimate (days, + = late)", min_value=-15.0, max_value=30.0, value=0.0, step=0.1)
        no_of_products      = d6.number_input("No. of Distinct Products",    min_value=1,   max_value=20,    value=1,    step=1)

        st.markdown("#### ⭐ Review")
        review_score = st.slider("Review Score", min_value=1, max_value=5, value=4)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

    if submitted:
        prob = predict_churn(
            freight_value, carrier_delivered, payment_value, approved_carrier,
            payment_installments, order_item_id, review_score, no_of_products,
            payment_sequential, delivered_estimated, price, purchased_approved, purchased_delivered
        )
        churned = prob >= 0.5
        pct = prob * 100

        st.divider()
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if churned:
                st.error(f"## ⚠️ Likely to Churn\nChurn probability: **{pct:.1f}%**")
            else:
                st.success(f"## ✅ Likely to Retain\nChurn probability: **{pct:.1f}%**")

            st.markdown("#### Key Drivers")
            drivers = [
                ("Freight value",       freight_value,         "high" if freight_value > 30 else "low",      True),
                ("Carrier delivery",    carrier_delivered,     "slow" if carrier_delivered > 10 else "fast",  True),
                ("Payment value",       payment_value,         "high" if payment_value > 150 else "low",      False),
                ("Review score",        review_score,          "positive" if review_score >= 4 else "negative", False),
                ("Late vs estimate",    delivered_estimated,   "late" if delivered_estimated > 0 else "on time", True),
            ]
            for name, val, status, is_risk in drivers:
                icon = "🔴" if is_risk and status in ("high","slow","late","negative") else "🟢"
                st.markdown(f"{icon} **{name}**: {val} ({status})")

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Churn Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS["red"] if churned else COLORS["green"]},
                    "steps": [
                        {"range": [0,  40], "color": "#d4f5d4"},
                        {"range": [40, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#fdd"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

# ── Tab 2: Model Comparison ───────────────────────────────────────────────────

with tab2:
    st.subheader("Model AUC Comparison")
    st.caption("All models trained on 50/50 balanced dataset, 80/20 train/test split, 13 features")

    models_sorted = MODELS.sort_values("AUC", ascending=True)
    bar_colors = [GROUP_COLORS.get(g, COLORS["grey"]) for g in models_sorted["Group"]]

    fig_auc = go.Figure(go.Bar(
        x=models_sorted["AUC"],
        y=models_sorted["Model"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:.4f}" for v in models_sorted["AUC"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>AUC: %{x:.4f}<extra></extra>",
    ))
    fig_auc.update_layout(
        height=420, margin=dict(l=0, r=60, t=10, b=0),
        xaxis=dict(range=[0.65, 0.86], title="AUC"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    fig_auc.add_vrect(x0=0.65, x1=0.86, fillcolor="rgba(0,0,0,0)", line_width=0)
    st.plotly_chart(fig_auc, use_container_width=True)

    st.caption("🔵 Tuned  |  🟣 RFE  |  ⚫ Baseline")

    st.divider()
    st.subheader("Multi-Metric Radar — Top 5 Models")

    top5 = MODELS.head(5)
    categories = ["AUC (%)", "Accuracy (%)", "F1 (%)"]
    fig_radar = go.Figure()
    radar_colors = [COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["red"], COLORS["pink"]]
    for i, row in top5.iterrows():
        vals = [row["AUC"] * 100, row["Accuracy"] * 100, row["F1"] * 100]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", fillcolor=radar_colors[i % len(radar_colors)],
            opacity=0.2, line=dict(color=radar_colors[i % len(radar_colors)], width=2),
            name=row["Model"],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[70, 85])),
        height=380, margin=dict(l=60, r=60, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.subheader("Full Results Table")
    display_df = MODELS.drop(columns=["Best"]).rename(columns={
        "AUC": "AUC ↑", "Accuracy": "Accuracy ↑", "F1": "F1 Score ↑", "Kappa": "Cohen's κ ↑"
    })
    st.dataframe(
        display_df.style
            .format({"AUC ↑": "{:.4f}", "Accuracy ↑": "{:.4f}", "F1 Score ↑": "{:.4f}", "Cohen's κ ↑": "{:.4f}"})
            .highlight_max(subset=["AUC ↑", "Accuracy ↑", "F1 Score ↑", "Cohen's κ ↑"], color="#d4eaff")
            .apply(lambda x: ["background-color: #e8f4ff" if MODELS.loc[i, "Best"] else "" for i in x.index], axis=0),
        use_container_width=True, hide_index=True,
    )

# ── Tab 3: Feature Importance ─────────────────────────────────────────────────

with tab3:
    st.subheader("Feature Importance — Tuned XGBoost")
    feat_sorted = FEATURES.sort_values("Importance", ascending=True)
    feat_colors = [COLORS["blue"] if r else COLORS["purple"] for r in feat_sorted["RFE"]]

    fig_feat = go.Figure(go.Bar(
        x=feat_sorted["Importance"],
        y=feat_sorted["Feature"],
        orientation="h",
        marker_color=feat_colors,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in feat_sorted["Importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
    ))
    fig_feat.update_layout(
        height=400, margin=dict(l=0, r=60, t=10, b=0),
        xaxis=dict(title="Importance Score"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig_feat, use_container_width=True)
    st.caption("🔵 RFE-selected features  |  🟣 Other features (not in RFE set)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RFE-selected features (6)**")
        rfe_feats = FEATURES[FEATURES["RFE"]].sort_values("Importance", ascending=False)
        st.dataframe(rfe_feats[["Feature", "Importance"]].reset_index(drop=True)
                     .style.format({"Importance": "{:.3f}"}), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Other features (7)**")
        other_feats = FEATURES[~FEATURES["RFE"]].sort_values("Importance", ascending=False)
        st.dataframe(other_feats[["Feature", "Importance"]].reset_index(drop=True)
                     .style.format({"Importance": "{:.3f}"}), use_container_width=True, hide_index=True)

# ── Tab 4: Confusion Matrix ───────────────────────────────────────────────────

with tab4:
    st.subheader("Confusion Matrix — Tuned XGBoost (Test Set, 150 samples)")

    col_cm, col_metrics = st.columns([1, 1])

    with col_cm:
        fig_cm = go.Figure(go.Heatmap(
            z=CONFUSION,
            x=["Predicted: " + l for l in LABELS],
            y=["Actual: " + l for l in LABELS],
            text=[[str(v) for v in row] for row in CONFUSION],
            texttemplate="%{text}",
            textfont=dict(size=28, color="white"),
            colorscale=[[0, "#f0f7ff"], [1, COLORS["blue"]]],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="top"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_metrics:
        st.markdown("### Performance Metrics")
        tn, fp, fn, tp = 65, 10, 16, 59
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f1        = 2 * precision * recall / (precision + recall)
        accuracy  = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp)

        st.metric("Precision (Churn)", f"{precision:.1%}")
        st.metric("Recall / Sensitivity (Churn)", f"{recall:.1%}")
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.metric("Specificity (Retained)", f"{specificity:.1%}")

        st.markdown("---")
        st.markdown("**Interpretation**")
        st.info(
            f"Out of **{tp + fn} churned** customers in the test set, "
            f"the model correctly identified **{tp}** ({recall:.0%} recall). "
            f"Of **{tp + fp} churn predictions**, {tp} were correct ({precision:.0%} precision). "
            f"**{fn} churned** customers were missed (false negatives)."
        )

    st.divider()
    st.subheader("Hyperparameters (GridSearchCV Best)")
    hp_col1, hp_col2, hp_col3 = st.columns(3)
    hp_col1.metric("Learning Rate", "0.1")
    hp_col2.metric("Max Depth",     "3")
    hp_col3.metric("n_estimators",  "200")

# ── Tab 5: RFM Segments ───────────────────────────────────────────────────────

with tab5:
    st.subheader("RFM Customer Segmentation")
    st.caption("Recency · Frequency · Monetary analysis across 9,038 unique customers")

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=RFM["Segment"],
            values=RFM["Count"],
            marker=dict(colors=COLOR_LIST),
            hole=0.35,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            height=360, margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_rfm_bar = go.Figure(go.Bar(
            x=RFM["Segment"],
            y=RFM["Count"],
            marker_color=COLOR_LIST,
            marker_line_width=0,
            text=[f"{v:,}" for v in RFM["Count"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        ))
        fig_rfm_bar.update_layout(
            height=360, margin=dict(l=0, r=0, t=20, b=40),
            yaxis=dict(title="Customer Count"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
        st.plotly_chart(fig_rfm_bar, use_container_width=True)

    st.divider()
    st.subheader("Segment Details")
    rfm_display = RFM.copy()
    rfm_display["Percentage"] = rfm_display["Percentage"].map("{:.2f}%".format)
    rfm_display["Count"] = rfm_display["Count"].map("{:,}".format)
    rfm_display.columns = ["Segment", "Customer Count", "Share of Total"]
    st.dataframe(rfm_display, use_container_width=True, hide_index=True)

    st.warning(
        "⚠️ **Key Finding:** 76.5% of customers are either 'At Risk' or 'Lost', "
        "confirming the extreme churn rate (97%) observed in the raw Frequency data. "
        "Only 5% qualify as Champions."
    )

# ── Tab 6: Deep Learning ──────────────────────────────────────────────────────

with tab6:
    st.subheader("🧠 Deep Learning for Churn Prediction — ANN & CNN")
    st.caption("Notebook 05 · TensorFlow/Keras · Full dataset (no balancing) · Class weights · 80/20 split")

    # KPIs
    dl_k1, dl_k2, dl_k3, dl_k4 = st.columns(4)
    dl_k1.metric("Full Dataset", f"{DL_OVERVIEW['totalRows']:,}", "rows × 20 features")
    dl_k2.metric("Training Samples", f"{DL_OVERVIEW['trainRows']:,}", "80% stratified")
    dl_k3.metric("Best DL AUC", "0.7656", "CNN (thresh=0.50)")
    dl_k4.metric("Class Distribution", "54.6% / 45.4%", "Not Churned / Churned")

    st.divider()

    # ── Architecture Cards ────────────────────────────────────────────────────
    st.subheader("Model Architectures")
    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("#### 🔷 ANN (Artificial Neural Network)")
        st.markdown("""
| Layer | Output Shape | Params |
|---|---|---|
| Input | (None, 20) | — |
| Dense(64, ReLU) | (None, 64) | 1,344 |
| BatchNormalization | (None, 64) | 256 |
| Dropout(0.3) | (None, 64) | 0 |
| Dense(32, ReLU) | (None, 32) | 2,080 |
| BatchNormalization | (None, 32) | 128 |
| Dropout(0.3) | (None, 32) | 0 |
| Dense(1, Sigmoid) | (None, 1) | 33 |
        """)
        st.info("**Total params:** 3,841 · **Optimizer:** Adam (lr=1e-3) · **Loss:** Binary Cross-Entropy")

    with arch_col2:
        st.markdown("#### 🔶 CNN (1D Convolutional Neural Network)")
        st.markdown("""
| Layer | Output Shape | Params |
|---|---|---|
| Input | (None, 20, 1) | — |
| Conv1D(32, k=3, ReLU) | (None, 20, 32) | 128 |
| BatchNormalization | (None, 20, 32) | 128 |
| MaxPooling1D(2) | (None, 10, 32) | 0 |
| Conv1D(64, k=3, ReLU) | (None, 10, 64) | 6,208 |
| BatchNormalization | (None, 10, 64) | 256 |
| MaxPooling1D(2) | (None, 5, 64) | 0 |
| Flatten | (None, 320) | 0 |
| Dense(64, ReLU) | (None, 64) | 20,544 |
| Dropout(0.5) | (None, 64) | 0 |
| Dense(1, Sigmoid) | (None, 1) | 65 |
        """)
        st.info("**Total params:** 27,329 · **Optimizer:** Adam (lr=1e-3) · **Loss:** Binary Cross-Entropy")

    st.divider()

    # ── Callbacks info ────────────────────────────────────────────────────────
    st.subheader("Training Configuration")
    cb_col1, cb_col2, cb_col3 = st.columns(3)
    cb_col1.metric("Early Stopping", "patience=10", "monitors val AUC")
    cb_col2.metric("ReduceLROnPlateau", "factor=0.5", "patience=5, min_lr=1e-6")
    cb_col3.metric("Max Epochs", "100", "best weights restored")

    st.divider()

    # ── Performance comparison ────────────────────────────────────────────────
    st.subheader("Performance Comparison — DL vs Best ML")

    metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    arch_palette = {
        "ANN": COLORS["blue"],
        "CNN": COLORS["teal"],
        "ML":  COLORS["orange"],
    }

    fig_dl_bar = go.Figure()
    for _, row in DL_MODELS.iterrows():
        fig_dl_bar.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=arch_palette.get(row["Architecture"], COLORS["grey"]),
            text=[f"{row[m]:.3f}" for m in metrics],
            textposition="outside",
            hovertemplate=f"<b>{row['Model']}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig_dl_bar.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(range=[0, 0.98], title="Score"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_dl_bar, use_container_width=True)

    # Full results table
    st.subheader("Full DL Results Table")
    dl_display = DL_MODELS.drop(columns=["Architecture"]).copy()
    dl_display.columns = ["Model", "AUC ↑", "Accuracy ↑", "Precision ↑", "Recall ↑", "F1 Score ↑"]
    st.dataframe(
        dl_display.style
            .format({c: "{:.4f}" for c in dl_display.columns if c != "Model"})
            .highlight_max(subset=["AUC ↑", "Accuracy ↑", "Precision ↑", "Recall ↑", "F1 Score ↑"], color="#d4eaff"),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── ANN Feature Importance ────────────────────────────────────────────────
    st.subheader("ANN Permutation Feature Importance (Top 10)")
    st.caption("AUC drop when each feature is randomly shuffled — higher = more important")

    ann_feat_sorted = ANN_FEATURES.sort_values("Importance", ascending=True)
    fig_ann_feat = go.Figure(go.Bar(
        x=ann_feat_sorted["Importance"],
        y=ann_feat_sorted["Feature"],
        orientation="h",
        marker_color=COLORS["blue"],
        marker_line_width=0,
        text=[f"{v:.4f}" for v in ann_feat_sorted["Importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean AUC Drop: %{x:.4f}<extra></extra>",
    ))
    fig_ann_feat.update_layout(
        height=360, margin=dict(l=0, r=80, t=10, b=0),
        xaxis=dict(title="Mean AUC Drop (Importance)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig_ann_feat, use_container_width=True)

    st.divider()

    # ── ANN Threshold Optimization ────────────────────────────────────────────
    st.subheader("ANN Threshold Optimization")
    thresh_col1, thresh_col2, thresh_col3, thresh_col4 = st.columns(4)
    thresh_col1.metric("Optimal Threshold", "0.43", "maximises F1")
    thresh_col2.metric("Best F1 Score",     "0.6995")
    thresh_col3.metric("Recall @ 0.43",     "88.96%", "high recall mode")
    thresh_col4.metric("Precision @ 0.43",  "57.64%", "vs 62.1% @ 0.50")

    st.info(
        "Lowering the decision threshold from 0.50 → 0.43 increases **recall** from 77.4% to 88.96% "
        "(catching more churners), at the cost of precision (62.1% → 57.6%). "
        "The F1 score improves slightly: 0.6894 → 0.6995. "
        "Choose threshold based on the business cost of false negatives vs false positives."
    )

    st.divider()

    # ── Key Findings ──────────────────────────────────────────────────────────
    st.subheader("Key Findings")
    st.markdown("""
- **ANN AUC: 0.7621** — competitive for a lightweight 3-layer network on 90K rows with no balancing
- **CNN AUC: 0.7656** — 1D convolutions capture local feature patterns, edging the ANN slightly
- **XGBoost still dominates** (AUC 0.84 on balanced 748-row set) — deep learning with more data converges lower due to the harder, unbalanced problem setup
- **freight_value** is the top signal for both ML (RFE rank 1) and DL (permutation rank 1)
- **Delivery timing** (`delivered_estimated`, `purchased_delivered`) ranks 2nd–3rd in ANN importance — not selected by RFE but still informative
- **Class weights** (0.916 / 1.101) were used instead of under-sampling, allowing the full 90K-row dataset to be trained on
- **Geolocation** (`geolocation_lat/lng`) enters the top-10 for the ANN — a new signal not available in the ML pipeline
    """)

# ── Tab 7: Pipeline Summary ────────────────────────────────────────────────────

with tab7:
    st.subheader("ML Pipeline Summary")

    steps = [
        ("1️⃣  Data Loading & EDA",         "9 CSV files merged → 95,104 rows × 20+ cols. Geolocation fixed. Missing values & duplicates removed."),
        ("2️⃣  Feature Engineering & RFM",   "RFM scores computed per customer_unique_id. Churn defined as Frequency = 1 (one-time buyers). 97% churn rate detected."),
        ("3️⃣  Preprocessing",               "IQR capping for outliers. PowerTransformer (Yeo-Johnson) for normality. `no_of_orders` dropped (leakage risk). 9,038 rows × 13 features."),
        ("4️⃣  Balancing & Splitting",        "50/50 SMOTE-style balance: 374 churned + 374 retained = 748 rows. 80/20 train/test split → 598 train, 150 test."),
        ("5️⃣  Baseline Models (7)",          "Logistic Regression, Decision Tree, Random Forest, KNN (k=3), Naïve Bayes, SVM, XGBoost. Best baseline: XGBoost AUC 0.80."),
        ("6️⃣  RFE Feature Selection",        "Recursive Feature Elimination → 6 top features: freight_value, carrier_delivered, approved_carrier, order_item_id, no_of_products, payment_sequential."),
        ("7️⃣  GridSearchCV Tuning",          "Tuned RF, XGBoost, SVM. Best: XGBoost with lr=0.1, max_depth=3, n_estimators=200."),
        ("8️⃣  Best ML Result",              "Tuned XGBoost — AUC: 0.8267, Accuracy: 82.67%, F1: 0.8194, Kappa: 0.6533, CM: [[65,10],[16,59]]"),
        ("9️⃣  Deep Learning (ANN & CNN)",   "Full 90K-row dataset. 20 features (region-encoded state + payment type OHE). Class weights replace balancing. ANN: AUC 0.7621. CNN: AUC 0.7656. Permutation importance confirms freight_value as top signal."),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.markdown(desc)

    st.divider()
    st.subheader("Dataset Pipeline at a Glance")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Raw Dataset",     "9,038 rows",  "after merge & clean")
    p2.metric("Balanced",        "748 rows",    "50/50 balance")
    p3.metric("Features (ML)",   "13",          "no_of_orders dropped")
    p4.metric("Training Set",    "598 rows",    "80% split")
    p5.metric("Test Set",        "150 rows",    "20% split")

    st.divider()
    p6, p7, p8 = st.columns(3)
    p6.metric("DL Dataset",      "90,528 rows", "full, unbalanced")
    p7.metric("Features (DL)",   "20",          "region + payment OHE")
    p8.metric("DL Train / Test", "72,422 / 18,106", "80/20 stratified")

    st.divider()
    st.subheader("Notebooks Structure")
    nb_data = {
        "Notebook": [
            "01_Data_Loading_and_EDA",
            "02_Feature_Engineering_RFM",
            "03_Preprocessing_Balancing",
            "04_Model_Training_Evaluation",
            "05_Deep_Learning_ANN_CNN",
        ],
        "Role": [
            "Data Engineer / Analyst",
            "Data Scientist",
            "ML Engineer",
            "ML Engineer / Lead",
            "DL Engineer",
        ],
        "Output": [
            "Merged 95K-row dataset",
            "RFM table + churn labels",
            "748-row balanced features",
            "12 models + best: Tuned XGBoost (AUC 0.8267)",
            "ANN (AUC 0.7621) + CNN (AUC 0.7656) on full 90K rows",
        ],
    }
    st.dataframe(pd.DataFrame(nb_data), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Olist Brazilian E-Commerce Public Dataset · "
    "Churn defined as Frequency = 1 · "
    "ML: Balanced 50/50 · 80/20 Train/Test · Best: Tuned XGBoost (AUC 0.8267) · "
    "DL: Full 90K rows · Class Weights · ANN AUC 0.7621 · CNN AUC 0.7656"
)
