import os
import re
import random
import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(
    page_title="Customer Financial Behaviour Analytics System",
    layout="wide"
)

DATA_PATH = "df_features.csv"
DB_PATH = "users.db"

FEATURES = ["total_spent", "avg_spent", "transaction_count", "recency"]

# ===============================
# EMAIL CONFIGURATION
# ===============================
# Replace with your own email and Gmail App Password.
# Do NOT use your normal Gmail password.
# Example:
# EMAIL_SENDER = "your_email@gmail.com"
# EMAIL_APP_PASSWORD = "your_16_character_app_password"

EMAIL_SENDER = "naveinrajanderan@gmail.com"
EMAIL_APP_PASSWORD = "vneaqmrdocfmiczn"


# ===============================
# SECURITY + OTP
# ===============================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def validate_username(username):
    if not username:
        return False, "Username cannot be empty."
    if " " in username:
        return False, "Username cannot contain spaces."
    if len(username) < 4:
        return False, "Username must be at least 4 characters."
    return True, "Valid username."


def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one number."
    return True, "Valid password."


def validate_email(email):
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email) is not None


def generate_otp():
    return str(random.randint(100000, 999999))


def send_otp_email(receiver_email, otp):
    if EMAIL_SENDER == "YOUR_EMAIL@gmail.com" or EMAIL_APP_PASSWORD == "YOUR_16_CHARACTER_GMAIL_APP_PASSWORD":
        st.error("Please update EMAIL_SENDER and EMAIL_APP_PASSWORD in the code.")
        return False

    try:
        body = f"""
Hello,

Your OTP verification code is: {otp}

This code is required to complete your registration.

Regards,
Customer Financial Behaviour Analytics System
"""

        msg = MIMEText(body)
        msg["Subject"] = "OTP Verification"
        msg["From"] = EMAIL_SENDER
        msg["To"] = receiver_email

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        return True

    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False


# ===============================
# DATABASE
# ===============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL,
            failed_attempts INTEGER DEFAULT 0,
            locked INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_username TEXT,
            action TEXT,
            target_user TEXT,
            timestamp TEXT
        )
    """)

    c.execute("SELECT username FROM users WHERE username = ?", ("admin",))
    if not c.fetchone():
        c.execute("""
            INSERT INTO users
            (username, email, password, role, status, failed_attempts, locked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("admin", "admin@example.com", hash_password("admin123"), "Admin", "Approved", 0, 0))

    conn.commit()
    conn.close()


def register_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute("""
            INSERT INTO users
            (username, email, password, role, status, failed_attempts, locked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, email, hash_password(password), "User", "Pending", 0, 0))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT username, email, password, role, status, failed_attempts, locked
        FROM users WHERE username = ?
    """, (username,))

    user = c.fetchone()

    if not user:
        conn.close()
        return None, "Username not found."

    if user[6] == 1:
        conn.close()
        return None, "Account locked after too many failed login attempts."

    if user[2] == hash_password(password):
        c.execute("UPDATE users SET failed_attempts = 0 WHERE username = ?", (username,))
        conn.commit()
        conn.close()

        return {
            "username": user[0],
            "email": user[1],
            "role": user[3],
            "status": user[4]
        }, "Login successful."

    new_attempts = user[5] + 1
    locked = 1 if new_attempts >= 5 else 0

    c.execute("""
        UPDATE users SET failed_attempts = ?, locked = ?
        WHERE username = ?
    """, (new_attempts, locked, username))

    conn.commit()
    conn.close()

    return None, "Invalid password."


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT username, email, role, status, failed_attempts, locked FROM users",
        conn
    )
    conn.close()
    return df


def update_user_status(username, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET status = ? WHERE username = ?", (status, username))
    conn.commit()
    conn.close()


def unlock_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET locked = 0, failed_attempts = 0 WHERE username = ?", (username,))
    conn.commit()
    conn.close()


def delete_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()


def add_audit_log(admin_username, action, target_user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO audit_logs (admin_username, action, target_user, timestamp)
        VALUES (?, ?, ?, ?)
    """, (admin_username, action, target_user, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_audit_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT admin_username, action, target_user, timestamp
        FROM audit_logs ORDER BY id DESC
    """, conn)
    conn.close()
    return df


# ===============================
# DATA + ANALYTICS
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in FEATURES:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    if "target" not in df.columns:
        median_spending = df["total_spent"].median()
        df["target"] = (df["total_spent"] > median_spending).astype(int)

    return df


@st.cache_resource
def run_analytics(df):
    X = df[FEATURES]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    try:
        silhouette = silhouette_score(X_scaled, df["kmeans_cluster"])
    except Exception:
        silhouette = 0

    correlation_matrix = df[FEATURES].corr()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4)
        })

        trained_models[name] = {
            "model": model,
            "confusion_matrix": confusion_matrix(y_test, preds)
        }

    rf_model = trained_models["Random Forest"]["model"]
    feature_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    analytics_summary = pd.DataFrame({
        "Technique": [
            "K-Means Clustering",
            "Correlation Analysis",
            "Feature Importance Analysis",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Purpose": [
            "Customer segmentation",
            "Relationship analysis",
            "Identify important variables",
            "Customer classification",
            "Customer classification",
            "Customer classification"
        ],
        "Status": [
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented"
        ]
    })

    return df, pd.DataFrame(results), trained_models, correlation_matrix, feature_importance, silhouette, analytics_summary


def calculate_customer_score(total_spent, avg_spent, transaction_count, recency, df):
    spend_score = min((total_spent / df["total_spent"].max()) * 40, 40)
    avg_score = min((avg_spent / df["avg_spent"].max()) * 20, 20)
    frequency_score = min((transaction_count / df["transaction_count"].max()) * 30, 30)
    recency_score = max(10 - ((recency / df["recency"].max()) * 10), 0)
    return round(spend_score + avg_score + frequency_score + recency_score, 2)


def score_level(score):
    if score >= 75:
        return "Excellent"
    if score >= 50:
        return "Medium"
    return "Low"


def recommendation(score, prediction):
    if prediction == 1 and score >= 75:
        return "Offer premium financial products, loyalty rewards, and cross-selling opportunities."
    if prediction == 1:
        return "Provide personalised offers and targeted product recommendations."
    if score >= 50:
        return "Use engagement campaigns to increase transaction frequency."
    return "Focus on retention offers and financial awareness campaigns."


def cluster_persona(cluster):
    personas = {
        0: "Budget Conscious Customer",
        1: "Frequent Transaction Customer",
        2: "High Value Customer",
        3: "Inactive or Low Engagement Customer"
    }
    return personas.get(int(cluster), "General Customer Segment")


# ===============================
# APP START
# ===============================

init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "pending_otp" not in st.session_state:
    st.session_state.pending_otp = ""
if "pending_registration" not in st.session_state:
    st.session_state.pending_registration = {}

st.sidebar.title("Customer Analytics System")

if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

    if menu == "Register":
        st.title("User Registration with Email OTP")

        new_user = st.text_input("Create Username")
        new_email = st.text_input("Email Address")
        new_password = st.text_input("Create Password", type="password")

        if st.button("Send OTP"):
            username_valid, username_msg = validate_username(new_user)
            password_valid, password_msg = validate_password(new_password)

            if not username_valid:
                st.warning(username_msg)
            elif not validate_email(new_email):
                st.warning("Please enter a valid email address.")
            elif not password_valid:
                st.warning(password_msg)
            else:
                otp = generate_otp()
                st.session_state.pending_otp = otp
                st.session_state.pending_registration = {
                    "username": new_user,
                    "email": new_email,
                    "password": new_password
                }

                if send_otp_email(new_email, otp):
                    st.success("OTP sent to your email.")

        entered_otp = st.text_input("Enter OTP")

        if st.button("Verify OTP and Register"):
            if entered_otp == st.session_state.pending_otp and entered_otp != "":
                reg = st.session_state.pending_registration
                success = register_user(reg["username"], reg["email"], reg["password"])

                if success:
                    st.success("Registration submitted successfully.")
                    st.info("Waiting for admin approval.")
                    st.session_state.pending_otp = ""
                    st.session_state.pending_registration = {}
                else:
                    st.error("Username already exists.")
            else:
                st.error("Invalid OTP.")

    elif menu == "Login":
        st.title("Login System")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user, msg = login_user(username, password)

            if user:
                if user["status"] != "Approved":
                    st.warning("Your account is waiting for admin approval.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]
                    st.rerun()
            else:
                st.error(msg)

else:
    df = load_data()
    df, model_results, trained_models, correlation_matrix, feature_importance, silhouette, analytics_summary = run_analytics(df)
    best_model = trained_models["Random Forest"]["model"]

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    st.sidebar.info(f"Role: {st.session_state.role}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

    if st.session_state.role == "Admin":
        section = st.sidebar.radio("Navigation", [
            "Admin Dashboard",
            "User Management",
            "Audit Logs",
            "Data Quality",
            "Analytics Techniques",
            "Dashboard Overview",
            "Customer Segmentation",
            "Behaviour Analysis",
            "Correlation Analysis",
            "Model Comparison",
            "Feature Importance",
            "Prediction System"
        ])
    else:
        section = st.sidebar.radio("Navigation", [
            "Analytics Techniques",
            "Dashboard Overview",
            "Customer Segmentation",
            "Behaviour Analysis",
            "Correlation Analysis",
            "Model Comparison",
            "Feature Importance",
            "Prediction System"
        ])

    if section == "Admin Dashboard":
        st.title("Executive Admin Dashboard")

        users_df = get_all_users()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", len(users_df))
        col2.metric("Approved Users", len(users_df[users_df["status"] == "Approved"]))
        col3.metric("Pending Users", len(users_df[users_df["status"] == "Pending"]))
        col4.metric("Locked Accounts", len(users_df[users_df["locked"] == 1]))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        col2.metric("Total Revenue", round(df["total_spent"].sum(), 2))
        col3.metric("Customer Clusters", df["kmeans_cluster"].nunique())
        col4.metric("Best Accuracy", model_results["Accuracy"].max())

    elif section == "User Management":
        st.title("User Management")

        users_df = get_all_users()
        st.dataframe(users_df, use_container_width=True)

        non_admin_users = users_df[users_df["username"] != "admin"]["username"].tolist()

        if non_admin_users:
            selected_user = st.selectbox("Select User", non_admin_users)
            action = st.radio("Action", ["Approve", "Reject", "Unlock", "Delete"])

            if st.button("Apply Action"):
                if action == "Approve":
                    update_user_status(selected_user, "Approved")
                    add_audit_log(st.session_state.username, "Approved user", selected_user)
                elif action == "Reject":
                    update_user_status(selected_user, "Rejected")
                    add_audit_log(st.session_state.username, "Rejected user", selected_user)
                elif action == "Unlock":
                    unlock_user(selected_user)
                    add_audit_log(st.session_state.username, "Unlocked user", selected_user)
                elif action == "Delete":
                    delete_user(selected_user)
                    add_audit_log(st.session_state.username, "Deleted user", selected_user)

                st.success("Action completed.")
                st.rerun()
        else:
            st.info("No user accounts available.")

    elif section == "Audit Logs":
        st.title("Security Audit Logs")

        logs_df = get_audit_logs()
        st.dataframe(logs_df, use_container_width=True)

        if not logs_df.empty:
            fig = px.histogram(logs_df, x="action", title="Admin Activity Distribution")
            st.plotly_chart(fig, use_container_width=True)

    elif section == "Data Quality":
        st.title("Data Quality Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicate Rows", int(df.duplicated().sum()))

        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

    elif section == "Analytics Techniques":
        st.title("Six Analytical Techniques Implemented")

        st.dataframe(analytics_summary, use_container_width=True)

        st.metric("K-Means Silhouette Score", round(silhouette, 4))

        st.info("""
The system implements six analytical techniques:

1. K-Means Clustering
2. Correlation Analysis
3. Feature Importance Analysis
4. Logistic Regression
5. Decision Tree
6. Random Forest

K-Means is used for customer segmentation.

Correlation Analysis identifies relationships between spending, frequency, average spending, and recency.

Feature Importance identifies the most influential customer behaviour variables.

Logistic Regression, Decision Tree, and Random Forest are used for customer classification and prediction.

Only the three classification models are shown in the Model Comparison page because they produce accuracy scores and confusion matrices.
""")

    elif section == "Dashboard Overview":
        st.title("Customer Financial Behaviour Analytics System")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        col2.metric("Total Revenue", round(df["total_spent"].sum(), 2))
        col3.metric("Average Spending", round(df["avg_spent"].mean(), 2))
        col4.metric("Customer Clusters", df["kmeans_cluster"].nunique())

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Processed Dataset", csv, "processed_customer_features.csv", "text/csv")

    elif section == "Customer Segmentation":
        st.title("Customer Segmentation")

        cluster_counts = df["kmeans_cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Number of Customers"]

        fig = px.bar(cluster_counts, x="Cluster", y="Number of Customers", color="Cluster",
                     title="Customer Distribution by K-Means Cluster")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df, x="total_spent", y="transaction_count", color="kmeans_cluster",
                         hover_data=["avg_spent", "recency"],
                         title="Customer Segmentation Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)

        summary = df.groupby("kmeans_cluster")[FEATURES].mean().round(2).reset_index()
        summary["Persona"] = summary["kmeans_cluster"].apply(cluster_persona)
        st.subheader("Cluster Summary with Personas")
        st.dataframe(summary, use_container_width=True)

        st.subheader("Business Insights")
        for _, row in summary.iterrows():
            st.write(
                f"Cluster {row['kmeans_cluster']} - {row['Persona']}: "
                f"Average Spending = {row['avg_spent']:.2f}, "
                f"Average Transactions = {row['transaction_count']:.2f}, "
                f"Average Recency = {row['recency']:.2f}"
            )

        fig = px.box(df, x="kmeans_cluster", y="total_spent", color="kmeans_cluster",
                     title="Spending Distribution by Cluster")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Behaviour Analysis":
        st.title("Customer Behaviour Analysis")

        feature = st.selectbox("Select Feature", FEATURES)

        fig = px.histogram(df, x=feature, nbins=50, title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)

        fig = px.scatter_matrix(df, dimensions=FEATURES, color="kmeans_cluster",
                                title="Feature Relationship Matrix")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Correlation Analysis":
        st.title("Correlation Analysis")

        fig = px.imshow(correlation_matrix, text_auto=True,
                        title="Pearson Correlation Heatmap",
                        color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)

        pairs = correlation_matrix.abs().unstack().reset_index()
        pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        pairs = pairs[pairs["Feature 1"] != pairs["Feature 2"]]
        strongest = pairs.sort_values(by="Correlation", ascending=False).iloc[0]

        st.info(
            f"Strongest relationship: {strongest['Feature 1']} and "
            f"{strongest['Feature 2']} = {round(strongest['Correlation'], 3)}"
        )

        st.subheader("Interpretation")
        st.write(
            "The correlation analysis helps identify relationships between customer "
            "behaviour variables such as total spending, average spending, transaction "
            "frequency, and recency."
        )

    elif section == "Model Comparison":
        st.title("Machine Learning Model Comparison")

        st.dataframe(model_results, use_container_width=True)

        best_model_name = model_results.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
        st.success(f"Best Performing Model: {best_model_name}")

        fig = px.bar(model_results, x="Model", y="Accuracy", color="Model",
                     title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)

        selected_model = st.selectbox("Select Model", list(trained_models.keys()))
        cm = trained_models[selected_model]["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Low Value", "Predicted High Value"],
            index=["Actual Low Value", "Actual High Value"]
        )

        fig = px.imshow(cm_df, text_auto=True,
                        title=f"Confusion Matrix - {selected_model}",
                        color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Feature Importance":
        st.title("Feature Importance Analysis")

        fig = px.bar(feature_importance, x="Feature", y="Importance", color="Feature",
                     title="Random Forest Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(feature_importance.round(4), use_container_width=True)

        top_feature = feature_importance.iloc[0]
        st.success(
            f"Most Important Feature: {top_feature['Feature']} "
            f"({round(top_feature['Importance'], 4)})"
        )

    elif section == "Prediction System":
        st.title("Customer Value Prediction System")

        col1, col2 = st.columns(2)

        with col1:
            total_spent = st.number_input("Total Spending", min_value=0.0, value=1000.0)
            avg_spent = st.number_input("Average Spending", min_value=0.0, value=50.0)

        with col2:
            transaction_count = st.number_input("Transaction Count", min_value=0, value=20)
            recency = st.number_input("Recency in Days", min_value=0, value=30)

        if st.button("Predict Customer Value"):
            input_data = pd.DataFrame([{
                "total_spent": total_spent,
                "avg_spent": avg_spent,
                "transaction_count": transaction_count,
                "recency": recency
            }])

            prediction = best_model.predict(input_data)[0]
            probability = best_model.predict_proba(input_data)[0]

            score = calculate_customer_score(total_spent, avg_spent, transaction_count, recency, df)

            if prediction == 1:
                st.success("Prediction: High Value Customer")
            else:
                st.warning("Prediction: Low Value Customer")

            col1, col2, col3 = st.columns(3)
            col1.metric("Customer Score", f"{score}/100")
            col2.metric("Score Level", score_level(score))
            col3.metric("High Value Probability", round(probability[1], 3))

            st.subheader("Business Recommendation")
            st.info(recommendation(score, prediction))

            st.subheader("Fraud / Abnormal Behaviour Indicator")
            if total_spent > df["total_spent"].quantile(0.95):
                st.error("Potential abnormal high-spending behaviour detected.")
            else:
                st.success("No major abnormal spending indicator detected.")

            st.subheader("Input Summary")
            st.dataframe(input_data, use_container_width=True)

            report_df = pd.DataFrame({
                "Metric": [
                    "Customer Score",
                    "Score Level",
                    "Prediction",
                    "High Value Probability",
                    "Business Recommendation"
                ],
                "Value": [
                    score,
                    score_level(score),
                    "High Value Customer" if prediction == 1 else "Low Value Customer",
                    round(probability[1], 3),
                    recommendation(score, prediction)
                ]
            })

            csv = report_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Report",
                data=csv,
                file_name="customer_prediction_report.csv",
                mime="text/csv"
            )