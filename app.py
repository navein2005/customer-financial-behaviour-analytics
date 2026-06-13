import sqlite3
import hashlib
import re
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Customer Financial Behaviour Analytics System",
    layout="wide"
)

DATA_PATH = "df_features.csv"
DB_PATH = "users.db"

FEATURES = ["total_spent", "avg_spent", "transaction_count", "recency"]


# ===============================
# SECURITY FUNCTIONS
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


# ===============================
# DATABASE FUNCTIONS
# ===============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL
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

    c.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    admin_exists = c.fetchone()

    if not admin_exists:
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "Admin", "Approved")
        )

    conn.commit()
    conn.close()


def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            (username, hash_password(password), "User", "Pending")
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT username, password, role, status FROM users WHERE username = ?",
        (username,)
    )

    user = c.fetchone()
    conn.close()

    if user and user[1] == hash_password(password):
        return {
            "username": user[0],
            "role": user[2],
            "status": user[3]
        }

    return None


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT username, role, status FROM users",
        conn
    )
    conn.close()
    return df


def update_user_status(username, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "UPDATE users SET status = ? WHERE username = ?",
        (status, username)
    )

    conn.commit()
    conn.close()


def delete_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "DELETE FROM users WHERE username = ?",
        (username,)
    )

    conn.commit()
    conn.close()


def add_audit_log(admin_username, action, target_user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO audit_logs
        (admin_username, action, target_user, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
            admin_username,
            action,
            target_user,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()


def get_audit_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT admin_username, action, target_user, timestamp
        FROM audit_logs
        ORDER BY id DESC
        """,
        conn
    )
    conn.close()
    return df


# ===============================
# DATA FUNCTIONS
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in FEATURES:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    if "cluster" not in df.columns:
        df["cluster"] = 0

    if "target" not in df.columns:
        median_spending = df["total_spent"].median()
        df["target"] = (df["total_spent"] > median_spending).astype(int)

    return df


@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=8
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4)
        })

        trained_models[name] = {
            "model": model,
            "confusion_matrix": cm
        }

    return pd.DataFrame(results), trained_models


# ===============================
# APP INITIALIZATION
# ===============================

init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""


# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("Customer Analytics System")

if not st.session_state.logged_in:

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login", "Register"]
    )

    if menu == "Register":

        st.title("User Registration")

        new_user = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")

        st.info(
            "Password must contain at least 8 characters, one uppercase letter, "
            "one lowercase letter, and one number."
        )

        if st.button("Register"):

            username_valid, username_msg = validate_username(new_user)
            password_valid, password_msg = validate_password(new_password)

            if not username_valid:
                st.warning(username_msg)

            elif not password_valid:
                st.warning(password_msg)

            else:
                success = register_user(new_user, new_password)

                if success:
                    st.success("Registration submitted successfully.")
                    st.info("Your account is pending admin approval.")
                else:
                    st.error("Username already exists.")

    elif menu == "Login":

        st.title("Login System")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            user = login_user(username, password)

            if user:

                if user["status"] != "Approved":
                    st.warning("Your account is waiting for admin approval.")

                else:
                    st.session_state.logged_in = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]
                    st.rerun()

            else:
                st.error("Invalid username or password.")


# ===============================
# LOGGED IN AREA
# ===============================

else:

    df = load_data()
    model_results, trained_models = train_models(df)
    best_model = trained_models["Random Forest"]["model"]

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    st.sidebar.info(f"Role: {st.session_state.role}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

    if st.session_state.role == "Admin":

        section = st.sidebar.radio(
            "Navigation",
            [
                "Admin Dashboard",
                "User Management",
                "Audit Logs",
                "Dashboard Overview",
                "Customer Segmentation",
                "Behaviour Analysis",
                "Correlation Analysis",
                "Model Comparison",
                "Feature Importance",
                "Prediction System"
            ]
        )

    else:

        section = st.sidebar.radio(
            "Navigation",
            [
                "Dashboard Overview",
                "Customer Segmentation",
                "Behaviour Analysis",
                "Correlation Analysis",
                "Model Comparison",
                "Feature Importance",
                "Prediction System"
            ]
        )

    # ===============================
    # ADMIN DASHBOARD
    # ===============================

    if section == "Admin Dashboard":

        st.title("Admin Dashboard")

        users_df = get_all_users()

        total_users = len(users_df)
        approved_users = len(users_df[users_df["status"] == "Approved"])
        pending_users = len(users_df[users_df["status"] == "Pending"])
        rejected_users = len(users_df[users_df["status"] == "Rejected"])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Users", total_users)
        col2.metric("Approved Users", approved_users)
        col3.metric("Pending Users", pending_users)
        col4.metric("Rejected Users", rejected_users)

        st.subheader("System Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(df))
        col2.metric("Customer Clusters", df["cluster"].nunique())
        col3.metric(
            "Best Model Accuracy",
            model_results["Accuracy"].max()
        )

        st.write(
            "This admin dashboard supports user monitoring, account approval, "
            "customer segmentation analysis, model evaluation, and audit logging."
        )

    # ===============================
    # USER MANAGEMENT
    # ===============================

    elif section == "User Management":

        st.title("User Management")

        users_df = get_all_users()
        st.dataframe(users_df, use_container_width=True)

        non_admin_users = users_df[
            users_df["username"] != "admin"
        ]["username"].tolist()

        if non_admin_users:

            selected_user = st.selectbox("Select User", non_admin_users)
            action = st.radio("Action", ["Approve", "Reject", "Delete"])

            if st.button("Apply Action"):

                if action == "Approve":
                    update_user_status(selected_user, "Approved")
                    add_audit_log(
                        st.session_state.username,
                        "Approved user",
                        selected_user
                    )
                    st.success(f"{selected_user} has been approved.")

                elif action == "Reject":
                    update_user_status(selected_user, "Rejected")
                    add_audit_log(
                        st.session_state.username,
                        "Rejected user",
                        selected_user
                    )
                    st.warning(f"{selected_user} has been rejected.")

                elif action == "Delete":
                    delete_user(selected_user)
                    add_audit_log(
                        st.session_state.username,
                        "Deleted user",
                        selected_user
                    )
                    st.error(f"{selected_user} has been deleted.")

                st.rerun()

        else:
            st.info("No user accounts available.")

    # ===============================
    # AUDIT LOGS
    # ===============================

    elif section == "Audit Logs":

        st.title("Security Audit Logs")

        logs_df = get_audit_logs()

        if logs_df.empty:
            st.info("No admin activities recorded yet.")

        else:
            st.dataframe(logs_df, use_container_width=True)

    # ===============================
    # DASHBOARD OVERVIEW
    # ===============================

    elif section == "Dashboard Overview":

        st.title("Customer Financial Behaviour Analytics System")

        st.write(
            "This system analyses customer financial behaviour using data analytics "
            "and machine learning techniques. It supports segmentation, behaviour "
            "analysis, customer value prediction, and admin monitoring."
        )

        total_customers = len(df)
        total_revenue = round(df["total_spent"].sum(), 2)
        avg_spending = round(df["avg_spent"].mean(), 2)
        total_clusters = df["cluster"].nunique()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", total_customers)
        col2.metric("Total Revenue", total_revenue)
        col3.metric("Average Spending", avg_spending)
        col4.metric("Customer Clusters", total_clusters)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Processed Dataset",
            data=csv,
            file_name="processed_customer_features.csv",
            mime="text/csv"
        )

    # ===============================
    # CUSTOMER SEGMENTATION
    # ===============================

    elif section == "Customer Segmentation":

        st.title("Customer Segmentation")

        cluster_counts = df["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Number of Customers"]

        fig_cluster = px.bar(
            cluster_counts,
            x="Cluster",
            y="Number of Customers",
            color="Cluster",
            title="Customer Distribution by Cluster"
        )

        st.plotly_chart(fig_cluster, use_container_width=True)

        fig_scatter = px.scatter(
            df,
            x="total_spent",
            y="transaction_count",
            color="cluster",
            hover_data=["avg_spent", "recency"],
            title="Customer Segmentation Based on Spending and Transaction Count"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Cluster Summary")

        cluster_summary = df.groupby("cluster")[FEATURES].mean().round(2)
        st.dataframe(cluster_summary, use_container_width=True)

        st.subheader("Spending Pattern by Cluster")

        fig_box = px.box(
            df,
            x="cluster",
            y="total_spent",
            color="cluster",
            title="Total Spending Distribution by Customer Cluster"
        )

        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Recency Compared with Spending")

        fig_recency = px.scatter(
            df,
            x="recency",
            y="total_spent",
            color="cluster",
            hover_data=["avg_spent", "transaction_count"],
            title="Customer Recency Compared with Total Spending"
        )

        st.plotly_chart(fig_recency, use_container_width=True)

    # ===============================
    # BEHAVIOUR ANALYSIS
    # ===============================

    elif section == "Behaviour Analysis":

        st.title("Customer Behaviour Analysis")

        feature = st.selectbox(
            "Select Behaviour Feature",
            FEATURES
        )

        fig_hist = px.histogram(
            df,
            x=feature,
            nbins=50,
            title=f"Distribution of {feature}"
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Feature Statistics")
        st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)

        st.subheader("Feature Relationship")

        fig_pair = px.scatter_matrix(
            df,
            dimensions=FEATURES,
            color="cluster",
            title="Customer Behaviour Feature Relationship Matrix"
        )

        st.plotly_chart(fig_pair, use_container_width=True)

    # ===============================
    # CORRELATION ANALYSIS
    # ===============================

    elif section == "Correlation Analysis":

        st.title("Correlation Analysis")

        correlation = df[FEATURES].corr()

        fig_corr = px.imshow(
            correlation,
            text_auto=True,
            title="Pearson Correlation Heatmap",
            color_continuous_scale="RdBu"
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    # ===============================
    # MODEL COMPARISON
    # ===============================

    elif section == "Model Comparison":

        st.title("Machine Learning Model Comparison")

        st.dataframe(model_results, use_container_width=True)

        fig_model = px.bar(
            model_results,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Model Accuracy Comparison"
        )

        st.plotly_chart(fig_model, use_container_width=True)

        best_model_name = model_results.sort_values(
            by="Accuracy",
            ascending=False
        ).iloc[0]["Model"]

        st.success(f"Best Performing Model: {best_model_name}")

        selected_model = st.selectbox(
            "Select Model to View Confusion Matrix",
            list(trained_models.keys())
        )

        cm = trained_models[selected_model]["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Low Value", "Predicted High Value"],
            index=["Actual Low Value", "Actual High Value"]
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            title=f"Confusion Matrix - {selected_model}",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================

    elif section == "Feature Importance":

        st.title("Feature Importance Analysis")

        rf_model = trained_models["Random Forest"]["model"]

        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig_importance = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            color="Feature",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(fig_importance, use_container_width=True)
        st.dataframe(importance_df.round(4), use_container_width=True)

        top_feature = importance_df.iloc[0]["Feature"]

        st.info(
            f"The most influential feature in the Random Forest model is: {top_feature}"
        )

    # ===============================
    # PREDICTION SYSTEM
    # ===============================

    elif section == "Prediction System":

        st.title("Customer Value Prediction System")

        st.write(
            "Enter customer behaviour values to predict whether the customer "
            "is likely to be classified as a High Value Customer or Low Value Customer."
        )

        col1, col2 = st.columns(2)

        with col1:
            total_spent = st.number_input(
                "Total Spending",
                min_value=0.0,
                value=1000.0
            )

            avg_spent = st.number_input(
                "Average Spending",
                min_value=0.0,
                value=50.0
            )

        with col2:
            transaction_count = st.number_input(
                "Transaction Count",
                min_value=0,
                value=20
            )

            recency = st.number_input(
                "Recency in Days",
                min_value=0,
                value=30
            )

        if st.button("Predict Customer Value"):

            input_data = pd.DataFrame([{
                "total_spent": total_spent,
                "avg_spent": avg_spent,
                "transaction_count": transaction_count,
                "recency": recency
            }])

            prediction = best_model.predict(input_data)[0]
            probability = best_model.predict_proba(input_data)[0]

            if prediction == 1:
                st.success("Prediction: High Value Customer")
                st.write(
                    "This customer shows strong financial behaviour and may be suitable "
                    "for cross-selling opportunities or premium financial services."
                )

            else:
                st.warning("Prediction: Low Value Customer")
                st.write(
                    "This customer shows lower financial activity and may require "
                    "engagement strategies or personalised retention offers."
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Low Value Probability",
                round(probability[0], 3)
            )

            col2.metric(
                "High Value Probability",
                round(probability[1], 3)
            )

            st.subheader("Input Summary")
            st.dataframe(input_data, use_container_width=True)

            st.subheader("Prediction Explanation")

            explanation = []

            if total_spent > df["total_spent"].median():
                explanation.append("Total spending is above the median customer spending level.")
            else:
                explanation.append("Total spending is below the median customer spending level.")

            if transaction_count > df["transaction_count"].median():
                explanation.append("Transaction count is higher than the median customer activity level.")
            else:
                explanation.append("Transaction count is lower than the median customer activity level.")

            if recency < df["recency"].median():
                explanation.append("The customer has recent transaction activity.")
            else:
                explanation.append("The customer has not transacted recently compared to other customers.")

            for item in explanation:
                st.write(f"- {item}")