import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Financial Behaviour Analytics System",
    layout="wide"
)

# ==========================================
# USER DATABASE
# ==========================================

if 'users' not in st.session_state:

    st.session_state.users = {

        "admin": {
            "password": "admin123",
            "role": "Admin",
            "status": "Approved"
        }

    }

# ==========================================
# SESSION STATE
# ==========================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

if 'role' not in st.session_state:
    st.session_state.role = ""

# ==========================================
# SIDEBAR MENU
# ==========================================

menu = st.sidebar.selectbox(
    "Menu",
    ["Login", "Register"]
)

# ==========================================
# REGISTER PAGE
# ==========================================

if menu == "Register":

    st.title("📝 User Registration")

    new_user = st.text_input(
        "Create Username"
    )

    new_password = st.text_input(
        "Create Password",
        type="password"
    )

    if st.button("Register"):

        if new_user in st.session_state.users:

            st.warning(
                "⚠ Username already exists"
            )

        else:

            st.session_state.users[new_user] = {

                "password": new_password,
                "role": "User",
                "status": "Pending"

            }

            st.success(
                "✅ Registration Submitted"
            )

            st.info(
                "⏳ Waiting for Admin Approval"
            )

# ==========================================
# LOGIN PAGE
# ==========================================

elif menu == "Login":

    # --------------------------------------
    # LOGIN FORM
    # --------------------------------------

    if not st.session_state.logged_in:

        st.title("🔐 Login System")

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if username in st.session_state.users:

                user = st.session_state.users[username]

                if user["password"] == password:

                    # CHECK APPROVAL STATUS

                    if user["status"] != "Approved":

                        st.warning(
                            "⏳ Account Waiting For Admin Approval"
                        )

                    else:

                        st.session_state.logged_in = True

                        st.session_state.username = username

                        st.session_state.role = user["role"]

                        st.success(
                            f"✅ Welcome {username}"
                        )

                        st.rerun()

                else:

                    st.error(
                        "❌ Incorrect Password"
                    )

            else:

                st.error(
                    "❌ Username Not Found"
                )

    # --------------------------------------
    # AFTER LOGIN
    # --------------------------------------

    else:

        # ==========================================
        # LOAD DATASET
        # ==========================================

        df = pd.read_csv(
            r"C:\Users\User\BigData\df_features.csv"
        )

        # ==========================================
        # SIDEBAR USER INFO
        # ==========================================

        st.sidebar.success(
            f"Logged in as: {st.session_state.username}"
        )

        st.sidebar.info(
            f"Role: {st.session_state.role}"
        )

        if st.sidebar.button("Logout"):

            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""

            st.rerun()

        # ==========================================
        # ADMIN NAVIGATION
        # ==========================================

        if st.session_state.role == "Admin":

            section = st.sidebar.radio(

                "Navigation",

                [

                    "Admin Dashboard",
                    "User Management",
                    "Dashboard Overview",
                    "Customer Segmentation",
                    "Behaviour Analysis",
                    "Correlation Analysis",
                    "Model Comparison",
                    "Feature Importance",
                    "Prediction System"

                ]

            )

        # ==========================================
        # USER NAVIGATION
        # ==========================================

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

        # ==========================================
        # ADMIN DASHBOARD
        # ==========================================

        if section == "Admin Dashboard":

            st.title("🛠️ Admin Dashboard")

            total_users = len(st.session_state.users)

            approved_users = sum(
                1 for u in st.session_state.users.values()
                if u["status"] == "Approved"
            )

            pending_users = sum(
                1 for u in st.session_state.users.values()
                if u["status"] == "Pending"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Users",
                total_users
            )

            col2.metric(
                "Approved Users",
                approved_users
            )

            col3.metric(
                "Pending Users",
                pending_users
            )

        # ==========================================
        # USER MANAGEMENT
        # ==========================================

        elif section == "User Management":

            st.title("👥 User Management")

            users_data = []

            for username, details in st.session_state.users.items():

                users_data.append({

                    "Username": username,
                    "Role": details["role"],
                    "Status": details["status"]

                })

            users_df = pd.DataFrame(users_data)

            st.dataframe(users_df)

            st.subheader("Approve / Reject Users")

            selected_user = st.selectbox(

                "Select User",

                [
                    user for user in st.session_state.users
                    if user != "admin"
                ]

            )

            action = st.radio(

                "Action",

                ["Approve", "Reject", "Delete"]

            )

            if st.button("Apply Action"):

                if action == "Approve":

                    st.session_state.users[selected_user][
                        "status"
                    ] = "Approved"

                    st.success(
                        f"{selected_user} Approved"
                    )

                elif action == "Reject":

                    st.session_state.users[selected_user][
                        "status"
                    ] = "Rejected"

                    st.warning(
                        f"{selected_user} Rejected"
                    )

                elif action == "Delete":

                    del st.session_state.users[selected_user]

                    st.error(
                        f"{selected_user} Deleted"
                    )

                st.rerun()

        # ==========================================
        # DASHBOARD OVERVIEW
        # ==========================================

        elif section == "Dashboard Overview":

            st.title(
                "📊 Customer Financial Behaviour Analytics System"
            )

            st.markdown(
                """
                This system analyses customer financial behaviour
                using machine learning and data analytics techniques.
                """
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Customers",
                len(df)
            )

            col2.metric(
                "Total Clusters",
                df['cluster'].nunique()
            )

            col3.metric(
                "Average Spending",
                round(df['total_spent'].mean(), 2)
            )

            st.subheader("📁 Dataset Preview")

            st.dataframe(df.head())

        # ==========================================
        # CUSTOMER SEGMENTATION
        # ==========================================

        elif section == "Customer Segmentation":

            st.title("👥 Customer Segmentation")

            cluster_counts = (
                df['cluster']
                .value_counts()
                .reset_index()
            )

            cluster_counts.columns = [
                'Cluster',
                'Number of Customers'
            ]

            fig_cluster = px.bar(
                cluster_counts,
                x='Cluster',
                y='Number of Customers',
                color='Cluster',
                title="Customer Distribution by Cluster"
            )

            st.plotly_chart(
                fig_cluster,
                use_container_width=True
            )

            st.subheader(
                "🔍 Customer Segmentation Scatter Plot"
            )

            fig_scatter = px.scatter(
                df,
                x='total_spent',
                y='transaction_count',
                color='cluster',
                hover_data=[
                    'avg_spent',
                    'recency'
                ],
                title="Customer Segments"
            )

            st.plotly_chart(
                fig_scatter,
                use_container_width=True
            )

        # ==========================================
        # BEHAVIOUR ANALYSIS
        # ==========================================

        elif section == "Behaviour Analysis":

            st.title("📈 Customer Behaviour Analysis")

            feature = st.selectbox(

                "Select Feature",

                [
                    'total_spent',
                    'avg_spent',
                    'transaction_count',
                    'recency'
                ]

            )

            fig_feature = px.histogram(
                df,
                x=feature,
                nbins=50,
                title=f"Distribution of {feature}"
            )

            fig_feature.update_layout(
                xaxis_title=feature,
                yaxis_title="Customer Count"
            )

            st.plotly_chart(
                fig_feature,
                use_container_width=True
            )

        # ==========================================
        # CORRELATION ANALYSIS
        # ==========================================

        elif section == "Correlation Analysis":

            st.title("📊 Correlation Analysis")

            correlation = df[
                [
                    'total_spent',
                    'avg_spent',
                    'transaction_count',
                    'recency'
                ]
            ].corr()

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            sns.heatmap(
                correlation,
                annot=True,
                cmap='coolwarm',
                ax=ax
            )

            st.pyplot(fig)

        # ==========================================
        # MODEL COMPARISON
        # ==========================================

        elif section == "Model Comparison":

            st.title("🤖 Machine Learning Model Comparison")

            model_data = pd.DataFrame({

                'Model': [

                    'Logistic Regression',
                    'Decision Tree',
                    'Random Forest'

                ],

                'Accuracy': [

                    0.85,
                    0.90,
                    0.94

                ]

            })

            fig_model = px.bar(

                model_data,

                x='Model',
                y='Accuracy',
                color='Model',
                title="Model Accuracy Comparison"

            )

            st.plotly_chart(
                fig_model,
                use_container_width=True
            )

        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        elif section == "Feature Importance":

            st.title("⭐ Feature Importance")

            feature_data = pd.DataFrame({

                'Feature': [

                    'transaction_count',
                    'avg_spent',
                    'recency'

                ],

                'Importance': [

                    0.40,
                    0.35,
                    0.25

                ]

            })

            fig_importance = px.bar(

                feature_data,

                x='Feature',
                y='Importance',
                color='Feature',
                title="Feature Importance"

            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

        # ==========================================
        # PREDICTION SYSTEM
        # ==========================================

        elif section == "Prediction System":

            st.title("🧠 Predict Customer Value")

            col1, col2 = st.columns(2)

            with col1:

                total_spent = st.number_input(
                    "Total Spending ($)"
                )

                avg_spent = st.number_input(
                    "Average Spending ($)"
                )

            with col2:

                transaction_count = st.number_input(
                    "Transaction Count"
                )

                recency = st.number_input(
                    "Recency (days)"
                )

            if st.button("Predict Customer Value"):

                if (
                    total_spent
                    > df['total_spent'].median()
                ):

                    st.success(
                        "🟢 High Value Customer "
                        "(Cross-Selling Opportunity)"
                    )

                else:

                    st.warning(
                        "🔴 Low Value Customer"
                    )