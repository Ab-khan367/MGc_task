"""
Streamlit Web Interface for MGC Sales Assistant.

Combines the document assistant (Part 1) and lead scoring tool (Part 3)
into a single web application.
"""

import streamlit as st
import joblib
import numpy as np
import os

# Set up page layout and title
st.set_page_config(
    page_title="MGC Aurora Heights — Sales Assistant",
    page_icon="🏢",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load or train the lead scoring model automatically
@st.cache_resource
def load_lead_model():
    model_path = os.path.join(BASE_DIR, "lead_model.joblib")
    if not os.path.exists(model_path):
        with st.spinner("Training lead scoring model for the first time..."):
            import subprocess
            subprocess.run(["python", os.path.join(BASE_DIR, "lead_scoring.py")], check=True)
    return joblib.load(model_path)


def load_document_assistant():
    """Initializes the document assistant module if API key is present."""
    try:
        from document_assistant import create_assistant
        return create_assistant()
    except Exception as e:
        return None, str(e)


# Main App Title
st.title("🏢 MGC Aurora Heights — Sales Assistant")
st.markdown("*Internal tool for the MGC sales team*")

# Navigation tabs for Q&A and Lead Scoring
tab1, tab2 = st.tabs(["📄 Document Assistant", "📊 Lead Scoring"])

# Tab 1: Document Assistant
with tab1:
    st.header("Ask about MGC Aurora Heights")
    st.markdown(
        "Ask any question about pricing, payment plans, or booking policy. "
        "Answers are grounded in the official MGC documents."
    )

    # Initialize chat message state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat prompt input
    if prompt := st.chat_input("e.g. What's the base price of a 2-bed in Block B?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call document assistant module
        res = load_document_assistant()
        if isinstance(res, tuple):
            ask, err_msg = res
        else:
            ask, err_msg = res, None

        if ask:
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    try:
                        answer = ask(prompt)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"⚠️ API Error: {e}\n\nPlease verify your `GOOGLE_API_KEY` in the `.env` file.")
        else:
            st.error(
                f"⚠️ Document assistant not initialized: {err_msg}\n\n"
                "Please set a valid `GOOGLE_API_KEY` in your `.env` file and refresh the page."
            )

# Tab 2: Lead Scoring Form
with tab2:
    st.header("Score a Lead")
    st.markdown(
        "Enter a lead's details to see their predicted conversion probability."
    )

    model_data = load_lead_model()

    if model_data is None:
        st.error(
            "Lead model not found. Run `python lead_scoring.py` first to train it."
        )
    else:
        model = model_data["model"]
        label_encoders = model_data["label_encoders"]
        feature_columns = model_data["feature_columns"]
        categorical_columns = model_data["categorical_columns"]

        # Lead input form layout
        with st.form("lead_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                source = st.selectbox("Lead Source", [
                    "Facebook Ads", "Property Portal", "Google Search",
                    "Instagram", "Referral", "Walk-in",
                    "WhatsApp Campaign", "Expo Stall", "Billboard",
                ])
                city = st.selectbox("City", [
                    "Islamabad", "Rawalpindi", "Lahore", "Karachi",
                    "Peshawar", "Faisalabad", "Multan", "Gujranwala",
                    "Abbottabad",
                ])
                area = st.selectbox("Area", [
                    "Unknown", "Bahria Town", "DHA", "B-17", "Top City",
                    "Gulberg", "F-Sectors", "G-Sectors", "E-Sectors",
                    "I-Sectors", "Blue Area", "Hayatabad", "Saddar",
                    "Model Town",
                ])
                property_type = st.selectbox("Property Type", [
                    "Apartment", "Plot", "Villa", "Commercial Shop",
                    "Penthouse", "Farmhouse",
                ])

            with col2:
                budget = st.number_input("Budget (PKR Lacs)", min_value=0.0, value=150.0, step=10.0)
                bedrooms = st.number_input("Bedrooms", min_value=0, max_value=6, value=2)
                first_response = st.number_input(
                    "First Response (minutes)", min_value=0.0, value=30.0, step=5.0
                )
                agent_exp = st.number_input(
                    "Agent Experience (years)", min_value=0.0, value=3.0, step=0.5
                )

            with col3:
                calls_made = st.number_input("Calls Made", min_value=0, value=2)
                call_seconds = st.number_input("Total Call Seconds", min_value=0.0, value=120.0)
                whatsapp = st.number_input("WhatsApp Replies", min_value=0, value=2)
                site_visits = st.number_input("Site Visits", min_value=0, value=0)

            col4, col5 = st.columns(2)
            with col4:
                is_overseas = st.checkbox("Overseas Pakistani")
                referred = st.checkbox("Referred by existing client")
            with col5:
                financing = st.checkbox("Has financing approved")

            submitted = st.form_submit_button("🔍 Score This Lead", use_container_width=True)

        if submitted:
            # Construct feature dictionary in exact model schema order
            input_data = {
                "source": source,
                "city": city,
                "area": area,
                "property_type": property_type,
                "budget_pkr_lac": budget,
                "bedrooms": bedrooms,
                "first_response_minutes": first_response,
                "calls_made": calls_made,
                "total_call_seconds": call_seconds,
                "whatsapp_replies": whatsapp,
                "site_visits": site_visits,
                "agent_experience_years": agent_exp,
                "is_overseas": int(is_overseas),
                "referred_by_existing_client": int(referred),
                "has_financing_approved": int(financing),
            }

            # Apply label encoders trained during model fitting
            for col in categorical_columns:
                le = label_encoders[col]
                val = input_data[col]
                if val in le.classes_:
                    input_data[col] = le.transform([val])[0]
                else:
                    input_data[col] = 0

            # Predict probability using Random Forest classifier
            features = np.array([[input_data[col] for col in feature_columns]])
            probability = model.predict_proba(features)[0][1]

            # Render conversion likelihood badge and action item
            st.divider()

            if probability >= 0.7:
                st.success(f"### 🟢 High Priority — {probability*100:.1f}% conversion likelihood")
                st.markdown("**Action:** Call this lead immediately.")
            elif probability >= 0.4:
                st.warning(f"### 🟡 Medium Priority — {probability*100:.1f}% conversion likelihood")
                st.markdown("**Action:** Follow up within 24 hours.")
            else:
                st.info(f"### 🔵 Low Priority — {probability*100:.1f}% conversion likelihood")
                st.markdown("**Action:** Add to nurture campaign.")
