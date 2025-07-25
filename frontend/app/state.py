import streamlit as st

class AppState:
    _defaults = {
        'raw_data': None,
        'validated_data': None,
        'assessment_results': None,
        'validation_report': None,
        'uploaded_file_name': None,
        'csv_content': None,  # Added to store the raw CSV string for the API
    }

    @staticmethod
    def initialize():
        for key, value in AppState._defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get_state(key):
        return st.session_state.get(key)

    @staticmethod
    def set_state(key, value):
        st.session_state[key] = value

    @staticmethod
    def reset_all():
        """Resets all session state variables to their default values."""
        for key, value in AppState._defaults.items():
            st.session_state[key] = value
