# dashboard.py
import pandas as pd
import streamlit as st
import time

st.title("MD Simulation Monitor")

placeholder = st.empty()

# cols = ["PotEng", "TotEng", "KinEng"]
# cols = ["Press"] 
cols = ["Density"] 
# cols = ["AvgNgh"] 

while True:
    try:
        df = pd.read_csv("/tmp/thermo.csv", delimiter=",")

        with placeholder.container():
            st.line_chart(
                df.set_index("Step")[cols]
            )
        time.sleep(0.5)

    except Exception:
        pass

