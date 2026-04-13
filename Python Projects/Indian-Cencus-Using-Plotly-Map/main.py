import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px

# df = pd.read_csv("final_DF.csv")
df = pd.read_csv("/mount/src/python_growing_projects/Mini-Projects/Indian-Cencus-Using-Plotly-Map/final_DF.csv")



st.set_page_config(layout="wide")
l = list(df["State"].unique())
l.insert(0 , "Overall India")
st.sidebar.title("India Data Visulisation ")

selected_state = st.sidebar.selectbox("Select State" , l)
primary_parameter = st.sidebar.selectbox("Select Primary Parameter" , sorted(df.columns[6:]))
secondary_parameter = st.sidebar.selectbox("Select Secondary Parameter" , sorted(df.columns[6:]))

plot = st.sidebar.button("Plot" )

if plot:

    st.text("Size Represent Primary Parameter")
    st.text("Color Represent Secondary Parameter")
    if selected_state == "Overall India":
        # Plot for india
        fig = px.scatter_map(
                        df, 
                        lat="Latitude", 
                        lon="Longitude",
                        size=primary_parameter,
                        color=secondary_parameter,
                        size_max=35,
                        zoom=3,
                        map_style="carto-positron",
                        height=1080,
                        width=950,
                        hover_name="District"
        )
        st.plotly_chart(fig , use_container_width=True)
    else:
        state_df = df[df["State"] == selected_state]
        fig = px.scatter_map(
                        state_df, 
                        lat="Latitude", 
                        lon="Longitude",
                        size=primary_parameter,
                        color=secondary_parameter,
                        size_max=35,
                        zoom=6,
                        map_style="carto-positron",
                        height=1080,
                        width=950,
                        hover_name="District"
        )
        st.plotly_chart(fig , use_container_width=True)
   
