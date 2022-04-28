# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:10:31 2022

@author: duyhi
"""

# import libs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# Define functions

## Functions for styling
def style_negative(v, props=''):
    """ Style negative values in dataframe """
    try:
        # Apply style if value is negative
        return props if v < 0 else None
    except:
        pass   # try/except-pass to bypass raising error on the Video title column
        
def style_positive(v, props=''):
    """ Style positive values in dataframe """
    try:
        # Apply style if value is positive
        return props if v > 0 else None
    except:
        pass   # try/except-pass to bypass raising error on the Video title column
    
## Other functions
### For categorizing subscribers' country into main/most popular groups
def audience_simple(country):
    """ Show top countries """
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'


# Load data + Feature engineering
# Streamlit's cache data to load in once, don't have to load every time you reload the page
@st.cache
def load_data():
    # skip the first row, include everything else
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:, :]
    # Rename all columns
    df_agg.columns = ['Video','Video title','Video publish time','Comments added','Shares','Dislikes','Likes',
                          'Subscribers lost','Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed','Average view duration',
                          'Views','Watch time (hours)','Subscribers','Your estimated revenue (USD)','Impressions','Impressions ctr(%)']
    # Converts dates and view times to datetime objects (dates and seconds)
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'])
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    # Add in some extra stats
    #Engagement ratio = # comments + likes + shares + dislikes / # views
    df_agg['Engagement ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Likes'] + df_agg['Dislikes']) / df_agg['Views']
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending=False, inplace=True)
    
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('All_Comments_Final.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    df_time['Date'] = pd.to_datetime(df_time['Date'])
    return df_agg, df_agg_sub, df_comments, df_time


# create dataframes from the function
df_agg, df_agg_sub, df_comments, df_time = load_data()

# Engineer data
df_agg_diff = df_agg.copy()
# Get 12-month medians of metrics
#find the 12-month date from the most recent video
metric_date_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)
#For all videos published after the 12-month date, get the median values of those videos' agg stats
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12mo].median()

# Create differences from the median for the values
#get just numeric columns
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
#get difference between the value and the median, then divide by the median to get a percent ratio
#comparing the particular video's stats to the median stat to gauge their performance across different categories
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:, numeric_cols] - median_agg).div(median_agg)

# Set up data for time-series graphs for individual videos
## Merge daily data with publish data to get delta/difference
df_time_diff = pd.merge(df_time, df_agg.loc[:, ['Video', 'Video publish time']], left_on='External Video ID', right_on='Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days
## Get data for last 12 months only
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

## Get daily view data of first 30 days, median & percentiles
### Use pivot table: group all Views values of all videos by days_published, apply aggfuncs to get mean, median, 20/80th percentiles
### We'll use this to look at a specific video's performance compared to other videos in the same timeframe (first 30 days since publish)
views_days = pd.pivot_table(df_time_diff_yr, index='days_published', values='Views',
                            aggfunc=[np.mean, np.median, lambda x: np.percentile(x, 80), lambda x: np.percentile(x, 20)]).reset_index()
### Rename columns
views_days.columns = ['days_published', 'mean_views', 'median_views', '80pct_views', '20pct_views']
### Get only data for the 30 days after publish date
views_days = views_days[views_days['days_published'].between(0, 30)]
### Calculate cumulative (ie total) views
views_cumulative = views_days.loc[:, ['days_published', 'median_views', '80pct_views', '20pct_views']]
### use cumsum() func to calculate cumulative
views_cumulative.loc[:, ['median_views', '80pct_views', '20pct_views']] = views_cumulative.loc[:,['median_views', '80pct_views', '20pct_views']].cumsum()


# Build dashboard
# Add sidebar
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics', 'Individual Video Analysis'))

# Distinguish between displaying Total picture (Aggregate) vs Individual video
## Total picture
if add_sidebar == 'Aggregate Metrics':
    #st.write('Agg')
    # Get the most relevant metrics that we want to show on top; 10 metrics, 'Video publish time' is for getting the differential
    df_agg_metrics = df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)',
                             'Average % viewed','Avg_duration_sec','Engagement ratio','Views / sub gained']]
    # Get the 6- and 12-month pivot dates from the most recent video, use for comparison
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    # Get the median values for all aggregated stat categories for all videos spanning back 6- and 12-month period
    metric_medians_6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median()
    metric_medians_12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median()
    
    #st.metric('Views', metric_medians_6mo['Views'], 500)   # One way to show metric
    # Columns in streamlit
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    
    count = 0
    for i in metric_medians_6mo.index:
        with columns[count]:
            delta_val = (metric_medians_6mo[i] - metric_medians_12mo[i]) / metric_medians_12mo[i]   # give the % change of last 6mo vs last 12mo
            st.metric(label=i, value=round(metric_medians_6mo[i], 1), delta="{:.2%}".format(delta_val))
            # Move on to next column
            count += 1
            # When have reached 5 columns for row, reset to make another 5-column row
            if count >= 5:
                count = 0
    
    # Convert 'Video publish time' to date format (remove timestamps) into new column 'Publish_date'
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
    df_agg_diff_final = df_agg_diff.loc[:, ['Video title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)',
                                            'Average % viewed','Avg_duration_sec','Engagement ratio','Views / sub gained']]
    
    # Get a list of the numeric columns (only numeric columns can have median, get index to get column names)
    df_agg_numeric_list = df_agg_diff_final.median().index.tolist()
    # Create a dictionary to keep the formatting rules
    df_to_pct = {}
    for metric in df_agg_numeric_list:         # For each of the column/metric,
        df_to_pct[metric] = '{:.1%}'.format    # it will have this format to % function
    
    # Add dataframe to streamlit
    # Styling: style.applymap; props: can be tuples or CSS string
    # Style so that negative numbers are red, positive numbers are green, convert all values to %s
    st.dataframe(df_agg_diff_final.style.applymap(style_negative, props='color:red;')
                                         .applymap(style_positive, props='color:green;')
                                         .format(df_to_pct))

## Individual video
if add_sidebar == 'Individual Video Analysis':
    #st.write('Ind')
    # Create a dropdown selection
    ## Convert all video titles into a tuple for streamlit.selectbox
    videos = tuple(df_agg['Video title'])
    video_select = st.selectbox('Pick a video:', videos)
    
    # Get data for individual video
    ## Filter by the video title selected, only get data of that video
    vid_agg_filtered = df_agg[df_agg['Video title'] == video_select]
    vid_agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
    ## Categorize subscribers' country into 3 main groups: USA, India, Other
    vid_agg_sub_filtered['Country'] = vid_agg_sub_filtered['Country Code'].apply(audience_simple)
    ## Sort by Is Subscribed so graph shows up consistently
    vid_agg_sub_filtered.sort_values('Is Subscribed', inplace=True)
    
    # Create graph for Views demographics
    fig = px.bar(vid_agg_sub_filtered, x='Views', y='Is Subscribed', color='Country',orientation='h')
    st.plotly_chart(fig)
    
    # Time Series Views performance graph
    ## get data for the selected video, and filter to only get the first 30 days since video is published
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    
    ## Create graph
    fig2 = go.Figure()    # Create plotly graph object Figure
    ### Add each individual line to the graph
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                   mode='lines',
                   name='20th percentile', line=dict(color='purple', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                              mode='lines',
                              name='50th percentile', line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                              mode='lines',
                              name='80th percentile', line=dict(color='royalblue', dash='dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                              mode='lines',
                              name='Current Video', line=dict(color='firebrick', width=8)))

    fig2.update_layout(title='View comparision of first 30 days',
                       xaxis_title='Days Since Published',
                       yaxis_title='Cumulative views')
    
    st.plotly_chart(fig2)




