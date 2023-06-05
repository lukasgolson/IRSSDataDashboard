from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import layout

layout.apply_layout()

API_URL = "https://hacvffgmaquyyeiusnbi.supabase.co/rest/v1/"
API_KEY = st.secrets['api_key']  # Get the API key from secrets
HEADERS = {'apikey': API_KEY}

with st.sidebar:
    st.title("Java Jotter Dashboard")
    st.write(
        "This dashboard analyses dice roll data between the given dates. "
        "In the IRSS, each member who drinks coffee must roll a dice. "
        "The lowest roller on each day has to make the coffee for the next morning. "
        "Data is scrapped in near-realtime through our JavaJotter bot.")


def make_request(path, *params):
    url = f"{API_URL}{path}?{'&'.join(params)}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Request failed with status code {response.status_code}: {response.text}")


def get_date_of_previous_sunday(input_date, weeks_before=1):
    date_weeks_before = input_date - timedelta(weeks=weeks_before)
    days_to_subtract = (date_weeks_before.weekday() + 1) % 7
    return date_weeks_before - timedelta(days=days_to_subtract)


def date_to_unix_ms(date):
    """Convert date to unix timestamp in ms"""
    dt = datetime(date.year, date.month, date.day)
    return int(dt.timestamp() * 1000)


@st.cache_data
def fetch_and_process_data(start_date, end_date):
    start_ums = date_to_unix_ms(start_date)
    end_ums = date_to_unix_ms(end_date)
    rolls = make_request("rolls",
                         "select=unix_milliseconds,dice_value,"
                         "channel:channels(name:channel_name),"
                         "username:usernames(name:username)",
                         f"unix_milliseconds=gte.{start_ums}",
                         f"unix_milliseconds=lte.{end_ums}")

    if rolls:
        df = pd.DataFrame(rolls)
        df["date_time"] = pd.to_datetime(df["unix_milliseconds"], unit='ms')
        df["channel"] = df["channel"].apply(lambda x: x['name'])
        df["username"] = df["username"].apply(lambda x: x['name'])
        df.set_index('date_time', inplace=True)
        df.sort_index(inplace=True)

        # Calculate the minimum dice roll per day
        min_roll_index = df.groupby(df.index.date)['dice_value'].idxmin()
        df_min_daily_roll = df.loc[min_roll_index]

        return df, df_min_daily_roll
    else:
        st.error('Error: Unable to retrieve data.')
        st.stop()
        return None, None


start_date = get_date_of_previous_sunday(datetime.today(), 4)
end_date = datetime.today()

with st.sidebar:
    dates = st.date_input('Select start and end date:', [start_date, end_date])
    confirmDatesButton = st.button('Confirm Dates')
    resetButton = st.button('Invalidate Cache')


def plot(df, df_min):
    # Distribution of Dice Values
    fig = px.histogram(df, x="dice_value", nbins=100, title='Distribution of Dice Values')
    fig.update_xaxes(title_text='Dice Value')
    fig.update_yaxes(title_text='Frequency')
    st.plotly_chart(fig)

    # Number of Rolls Per User
    fig = px.bar(df['username'].value_counts().reset_index(), x='index', y='username', title='Number of Rolls Per User')
    fig.update_xaxes(title_text='User')
    fig.update_yaxes(title_text='Number of Rolls')
    st.plotly_chart(fig)

    # Rolls Over Time
    rolls_per_day = df.resample('D').count()

    fig = px.line(x=rolls_per_day.index, y=rolls_per_day['dice_value'], title='Rolls Over Time')
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Number of Rolls')
    st.plotly_chart(fig)

    # Lowest Daily Rolls
    df_min_reset = df_min.reset_index()

    fig = px.bar(x=df_min_reset.index, y=df_min_reset['dice_value'], title='Lowest Daily Rolls')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Dice Value')
    st.plotly_chart(fig)

    # Roll Consistency per User
    roll_count = df['username'].value_counts()
    fig = px.bar(roll_count.reset_index(), x='index', y='username', title='Roll Consistency per User')
    fig.update_xaxes(title_text='Username')
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig)

    # Roll Closest to Specific Time
    df['time_diff'] = abs(df.index - df.index.to_series().dt.normalize().add(pd.DateOffset(hours=15)))
    idx = df.groupby(df.index.date)['time_diff'].idxmin()
    closest_to_three_pm = df.loc[idx, 'username']
    table_data = closest_to_three_pm.reset_index().groupby('username')['date_time'].apply(list).apply(pd.Series).T
    st.dataframe(table_data, column_config={"name": "Closest Rolls to 3PM"})
    df.drop(columns=['time_diff'], inplace=True)

    df_reset = df.reset_index()
    df_reset['date'] = df_reset['date_time'].dt.date
    idx = df_reset.groupby('date')['unix_milliseconds'].idxmax()
    latest_roll_of_day = df_reset.loc[idx]
    latest_roll_count = latest_roll_of_day['username'].value_counts()
    fig = px.bar(latest_roll_count.reset_index(), x='index', y='username', title='Latest Roll of the Day')
    st.plotly_chart(fig)

    # Roll Probability
    roll_probabilities = Counter(df['dice_value'])
    total_rolls = sum(roll_probabilities.values())
    for k in roll_probabilities:
        roll_probabilities[k] /= total_rolls
    fig = px.bar(x=list(roll_probabilities.keys()), y=list(roll_probabilities.values()), title='Roll Probability')
    fig.update_xaxes(title_text='Dice Value')
    fig.update_yaxes(title_text='Probability')
    st.plotly_chart(fig)

    # Day of the Week Analysis
    day_of_week = df['dice_value'].groupby(df.index.dayofweek).mean()
    fig = px.bar(day_of_week.reset_index(), x=day_of_week.index, y='dice_value', title='Day of the Week Analysis')
    fig.update_xaxes(title_text='Day of Week')
    fig.update_yaxes(title_text='Average Dice Value')
    st.plotly_chart(fig)

    # Average Roll per User
    avg_roll_per_user = df.groupby('username')['dice_value'].mean()
    fig = px.bar(avg_roll_per_user.reset_index(), x='username', y='dice_value', title='Average Roll per User')
    fig.update_xaxes(title_text='User')
    fig.update_yaxes(title_text='Average Roll')
    st.plotly_chart(fig)

    # Average Roll per Calendar Date
    avg_roll_per_date = df.resample('D')['dice_value'].mean().reset_index()

    fig = px.line(avg_roll_per_date, x=avg_roll_per_date.index, y='dice_value', title='Average Roll per Calendar Date')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Average Roll')
    st.plotly_chart(fig)


if confirmDatesButton:
    if len(dates) < 2:  # If the user did not select a second date
        st.warning('Please select a second date.')
    elif dates[0] > dates[1]:
        st.error('Error: End date must fall after start date.')
    else:
        start_date, end_date = dates
        end_date += timedelta(days=1)  # make end_date inclusive
        df, df_min = fetch_and_process_data(start_date, end_date)

       # st.dataframe(df)  # Display the full dataframe in the app
       # st.dataframe(df_min)  # Display the dataframe of minimum daily rolls

        plot(df, df_min)

if resetButton:
    st.cache_data.clear()
