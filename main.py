from datetime import timedelta, datetime

import pandas as pd
import plotly.express as px
import pytz
import requests
import streamlit as st

import layout
from collections import Counter

import plotly.graph_objects as go

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


def filter_values(df, column, min_val, max_val):
    """
    Filter dataframe based on a range (min_val, max_val) on a specific column.

    :param df: DataFrame to be filtered
    :param column: Name of the column to consider for filtering
    :param min_val: Minimum acceptable value
    :param max_val: Maximum acceptable value
    :return: Filtered DataFrame
    """
    return df[(df[column] >= min_val) & (df[column] <= max_val)]


@st.cache_data
def fetch_and_process_data(start, end, tz='America/Los_Angeles'):
    """
    :type end: datetime
    :type start: datetime
    :param tz: Timezone to Use for all DateTimes
    """
    # Make sure start and end are in the provided timezone

    start = datetime.combine(start, datetime.min.time())
    end = datetime.combine(end, datetime.min.time())

    start = start.astimezone(pytz.timezone(tz))
    end = end.astimezone(pytz.timezone(tz))

    start_ums = date_to_unix_ms(start)
    end_ums = date_to_unix_ms(end)

    rolls = make_request("rolls",
                         "select=unix_milliseconds,dice_value,"
                         "channel:channels(name:channel_name),"
                         "username:usernames(name:username)",
                         f"unix_milliseconds=gte.{start_ums}",
                         f"unix_milliseconds=lte.{end_ums}")

    if rolls:
        dataframe = pd.DataFrame(rolls)
        dataframe["date_time"] = pd.to_datetime(dataframe["unix_milliseconds"], unit='ms', utc=True)
        dataframe["date_time"] = dataframe["date_time"].dt.tz_convert(tz)  # Convert to the provided timezone
        dataframe["channel"] = dataframe["channel"].apply(lambda x: x['name'])
        dataframe["username"] = dataframe["username"].apply(lambda x: x['name'])
        dataframe.set_index('date_time', inplace=True)
        dataframe.sort_index(inplace=True)

        dataframe = filter_values(dataframe, "dice_value", 0, 100)

        # Calculate the minimum dice roll per day
        min_roll_index = dataframe.groupby(dataframe.index.date)['dice_value'].idxmin()
        df_min_daily_roll = dataframe.loc[min_roll_index]

        return dataframe, df_min_daily_roll
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


def dice_value_histogram(dataframe):
    fig = px.histogram(dataframe, x="dice_value", nbins=100, title='Dicey Distributions: A Look at Dice Values',
                       labels={'dice_value': 'Dice Value', 'Frequency': 'Frequency'})
    st.plotly_chart(fig)


def number_of_rolls_per_user(dataframe):
    counts = dataframe['username'].value_counts().reset_index()
    fig = px.bar(counts, x='index', y='username', labels={'index': 'User', 'username': 'Number of Rolls'},
                 title='Roll Call: Frequency of User Rolls')
    st.plotly_chart(fig)


def first_rolls_per_day(dataframe):
    df_reset = dataframe.reset_index()
    df_reset['date'] = df_reset['date_time'].dt.date
    idx = df_reset.groupby('date')['unix_milliseconds'].idxmin()
    earliest_roll_count = df_reset.loc[idx, 'username'].value_counts()
    fig = px.bar(earliest_roll_count.reset_index(), x='index', y='username',
                 title='Early Bird Rollers - Users With the Initial Daily Roll',
                 labels={'index': 'User', 'username': 'Number of Initial Rolls'})
    st.plotly_chart(fig)


def last_rolls_per_day(dataframe):
    df_reset = dataframe.reset_index()
    df_reset['date'] = df_reset['date_time'].dt.date
    idx = df_reset.groupby('date')['unix_milliseconds'].idxmax()
    latest_roll_count = df_reset.loc[idx, 'username'].value_counts()
    fig = px.bar(latest_roll_count.reset_index(), x='index', y='username',
                 title='Down to the Wire - Users With the Final Daily Roll',
                 labels={'index': 'User', 'username': 'Number of Final Rolls'})
    st.plotly_chart(fig)


def plot_roll_probability(dataframe):
    roll_counts = Counter(dataframe['dice_value'])
    total_rolls = sum(roll_counts.values())
    roll_probabilities = {k: v / total_rolls for k, v in roll_counts.items()}
    fig = px.bar(x=list(roll_probabilities.keys()), y=list(roll_probabilities.values()),
                 title='Roll of the Dice: Probability Breakdown',
                 labels={'x': 'Dice Value', 'y': 'Probability'})
    fig.update_yaxes(tickformat=".2%")  # Formatting the y-axis ticks as percentages
    st.plotly_chart(fig)


def plot_weekday_analysis(dataframe):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = dataframe['dice_value'].groupby(dataframe.index.dayofweek).mean().round(2)
    day_of_week.index = day_of_week.index.map(dict(zip(range(7), days)))  # mapping numerical days to weekday names
    fig = px.bar(x=day_of_week.index, y=day_of_week.values,
                 title='Weekday Wonders: Average Dice Value',
                 labels={'x': 'Day of Week', 'y': 'Average Dice Value'})
    st.plotly_chart(fig)


def plot_avg_user_roll(dataframe):
    avg_roll_per_user = dataframe.groupby('username')['dice_value'].mean().round(2)
    avg_roll_per_user = avg_roll_per_user.sort_values(ascending=False)  # sort in descending order

    roll_count_per_user = dataframe['username'].value_counts().loc[avg_roll_per_user.index]  # count rolls per user

    fig = px.bar(x=avg_roll_per_user.index, y=avg_roll_per_user.values,
                 title='Users and Their Lucky Numbers: Average Roll per User',
                 labels={'x': 'User', 'y': 'Average Roll'})

    # add line plot
    fig.add_trace(go.Scatter(x=roll_count_per_user.index, y=roll_count_per_user.values,
                             mode='lines', name='Number of Rolls'))

    # update legend position
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    st.plotly_chart(fig)


def plot_avg_roll_per_date(dataframe):
    avg_roll_per_date = dataframe.resample('D')['dice_value'].mean().round(2)

    # Create a full date range and interpolate
    full_date_range = pd.date_range(start=avg_roll_per_date.index.min(),
                                    end=avg_roll_per_date.index.max())
    interpolated_data = avg_roll_per_date.reindex(full_date_range).interpolate(method='linear')

    # Create a new figure
    fig = go.Figure()

    # Add interpolated data to the figure as a line
    fig.add_trace(go.Scatter(x=interpolated_data.index, y=interpolated_data.values,
                             mode='lines+markers', name='Interpolated Data'))

    # Add actual data to the figure as markers
    fig.add_trace(go.Scatter(x=avg_roll_per_date.index, y=avg_roll_per_date.values,
                             mode='markers', name='Actual Data'))

    fig.update_layout(title='Rolling Through Time: Average Roll per Calendar Date',
                      xaxis_title='Date',
                      yaxis_title='Average Roll',
                      showlegend=True)

    st.plotly_chart(fig)


def plot(dataframe, dataframe_min):
    dice_value_histogram(dataframe)
    first_rolls_per_day(dataframe)
    last_rolls_per_day(dataframe)
    plot_roll_probability(dataframe)
    plot_weekday_analysis(dataframe)

    number_of_rolls_per_user(dataframe)
    plot_avg_user_roll(dataframe)

    plot_avg_roll_per_date(dataframe)


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
