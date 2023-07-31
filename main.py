import re
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


def get_next_offset(content_range):
    """
    Calculate the next offset for pagination based on the 'Content-Range' header.

    :param content_range: The 'Content-Range' header from the HTTP response.
    :return: The next offset for pagination.
    """
    # Parse the 'Content-Range' header.
    start, end, total = re.match(r"(\d+)-(\d+)/(\*|\d+)", content_range).groups()

    next_offset = int(end) + 1

    return next_offset


def make_request(path, *params):
    url = f"{API_URL}{path}?{'&'.join(params)}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:

        content_range = response.headers["Content-Range"]

        return response.json(), content_range
    else:
        st.error(f"Request failed with status code {response.status_code}: {response.text}")


def get_date_of_previous_sunday(input_date, weeks_before=1):
    return get_date_of_previous_day(input_date, 6, weeks_before)


def get_date_of_previous_day(input_date, target_weekday, weeks_before=1):
    # Ensure target_weekday is in the range 0 to 6
    if not 0 <= target_weekday <= 6:
        raise ValueError('target_weekday should be in the range 0 (Monday) to 6 (Sunday)')

    date_weeks_before = input_date - timedelta(weeks=weeks_before)
    days_to_subtract = (date_weeks_before.weekday() - target_weekday) % 7

    return date_weeks_before - timedelta(days=days_to_subtract)


def date_to_unix_ms(date):
    """Convert date to unix timestamp in ms"""
    dt = datetime(date.year, date.month, date.day)
    return int(dt.timestamp() * 1000)


def filter_values(dataframe, column, min_val, max_val):
    """
    Filter dataframe based on a range (min_val, max_val) on a specific column.

    :param dataframe: DataFrame to be filtered
    :param column: Name of the column to consider for filtering
    :param min_val: Minimum acceptable value
    :param max_val: Maximum acceptable value
    :return: Filtered DataFrame
    """
    return dataframe[(dataframe[column] >= min_val) & (dataframe[column] <= max_val)]


@st.cache_data
def fetch_and_process_data(start, end, tz='America/Los_Angeles'):
    """
    :type end: datetime
    :type start: datetime
    :param tz: Timezone to Use for all DateTimes
    """
    # Convert start and end to the provided timezone
    start = datetime.combine(start, datetime.min.time()).astimezone(pytz.timezone(tz))
    end = datetime.combine(end, datetime.min.time()).astimezone(pytz.timezone(tz))

    start_ums = date_to_unix_ms(start)
    end_ums = date_to_unix_ms(end)

    all_rolls = fetch_all_rolls(start_ums, end_ums)

    if all_rolls:
        dataframe = process_data(all_rolls, tz)
        return dataframe, get_min_daily_roll(dataframe)
    else:
        st.error('Error: Unable to retrieve data.')
        st.stop()
        return None, None


def fetch_all_rolls(start_ums, end_ums, limit=500):
    all_rolls = []
    next_offset = 0

    while True:
        rolls, content_range = make_request("rolls",
                                            "select=unix_milliseconds,dice_value,"
                                            "channel:channels(name:channel_name),"
                                            "username:usernames(name:username)",
                                            f"unix_milliseconds=gte.{start_ums}",
                                            f"unix_milliseconds=lte.{end_ums}",
                                            f"limit={limit}",
                                            f"offset={next_offset}")

        if rolls:
            all_rolls.extend(rolls)
            next_offset = get_next_offset(content_range)
        else:
            break

    return all_rolls


def process_data(rolls, tz):
    dataframe = pd.DataFrame(rolls)
    dataframe["date_time"] = pd.to_datetime(dataframe["unix_milliseconds"], unit='ms', utc=True)
    dataframe["date_time"] = dataframe["date_time"].dt.tz_convert(tz)  # Convert to the provided timezone
    dataframe["channel"] = dataframe["channel"].apply(lambda x: x['name'])
    dataframe["username"] = dataframe["username"].apply(lambda x: x['name'])
    dataframe.set_index('date_time', inplace=True)
    dataframe.sort_index(inplace=True)
    return filter_values(dataframe, "dice_value", 0, 100)


def get_min_daily_roll(dataframe):
    # Calculate the minimum dice roll per day
    min_roll_index = dataframe.groupby(dataframe.index.date)['dice_value'].idxmin()
    return dataframe.loc[min_roll_index]


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
    counts.columns = ['username', 'count']
    fig = px.bar(counts, x='username', y='count', labels={'username': 'User', 'count': 'Number of Rolls'},
                 title='Roll Call: Frequency of User Rolls')
    st.plotly_chart(fig)


def first_rolls_per_day(dataframe):
    df_reset = dataframe.reset_index()
    df_reset['date'] = df_reset['date_time'].dt.date
    idx = df_reset.groupby('date')['unix_milliseconds'].idxmin()
    earliest_roll_count = df_reset.loc[idx, 'username'].value_counts().reset_index()
    earliest_roll_count.columns = ['username', 'count']
    fig = px.bar(earliest_roll_count, x='username', y='count',
                 title='Early Bird Rollers - Users and the Number of Times They Rolled First',
                 labels={'username': 'User', 'count': 'Number of Initial Rolls'})
    st.plotly_chart(fig)


def last_rolls_per_day(dataframe):
    df_reset = dataframe.reset_index()
    df_reset['date'] = df_reset['date_time'].dt.date
    idx = df_reset.groupby('date')['unix_milliseconds'].idxmax()
    latest_roll_count = df_reset.loc[idx, 'username'].value_counts().reset_index()
    latest_roll_count.columns = ['username', 'count']
    fig = px.bar(latest_roll_count, x='username', y='count',
                 title='Down to the Wire - Users and the Number of Times They Rolled Last',
                 labels={'username': 'User', 'count': 'Number of Final Rolls'})
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
                 title='Weekday Wonders: Average Daily Dice Value',
                 labels={'x': 'Day of Week', 'y': 'Average Dice Value'})
    st.plotly_chart(fig)


def weekday_wonders_average_number_daily_rolls(dataframe):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    dataframe.index = pd.to_datetime(dataframe.index)

    dataframe['date'] = dataframe.index.date

    day_of_week = dataframe.groupby([dataframe['date'], dataframe.index.dayofweek]).count()['dice_value'].groupby(
        level=1).mean()

    day_of_week = day_of_week.round(2)

    day_of_week.index = day_of_week.index.map(dict(zip(range(7), days)))

    fig = px.bar(x=day_of_week.index, y=day_of_week.values,
                 title='Weekday Wonders: Average Number of Daily Dice Rolls',
                 labels={'x': 'Day of Week', 'y': 'Average Dice Rolls'})

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
    interpolated_data = avg_roll_per_date.reindex(full_date_range).interpolate(method='linear').round(2)

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


def plot(dataframe):
    number_of_rolls_per_user(dataframe)
    plot_avg_user_roll(dataframe)

    dice_value_histogram(dataframe)
    plot_roll_probability(dataframe)

    plot_avg_roll_per_date(dataframe)
    plot_weekday_analysis(dataframe)
    weekday_wonders_average_number_daily_rolls(dataframe)

    first_rolls_per_day(dataframe)
    last_rolls_per_day(dataframe)


def overall_metrics(current_df, current_df_min):
    total_rolls = current_df.shape[0]

    total_users = current_df['username'].nunique()

    total_days = pd.Series(current_df.index.date).nunique()

    rolls_per_user = current_df['username'].value_counts()
    most_active_user = rolls_per_user.idxmax()

    coffee_maker_counts = current_df_min['username'].value_counts()
    top_coffee_maker = coffee_maker_counts.idxmax()

    avg_rolls_per_user = total_rolls / total_users if total_users != 0 else 0
    avg_rolls_per_user = round(avg_rolls_per_user, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rolls", total_rolls, None)
    col2.metric("Total Users", total_users, None)
    col3.metric("Total Days", total_days, None)
    col1.metric("Most Active User", most_active_user, None)
    col2.metric("Top Coffee Maker", top_coffee_maker, None)
    col3.metric("Average Rolls Per User", avg_rolls_per_user, None)


def compare_weekly_metrics(current_df, current_df_min, last_df):
    # Function to calculate metrics
    def calc_metrics(df):
        total_rolls = df.shape[0]
        total_users = df['username'].nunique()
        rolls_per_user = df['username'].value_counts()
        most_active_user = rolls_per_user.idxmax()
        avg_rolls_per_user = round(total_rolls / total_users, 1) if total_users != 0 else 0
        roll_variance = df['dice_value'].var().round(2) if total_rolls != 0 else 0
        return total_rolls, total_users, avg_rolls_per_user, most_active_user, roll_variance

    # Get metrics
    current_rolls, current_users, current_avg_rolls, current_most_active, current_roll_variance = calc_metrics(
        current_df)
    last_rolls, last_users, last_avg_rolls, last_most_active, last_roll_variance = calc_metrics(last_df)

    coffee_maker_counts = current_df_min['username'].value_counts()
    top_coffee_maker = coffee_maker_counts.idxmax()

    # Calculate deltas
    delta_rolls = current_rolls - last_rolls
    delta_users = current_users - last_users
    delta_avg_rolls = current_avg_rolls - last_avg_rolls
    delta_roll_variance = (current_roll_variance - last_roll_variance).round(2)

    # Set up columns for Streamlit
    col1, col2, col3 = st.columns(3)

    # Display metrics in columns
    col1.metric("Total Rolls", current_rolls, delta_rolls)
    col1.metric("Most Active User", current_most_active)

    col2.metric("Total Users", current_users, delta_users)
    col2.metric("Top Coffee Maker", top_coffee_maker, None)

    col3.metric("Roll Variance", current_roll_variance, delta_roll_variance)
    col3.metric("Average Rolls Per User", current_avg_rolls, delta_avg_rolls)


st.subheader("Weekly Metrics")

# Current week's data
current_today = datetime.today()
current_sunday = get_date_of_previous_day(current_today, 0, 0)
currentWeekDf, currentWeekDfMin = fetch_and_process_data(current_sunday, current_today)

# Past week's data
past_today = current_today - timedelta(days=7)
past_sunday = get_date_of_previous_day(past_today, 0, 0)
past_saturday = past_sunday + timedelta(days=6)  # The end of the past week

pastWeekDf, pastWeekDfMin = fetch_and_process_data(past_sunday, past_saturday)

# Comparison of the current week's metrics to the past week's
compare_weekly_metrics(currentWeekDf, currentWeekDfMin, pastWeekDf)

if confirmDatesButton:
    if len(dates) < 2:  # If the user did not select a second date
        st.warning('Please select a second date.')
    elif dates[0] > dates[1]:
        st.error('Error: End date must fall after start date.')
    else:
        start_date, end_date = dates
        end_date += timedelta(days=1)  # make end_date inclusive

        df, df_min = fetch_and_process_data(start_date, end_date)

        st.subheader("Metrics for Selected Date Range")

        overall_metrics(df, df_min)

        tab1, tab2 = st.tabs(["Plots", "Tables"])

        with tab1:
            plot(df)

        with tab2:
            st.dataframe(df)
            st.dataframe(df_min)

if resetButton:
    st.balloons()
    st.cache_data.clear()
