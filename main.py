import calendar
import re
from collections import Counter
from datetime import timedelta, datetime, timezone, time

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st

import layout

layout.apply_layout()

API_URL = st.secrets['supabase_url']  # Get the URL from secrets
API_KEY = st.secrets['supabase_key']  # Get the API key from secrets

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
    date = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)

    dt = datetime.combine(date, datetime.min.time())

    return int(dt.timestamp() * 1000)


def unix_ms_to_date(timestamp):
    """Convert unix timestamp in ms to datetime in the 'America/Los_Angeles' timezone"""
    dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    dt_la = dt.astimezone(pytz.timezone('America/Los_Angeles'))
    return dt_la


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
def fetch_and_process_data(starttime: datetime, endtime: datetime, tz='America/Los_Angeles'):
    """
    :type endtime: datetime
    :type starttime: datetime
    :param tz: Timezone to Use for all DateTimes
    """

    # Create a timezone object for the Pacific Time Zone
    pacific_tz = pytz.timezone(tz)

    # Make it timezone-aware
    starttime = pacific_tz.localize(starttime)
    endtime = pacific_tz.localize(endtime)

    start_ums = date_to_unix_ms(starttime)
    end_ums = date_to_unix_ms(endtime)

    if start_ums > end_ums:
        tmp = end_ums
        end_ums = start_ums
        start_ums = tmp
        st.warning("Start time greater than end time; flipping...")

    all_rolls = fetch_all_rolls(start_ums, end_ums)

    if len(all_rolls) > 0:
        dataframe = process_data(all_rolls, tz)

        return dataframe, get_min_daily_roll(dataframe)
    else:
        st.error('Error: Unable to retrieve data.')

        st.stop()
        return None, None


@st.cache_data
def make_request(path, *params):
    url = f"{API_URL}{path}?{'&'.join(params)}"
    HEADERS = {'apikey': API_KEY, 'Accept': 'application/json'}

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:

        json_response = response.json()

        return json_response
    else:
        st.error(f"Request failed with status code {response.status_code}: {response.text}")


@st.cache_data
def fetch_all_rolls(start_ums, end_ums):
    all_rolls = []
    offset = 0

    while True:
        rolls = make_request("rolls",
                             "select=unix_milliseconds,dice_value,"
                             "channel:channels(name:channel_name),"
                             "username:usernames(name:username)",
                             f"unix_milliseconds=gte.{start_ums}",
                             f"unix_milliseconds=lte.{end_ums}",
                             f"offset={offset}")

        if not rolls:
            break

        all_rolls.extend(rolls)
        offset += len(rolls)  # Increase offset by number of elements returned in this batch.

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


def plot_number_of_rolls_per_day(dataframe):
    rolls_per_day = dataframe.resample('D').count()['dice_value']
    fig = px.bar(x=rolls_per_day.index, y=rolls_per_day.values,
                 title='Dicey Days: Number of Rolls per Day',
                 labels={'x': 'Date', 'y': 'Number of Rolls'})
    st.plotly_chart(fig)


def plot_cumulative_rolls(dataframe):
    cumulative_rolls = dataframe.resample('D').count()['dice_value'].cumsum()
    fig = px.line(x=cumulative_rolls.index, y=cumulative_rolls.values,
                  title='Rolling Along: Cumulative Number of Rolls Over Time',
                  labels={'x': 'Date', 'y': 'Cumulative Number of Rolls'})
    st.plotly_chart(fig)


def plot_unique_users_per_day(dataframe):
    unique_users_per_day = dataframe.resample('D')['username'].nunique()
    fig = px.line(x=unique_users_per_day.index, y=unique_users_per_day.values,
                  title='A Dicey Crowd: Number of Unique Users per Day',
                  labels={'x': 'Date', 'y': 'Number of Unique Users'})
    st.plotly_chart(fig)


def plot_scatter(dataframe):
    user_rolls = dataframe['username'].value_counts()
    fig = px.scatter(x=user_rolls.index, y=user_rolls.values,
                     title='User Activity',
                     labels={'x': 'User', 'y': 'Total Rolls'})
    st.plotly_chart(fig)


def plot_heatmap(dataframe):
    dataframe.index = pd.to_datetime(dataframe.index)

    dataframe['DayOfWeek'] = dataframe.index.dayofweek
    dataframe['Month'] = dataframe.index.month

    day_order = [6, 0, 1, 2, 3, 4, 5]
    day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    month_order = list(range(1, 13))
    month_labels = list(calendar.month_abbr)[1:]  # Get abbreviated month names

    contributions = dataframe.groupby(['DayOfWeek', 'Month']).size().unstack().fillna(0)

    contributions = contributions.reindex(index=day_order, columns=month_order, fill_value=0)

    fig = ff.create_annotated_heatmap(z=contributions.values,
                                      x=month_labels,
                                      y=day_labels,
                                      showscale=True)

    fig.update_layout(title='Heatmap of Rolls',
                      xaxis=dict(title='Month'),
                      yaxis=dict(title='Day of the Week'))

    st.plotly_chart(fig)


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
        roll_std_dev = round(df['dice_value'].std(), 2) if total_rolls != 0 else 0
        return total_rolls, total_users, avg_rolls_per_user, most_active_user, roll_std_dev

    # Get metrics
    current_rolls, current_users, current_avg_rolls, current_most_active, current_roll_std_dev = calc_metrics(
        current_df)
    last_rolls, last_users, last_avg_rolls, last_most_active, last_roll_std_dev = calc_metrics(last_df)

    coffee_maker_counts = current_df_min['username'].value_counts()
    top_coffee_maker = coffee_maker_counts.idxmax()

    # Calculate deltas
    delta_rolls = current_rolls - last_rolls
    delta_users = current_users - last_users
    delta_avg_rolls = round(current_avg_rolls - last_avg_rolls, 2)
    delta_roll_std_dev = round(current_roll_std_dev - last_roll_std_dev, 2)

    # Set up columns for Streamlit
    col1, col2, col3 = st.columns(3)

    # Display metrics in columns
    col1.metric("Total Rolls", current_rolls, delta_rolls)
    col1.metric("Most Active User", current_most_active)

    col2.metric("Total Users", current_users, delta_users)
    col2.metric("Top Coffee Maker", top_coffee_maker, None)

    col3.metric("Roll Standard Deviation", current_roll_std_dev, delta_roll_std_dev)
    col3.metric("Average Rolls Per User", current_avg_rolls, delta_avg_rolls)


st.subheader("Weekly Metrics")

# Current week's data
today = datetime.today()
current_sunday = get_date_of_previous_day(today, 6, 1)
current_saturday = current_sunday + timedelta(days=6) if current_sunday < today else today
currentWeekDf, currentWeekDfMin = fetch_and_process_data(current_sunday, current_saturday)

# Past week's data
past_sunday = get_date_of_previous_day(today, 6, 2)
past_saturday = past_sunday + timedelta(days=6)
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

        start_datetime = datetime.combine(start_date, time())
        end_datetime = datetime.combine(end_date, time())

        df, df_min = fetch_and_process_data(start_datetime, end_datetime)

        st.subheader("Metrics for Selected Date Range")

        overall_metrics(df, df_min)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["User Activity Analysis", "Dice Roll Analysis", "Time Series Analysis", "Weekday Analysis",
             "General Plots", "Tables"])

        with tab1:
            number_of_rolls_per_user(df)
            plot_avg_user_roll(df)
            plot_unique_users_per_day(df)
            plot_number_of_rolls_per_day(df)
        with tab2:
            dice_value_histogram(df)
            plot_roll_probability(df)
            plot_avg_roll_per_date(df)
        with tab3:
            first_rolls_per_day(df)
            last_rolls_per_day(df)
            plot_cumulative_rolls(df)
        with tab4:
            plot_weekday_analysis(df)
            weekday_wonders_average_number_daily_rolls(df)
        with tab5:
            plot_scatter(df)
            plot_heatmap(df)

        with tab6:
            st.dataframe(df)
            st.dataframe(df_min)

if resetButton:
    st.balloons()
    st.cache_data.clear()
