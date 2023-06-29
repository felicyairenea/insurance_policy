import pandas as pd
import altair as alt
import numpy as np
import datetime as dt
from pyodide.http import open_url

alt.data_transformers.disable_max_rows()

policy = pd.read_csv(open_url(
    "https://raw.githubusercontent.com/felicyairenea/insurance_policy/main/data/policy.csv"))
client = pd.read_csv(open_url(
    "https://raw.githubusercontent.com/felicyairenea/insurance_policy/main/data/client.csv"))

# 1. policy dataset data preparation
policy_modified = policy.copy()

# a. change data type
policy_modified = policy_modified.astype(
    {'RIDER_COUNT': 'int8', 'APPLICATION_DATE': 'datetime64[ns]', 'ISSUE_DATE': 'datetime64[ns]'})

# b. mapped the payment term
frequency_mapping = {
    'Annually': 1,
    'Semi-Annually': 2,
    'Quarterly': 4,
    'Monthly': 12
}
policy_modified['TERM_MAPPED'] = policy_modified['PAYMENT_TERM'].map(
    frequency_mapping).astype(int)

# c. create annualized premium
policy_modified['ANNUALIZED_PREMIUM'] = policy_modified['TERM_MAPPED'] * \
    policy_modified['PREMIUM']

# d. get the policy processing period in days
policy_modified['POLICY_PROCESSING_PERIOD'] = (
    policy_modified['ISSUE_DATE']-policy_modified['APPLICATION_DATE']).dt.days

# e. create policy purchase sequence
policy_modified = policy_modified.assign(
    PURCHASE_SEQ=policy_modified.groupby(['CLIENT_ID']).cumcount() + 1)

# f. create number of policy ownership
policy_modified = policy_modified.assign(POLICY_OWNERSHIP=policy_modified.groupby([
                                         'CLIENT_ID'])['POLICY_ID'].transform('count'))

# g. create the year policy issued
policy_modified['ISSUED_YEAR'] = policy_modified['ISSUE_DATE'].dt.year

# h. remove policy with premium = 0
policy_modified = policy_modified.loc[policy_modified['PREMIUM'] > 0].reset_index(
    drop=True)

# i. remove policy with outlier annualized premium (the only Billion Peso premium)
policy_modified = policy_modified.loc[policy_modified['ANNUALIZED_PREMIUM']
                                      != policy_modified.ANNUALIZED_PREMIUM.max()].reset_index(drop=True)

# 2. client dataset data preparation
client_modified = client.copy()

# a. remove client with empty total coverage
client_modified = client_modified[client_modified.TOTAL_COVERAGE.notna()].reset_index(drop=True)

# b. create age of client
# take current date as the last record of issue date
current_date = pd.to_datetime(
    policy_modified.ISSUE_DATE.max().strftime('%Y-%m-%d'))
# get age by subtract current date with client's date of birth
client_modified['AGE'] = (
    current_date - client_modified['DOB'].astype('datetime64[ns]')).astype('<m8[Y]').astype('int8')

#------------------------------------------------------------------------------------------------------------------
# Data Visualization

year_base = alt.Chart(policy_modified, title='Total Annualized Premium and Policies Sold By Year').encode(
    alt.X('ISSUED_YEAR:N', title='Policy Issued Year'),
    tooltip=[alt.Tooltip('average(POLICY_PROCESSING_PERIOD):Q', title='Average Policy Processing Period (days)'),
             alt.Tooltip('average(ANNUALIZED_PREMIUM):Q', title='Average Annualized Premium')])

year_bar = year_base.mark_bar(color='#800020').encode(
    alt.Y('count(POLICY_ID):Q',          
          axis=alt.Axis(title='Policies Sold',
                        titleColor='#800020')))
    
year_line = year_base.mark_line(interpolate='monotone', color='#d0cbc6').encode(
    alt.Y('sum(ANNUALIZED_PREMIUM):Q', axis=alt.Axis(title='Sum of Annual Premium',
                                                     titleColor='#8a7e72')))

year = alt.layer(year_bar, year_line + year_line.mark_circle(size=150, color='#d0cbc6')).resolve_scale(
    y='independent'
).properties(
    width=300,
    height=400
)

pcode_base = alt.Chart(policy_modified, title='Total Policies and Average Annualized Premium By Product').encode(
    alt.X('PRODUCT_CODE:N', title='Products'),
    tooltip=[alt.Tooltip('count(PRODUCT_ID):Q', title='Total Products Sold'),
             alt.Tooltip('average(POLICY_PROCESSING_PERIOD):Q', title='Average Policy Processing Period (days)'), ],
)

pcode_bar = pcode_base.mark_bar(opacity=0.85).encode(
    alt.Y('count(POLICY_ID):Q', axis=alt.Axis(title='Total Product Sold', titleColor='#f2903a')),
    color=alt.Color('POLICY_STATUS:N', scale=alt.Scale(
        scheme='goldred'), legend=alt.Legend(title='Policy Status')),
)

pcode_line = pcode_base.mark_line(stroke='#800020', interpolate='monotone').encode(
    alt.Y('average(ANNUALIZED_PREMIUM)', axis=alt.Axis(title='Average Annualized Premium', titleColor='#800020'))
)

pcode_point = pcode_line.mark_circle(color='#800020').encode(
    alt.Size('average(POLICY_PROCESSING_PERIOD):Q', legend=alt.Legend(title='Average Policy Proc. Period')))

pcode = alt.layer(pcode_bar, pcode_line + pcode_point).resolve_scale(
    y='independent'
).properties(
    width=600,
    height=200
)

rider_premium_grid = alt.Chart(policy_modified, title='Annual Premium to Total Rider').mark_rect(color='#8a7e72').encode(
    alt.X('ANNUALIZED_PREMIUM:Q', title='Annual Premium',
          bin=alt.Bin(maxbins=20), axis=alt.Axis(grid=False)),
    alt.Y('RIDER_COUNT:Q', bin=alt.Bin(maxbins=8),
          axis=alt.Axis(title='Total Rider', grid=False)),
    tooltip=[alt.Tooltip('count(PRODUCT_ID):Q', title='Total Products Sold'),
             alt.Tooltip('average(ANNUALIZED_PREMIUM):Q', title='Average Annualized Premium')],
    color=alt.Color('count(POLICY_ID)', scale=alt.Scale(
        scheme='warmgreys'), legend=alt.Legend(title='Total Policies Sold')),
).properties(
    width=600,
    height=200
)

cgender_pie = alt.Chart(client_modified, title="Client's Gender Distribution").mark_arc(innerRadius=75).encode(
    theta=alt.Theta('count(GENDER):Q', stack=True, scale=alt.Scale(
        type="linear", rangeMax=1.5708, rangeMin=-1.5708)),
    tooltip = [alt.Tooltip('count(CLIENT_ID)', title='Count of Records')],
    color=alt.Color('GENDER:N', scale=alt.Scale(
        scheme='warmgreys'), legend=None))

cgender = cgender_pie + cgender_pie.mark_text(radius=170, fontSize=16).encode(text='GENDER') 

cmarital_status = alt.Chart(client_modified, title='Client\'s Marital Status').mark_arc(innerRadius=50).encode(
    theta=alt.Theta('count(MARITAL_STATUS):Q'),
    color=alt.Color('MARITAL_STATUS:N', title='Marital Status',
                    scale=alt.Scale(scheme='goldred'), legend=None),
    tooltip=[alt.Tooltip('MARITAL_STATUS:N', title='Marital Status'),
             alt.Tooltip('count(MARITAL_STATUS):Q', title='Total Client'),
             alt.Tooltip('average(ANNUAL_INCOME):Q', title='Annual Income'),
             alt.Tooltip('average(TOTAL_COVERAGE):Q', title='Total Coverage')])

cnat_base = alt.Chart(client_modified, title='Average Annual Income and Total Coverage By Client\'s Nationality').encode(
    alt.X('NATIONALITY:N', title='Nationality'),
    tooltip=[alt.Tooltip('count(NATIONALITY):Q', title='Total Client')],
)

cnat_bar = cnat_base.mark_bar(color='#800020').encode(
    alt.Y('average(ANNUAL_INCOME):Q').title('Annual Income', titleColor='#800020'),
)

cnat_line = cnat_base.mark_line(stroke='#d0cbc6', interpolate='monotone').encode(
    alt.Y('average(TOTAL_COVERAGE)').title(
        'Average Total Coverage', titleColor='#d0cbc6')
)

cnat_point = cnat_line.mark_circle(color='#d0cbc6')

cnat = alt.layer(cnat_bar, cnat_line + cnat_point).resolve_scale(
    y='independent'
).properties(
    width=400,
    height=200
)

cage_base = alt.Chart(client_modified, title='Average Annual Income and Total Coverage By Client\'s Age').encode(
    alt.X('AGE:Q', title='Age'),
    tooltip=[alt.Tooltip('count(CLIENT_ID):Q', title='Total Client')],
)

cage_bar = cage_base.mark_bar(color='#800020').encode(
    alt.Y('average(ANNUAL_INCOME):Q').title(
        'Annual Income', titleColor='#800020'),
)

cage_line = cage_base.mark_line(stroke='#d0cbc6', interpolate='monotone').encode(
    alt.Y('average(TOTAL_COVERAGE)').title(
        'Average Total Coverage', titleColor='#d0cbc6')
)

cage_point = cage_line.mark_circle(color='#d0cbc6')

cage = alt.layer(cage_bar, cage_line + cage_point).resolve_scale(
    y='independent'
).properties(
    width=400,
    height=200
)

client_cleaned = client_modified[~((client_modified['ANNUAL_INCOME'] > 400000000) | (
    client_modified['TOTAL_COVERAGE'] > 600000000))].reset_index(drop=True)

cover_income_scatter = alt.Chart(client_cleaned).mark_circle(color='maroon').encode(
    alt.X('ANNUAL_INCOME:Q', title='Annual Income',
            axis=alt.Axis(grid=False)),
    alt.Y('TOTAL_COVERAGE:Q', title='Total Coverage',
            axis=alt.Axis(grid=False)),
    tooltip=[alt.Tooltip('ANNUAL_INCOME:Q', title='Annual Income'),
             alt.Tooltip('TOTAL_COVERAGE:Q', title='Total Coverage'),
             alt.Tooltip('GENDER:N', title='Gender'),
             alt.Tooltip('NATIONALITY:N', title='Nationality'),
             alt.Tooltip('MARITAL_STATUS:N', title='Marital Status'),
             alt.Tooltip('SMOKER:N', title='Smoker')]
).properties(
    width = 400,
    height = 400
).interactive()

alt.vconcat(alt.hconcat(year, alt.vconcat(pcode, rider_premium_grid)
            ),alt.hconcat(alt.vconcat(cgender, cmarital_status), alt.vconcat(cnat, cage), cover_income_scatter)).configure(background='#f9f9dc')

# note - bikin customer retention rate and policy lapse rate kalo niat
