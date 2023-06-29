import pandas as pd
import numpy as np
import datetime as dt
import json
from pyodide.http import open_url

policy = pd.read_csv(open_url(
    "https://raw.githubusercontent.com/felicyairenea/insurance_policy/main/data/policy.csv"))
agent = pd.read_csv(open_url(
    "https://raw.githubusercontent.com/felicyairenea/insurance_policy/main/data/agent.csv"))
client = pd.read_csv(open_url(
    "https://raw.githubusercontent.com/felicyairenea/insurance_policy/main/data/client.csv"))


# 1. policy dataset data preparation
policy_modified = policy.copy()

## a. change data type
policy_modified = policy_modified.astype({'RIDER_COUNT': 'int8', 'APPLICATION_DATE': 'datetime64[ns]','ISSUE_DATE':'datetime64[ns]'})

## b. mapped the payment term
frequency_mapping = {
    'Annually': 1,
    'Semi-Annually': 2,
    'Quarterly': 4,
    'Monthly': 12
}
policy_modified['TERM_MAPPED'] = policy_modified['PAYMENT_TERM'].map(
    frequency_mapping).astype(int)

## c. create annualized premium
policy_modified['ANNUALIZED_PREMIUM'] = policy_modified['TERM_MAPPED'] * \
    policy_modified['PREMIUM']

## d. get the policy processing period in days
policy_modified['POLICY_PROCESSING_PERIOD'] = (policy_modified['ISSUE_DATE']-policy_modified['APPLICATION_DATE']).dt.days

## e. create policy purchase sequence
policy_modified = policy_modified.assign(
    PURCHASE_SEQ=policy_modified.groupby(['CLIENT_ID']).cumcount() + 1)

## f. create number of policy ownership
policy_modified = policy_modified.assign(POLICY_OWNERSHIP=policy_modified.groupby(['CLIENT_ID'])['POLICY_ID'].transform('count'))

## g. remove policy with premium = 0
policy_modified = policy_modified.loc[policy_modified['PREMIUM'] > 0].reset_index(
    drop=True)

## h. remove policy with outlier annualized premium (the only Billion Peso premium)
policy_modified = policy_modified.loc[policy_modified['ANNUALIZED_PREMIUM']!= policy_modified.ANNUALIZED_PREMIUM.max()].reset_index(drop=True)

#------------------------------------------------------------------------------------------------------------------
# for key stats
# a. Policies Sold
policy_sold = policy.shape[0]

# b. Assured Amount
# get client id with policy status of inforce from policy
active_client = list(policy[policy.POLICY_STATUS ==
                     'Inforce'].CLIENT_ID.unique())
# took client with active policy only
client_mod = client[client.CLIENT_ID.isin(
    active_client)].reset_index(drop=True)
# took client that have non-null data on its total coverage
client_mod = client_mod[client_mod.TOTAL_COVERAGE.notna()
                        ].reset_index(drop=True)
# get the sum of total coverage
assured_amount = client_mod.TOTAL_COVERAGE.sum()

# c. Average Annual 
# get policy with policy status of inforce
policy_mod = policy_modified[policy_modified.POLICY_STATUS =='Inforce'].reset_index(drop=True)
# get the mean of annualized premium
avg_ann_premium = policy_mod.ANNUALIZED_PREMIUM.mean()

# d. Total Premium
total_premium = policy_mod.ANNUALIZED_PREMIUM.sum()

toprows = [
    policy_sold,
    str(round(assured_amount/1000000000, 3))+'B',
    str(round(avg_ann_premium/1000, 3))+'K',
    str(round(total_premium/1000000000, 3))+'B',
    [
        policy_modified.ISSUE_DATE.min().strftime('%Y-%m-%d'),
        policy_modified.ISSUE_DATE.max().strftime('%Y-%m-%d'),
    ]
]
# ------------------------------------------------------------------------------------------------------------------

# 2. create best agent dataframe
## a. choose needed columns to use from policy_modified and agent dataset
col_pol = ['POLICY_ID', 'POLICY_STATUS', 'AGENT_ID',
           'ANNUALIZED_PREMIUM', 'PURCHASE_SEQ']
col_age = ['AGENT_ID', 'FIRST_YEAR_COMMISSION']

## b. select active agent only
agent = agent[agent.AGENT_STATUS == 'Active']

## c. left merge policy_modified and agent with chosen columns
joined = pd.merge(
    left=policy_modified[col_pol], right=agent[col_age], how='left', on='AGENT_ID')

## d. create column to see whether the customer is a repeat customer
joined['RECURRING_PURCHASE'] = np.where(joined.PURCHASE_SEQ == 2, 1, 0)

## e. transform policy_status to 1 indicates an active policy, or else is 0
joined['POLICY_STATUS'] = np.where(joined['POLICY_STATUS'] == 'Inforce', 1, 0)

## f. create best_agent
best_agent = joined.groupby(['AGENT_ID']).agg(Total_Policy_Sold=('POLICY_ID', 'count'),
                                              Total_Premium=('ANNUALIZED_PREMIUM', 'sum'),
                                              Total_Active_Policy=('POLICY_STATUS', 'count'),
                                              Total_Repeat_Client=('RECURRING_PURCHASE', 'sum')).sort_values(by=['Total_Premium',
                                                                                                                 'Total_Policy_Sold'], ascending=False).reset_index().head()

## g. rename columns
best_agent.rename(columns={'AGENT_ID': 'Agent ID', 'Total_Policy_Sold': 'Total Policies Sold', 'Total_Premium': 'Total Annualized Premiums',
                  'Total_Active_Policy': 'Total Inforce Policies', 'Total_Repeat_Client': 'Total Repeat Clients'}, inplace=True)

print(best_agent.set_index('Agent ID').reset_index().to_html(index=False, col_space=[150, 100, 100, 100, 100], table_id='bestagents', justify='justify',
      classes=['table table-bold table-striped table-bordered']))

print(json.dumps(toprows))
