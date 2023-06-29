import pandas as pd

policy = pd.read_csv("./data/policy.csv")

# policy dataset data preparation
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

# g. remove policy with premium = 0
policy_modified = policy_modified.loc[policy_modified['PREMIUM'] > 0].reset_index(
    drop=True)

# h. remove policy with outlier annualized premium (the only Billion Peso premium)
policy_modified = policy_modified.loc[policy_modified['ANNUALIZED_PREMIUM']
                                      != policy_modified.ANNUALIZED_PREMIUM.max()].reset_index(drop=True)

