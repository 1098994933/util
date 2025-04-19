import os
import unittest

import featuretools as ft

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


#
# data = ft.demo.load_mock_customer()
# customers_df = data["customers"]
# print(customers_df.head(50))
#
# sessions_df = data["sessions"]
# print(sessions_df)
#
# transactions_df = data["transactions"]
#
# entities = {
#     "customers": (customers_df, "customer_id"),
#     "sessions": (sessions_df, "session_id", "session_start"),
#     "transactions": (transactions_df, "transaction_id", "transaction_time")}
#
# relationships = [("sessions", "session_id", "transactions", "session_id"),
#                  ("customers", "customer_id", "sessions", "customer_id")]
#
# # 根据主表，计算附表的特征
# feature_matrix_customers, features_dfs = ft.dfs(dataframes=entities,
#                                                 relationships=relationships,
#                                                 target_dataframe_name="customers",
#                                                 agg_primitives=["mean", "median", "std"],
#                                                 max_depth=2,
#                                                 )
# feature_matrix_customers.to_csv("feature_matrix_customers.csv", index=False)
# print(features_dfs)


class FeatureToolsTests(unittest.TestCase):
    def setUp(self):
        pass

    def testFeatureToolsTests(self):
        import featuretools as ft

        data = ft.demo.load_mock_customer()
        transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])

        print(transactions_df)

    def test_EntitySet(self):
        data = ft.demo.load_mock_customer()
        transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])
        es = ft.EntitySet(id="database")
        es.add_dataframe(
            dataframe=transactions_df,
            dataframe_name="transactions",
            index="transaction_id",
            time_index="transaction_time",
        )
        customers_df = data["customers"]
        es.add_dataframe(dataframe=customers_df, dataframe_name="customers", index="customer_id")

        session_df = data["sessions"]
        es.add_dataframe(dataframe=session_df, dataframe_name="sessions", index="session_id")

        es = es.add_relationship("customers", "customer_id",
                                 "sessions", "customer_id")
        print(es)

    def test_Build_EntitySet(self):
        data = ft.demo.load_mock_customer()
        transactions_df = data["transactions"]
        sessions_df = data["sessions"]
        customers_df = data["customers"]
        dataframes = {
            "customers": (customers_df, "customer_id"),
            "sessions": (sessions_df, "session_id", "session_start"),
            "transactions": (transactions_df, "transaction_id", "transaction_time")}
        relationships = [("sessions", "session_id", "transactions", "session_id"),
                         ("customers", "customer_id", "sessions", "customer_id")]
        es = ft.EntitySet(id="database", dataframes=dataframes, relationships=relationships)
        print(es)

        feature_matrix, features = ft.dfs(entityset=es,
                                          target_dataframe_name="customers",
                                          agg_primitives=["mean", "median", "std"],
                                          max_depth=2)
        feature_matrix.to_csv("feature_matrix_customers.csv", index=False)
