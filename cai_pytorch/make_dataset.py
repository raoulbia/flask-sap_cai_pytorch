#!/usr/bin/env python3

from pprint import pprint
import requests
import pandas as pd
from pandas.io.json import json_normalize


def squad():

    # dev url
    url = "https://raw.githubusercontent.com/aswalin/SQuAD/master/data/dev-v1.1.json"

    # train url
    # url = "https://raw.githubusercontent.com/aswalin/SQuAD/master/data/train-v1.1.json"

    r = requests.get(url)
    json_dict = r.json()

    corpus = 'Economic_inequality'
    df = convert_squad_to_tidy_df(json_dict, corpus)  # .reset_index()
    # print(len(df))
    # pprint(df.context.iloc[0])
    # print(df.tail())
    pairs = list(zip(df['question'].tolist(), df['text'].tolist()))
    # print(pairs)
    return pairs


def convert_squad_to_tidy_df(json_dict, corpus):
    """This function converts the SQuAD JSON data to a Tidy Data Pandas Dataframe.

    :param obj json_dict: squad json data
    :param str corpus: name of squad corpora to select subset from json object

    :returns: converted json data
    :rtype: pandas dataframe

    """
    data = [c for c in json_dict['data'] if c['title'] == corpus][0]
    df = pd.DataFrame()
    data_paragraphs = data['paragraphs']
    for article_dict in data_paragraphs:
        row = []
        for answers_dict in article_dict['qas']:
            for answer in answers_dict['answers']:
                row.append((article_dict['context'],  # [:50],
                            answers_dict['question'],
                            answers_dict['id'],
                            answer['answer_start'],
                            answer['text']
                            ))
        df = pd.concat(
            [df, pd.DataFrame.from_records(row, columns=['context', 'question', 'id', 'answer_start', 'text'])], axis=0,
            ignore_index=True)
        df.drop_duplicates(inplace=True)
    return df

if __name__ == "__main__":
    squad()

