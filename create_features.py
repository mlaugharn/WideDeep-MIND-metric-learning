# creates X_wide, X_tab, X_wide_te, X_tab_te and saves them to csv files
import pathlib
config = {
    'train_folder': pathlib.Path('./smalldataset/train/'),
    'test_folder': pathlib.Path('./smalldataset/val/')
}

from tqdm.autonotebook import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy

import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

# silly thing required to use dict arguments w/ LRU cache
def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """
    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

logging.debug('loading impressions df...')
impressions_df = pd.read_csv(config['train_folder'] / 'behaviors.tsv', sep='\t', header=None)
logging.debug('loaded impressions df')
logging.debug('loading impressions_df for test..')
impressions_df_te = pd.read_csv(config['test_folder'] / 'behaviors.tsv', sep='\t', header=None)
logging.debug('loaded impressions df for test')

news_df = pd.read_csv(config['train_folder'] / 'news.tsv', sep='\t', header=None)
logging.debug('loaded news df')
news_df_te = pd.read_csv(config['test_folder'] / 'news.tsv', sep='\t', header=None)
logging.debug('loaded news df for test')

impressions_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

impressions_df.columns = impressions_cols
impressions_df_te.columns = impressions_cols
news_df.columns = news_cols
news_df_te.columns = news_cols

# replace NaN title entities with []
# replace NaN abstract entities with []
# replace NaN abstracts with ''
from ast import literal_eval
def clean_news(news_df):
    for row in news_df.loc[news_df.title_entities.isnull(), 'title_entities'].index:
        news_df.at[row, 'title_entities'] = '[]'
    for row in news_df.loc[news_df.abstract_entities.isnull(), 'abstract_entities'].index:
        news_df.at[row, 'abstract_entities'] = '[]'
    news_df.fillna({'abstract': ''
                    }, inplace=True)
    news_df['title_entities'] = news_df['title_entities'].map(literal_eval)
    news_df['abstract_entities'] = news_df['abstract_entities'].map(literal_eval)
    return news_df

news_df = clean_news(news_df)
logging.debug('cleaned news')
news_df_te = clean_news(news_df_te)
logging.debug('cleaned news for test')

def clean_impressions(impressions_df):
    # split history into a list
    # split impressions into a list
    for row in tqdm(impressions_df.loc[impressions_df.history.isnull(), 'history'].index):
        impressions_df.at[row, 'history'] = ''
    impressions_df['history'] = impressions_df['history'].str.split()
    impressions_df['impressions'] = impressions_df['impressions'].str.split()
    return impressions_df
impressions_df = clean_impressions(impressions_df)
logging.debug('cleaned impressions')
impressions_df_te = clean_impressions(impressions_df_te)
logging.debug('cleaned impressions for test')

news_id_idxer = dict(zip(news_df['news_id'], news_df.index)) #{newsid:idx for idx, newsid in news_df['news_id'].to_dict().items()}
logging.debug('indexed news ids')
news_id_idxer_te = dict(zip(news_df_te['news_id'], news_df_te.index)) # {newsid:idx for idx, newsid in news_df_te['news_id'].to_dict().items()}
logging.debug('indexed news ids for test')

import functools

@functools.lru_cache(maxsize=5000000)
def get_news_features(news_id, kind = 'train'):
    # category
    global news_df
    global news_df_te
    if news_id in news_id_idxer: 
        df = news_df
        idxer = news_id_idxer
    elif news_id in news_id_idxer_te:
        df = news_df_te
        idxer = news_id_idxer_te
    else:
        return {'category': 'None', 'subcategory': 'None', 'title': 'None', 'abstract': 'None'}
    # idxer = news_id_idxer if kind == 'train' else news_id_idxer_te
    # df = news_df if kind == 'train' else news_df_te
    idx = idxer[news_id]
    news = df.iloc[idx]
    category = news.category
    subcategory = news.subcategory
    title = news.title
    abstract = news.abstract
    
    # # first 4 title entities and first 8 abstract entities should normally be more than enough
    # title_entity1 = news.title_entities[0]['Label'] if len(news.title_entities) > 0 else None
    # title_entity2 = news.title_entities[1]['Label'] if len(news.title_entities) > 1 else None
    # title_entity3 = news.title_entities[2]['Label'] if len(news.title_entities) > 2 else None
    # title_entity4 = news.title_entities[3]['Label'] if len(news.title_entities) > 3 else None
    
    # abstract_entity1 = news.abstract_entities[0]['Label'] if len(news.abstract_entities) > 0 else None
    # abstract_entity2 = news.abstract_entities[1]['Label'] if len(news.abstract_entities) > 1 else None
    # abstract_entity3 = news.abstract_entities[2]['Label'] if len(news.abstract_entities) > 2 else None
    # abstract_entity4 = news.abstract_entities[3]['Label'] if len(news.abstract_entities) > 3 else None
    # abstract_entity5 = news.abstract_entities[4]['Label'] if len(news.abstract_entities) > 4 else None
    # abstract_entity6 = news.abstract_entities[5]['Label'] if len(news.abstract_entities) > 5 else None
    # abstract_entity7 = news.abstract_entities[6]['Label'] if len(news.abstract_entities) > 6 else None
    # abstract_entity8 = news.abstract_entities[7]['Label'] if len(news.abstract_entities) > 7 else None
    
    return {'category': category,
            'subcategory': subcategory,
            'title': title,
            'abstract': abstract,
            # 'title_entity1': title_entity1,
            # 'title_entity2': title_entity2,
            # 'title_entity3': title_entity3,
            # 'title_entity4': title_entity4,
            # 'abstract_entity1': abstract_entity1,
            # 'abstract_entity2': abstract_entity2,
            # 'abstract_entity3': abstract_entity3,
            # 'abstract_entity4': abstract_entity4,
            # 'abstract_entity5': abstract_entity5,
            # 'abstract_entity6': abstract_entity6,
            # 'abstract_entity7': abstract_entity7,
            # 'abstract_entity8': abstract_entity8
            }

            
#impression_id_idxer = {impression['impression_id']:i for i, impression in impressions_df.iterrows()}
#impression_id_idxer_te = {impression['impression_id']:i for i, impression in impressions_df_te.iterrows()}

logging.debug('processing clicks..')
user_clicks = {} # list of news articles that a user has clicked
for i, impression in tqdm(impressions_df.iterrows(), total=len(impressions_df)):
    user = impression.user_id
    if user not in user_clicks: user_clicks[user] = set()
    user_clicks[user].update(impression.history)
    # for h in impression.history:
    #     user_clicks[user].add(h)
#             print(impression.history)
    for x in impression.impressions:
        if x[-1] == '1':
            n, click = x[:-2], x[-1]
            user_clicks[user].add(n)

print('processing test clicks..')
user_clicks_te = user_clicks
for i, impression in tqdm(impressions_df_te.iterrows(), total=len(impressions_df_te)):
    user = impression.user_id
    if user not in user_clicks_te: user_clicks_te[user] = set()
    user_clicks_te[user].update(impression.history)
#     # else: 
#         # for h in impression.history:
#         #     user_clicks_te[user].add(h)
# #             print(impression.history)
    for x in impression.impressions:
        if x[-1] == '1':
            n, click = x[:-2], x[-1]
            user_clicks_te[user].add(n)
    
@functools.lru_cache(maxsize=500000)
def get_user_feats(user, kind = 'train'):
    clicks = user_clicks if (kind == 'train' or kind == 'val') else user_clicks_te
    if user not in clicks: 
        return {'cat0': 'None', 'cat1': 'None', 'cat2': 'None', 'cat_counts0': 0, 'cat_counts1': 0, 'cat_counts2': 0, 'subcat0': 'None', 'subcat1': 'None', 'subcat2': 'None', 'subcat_counts0': 0, 'subcat_counts1': 0, 'subcat_counts2': 0,
        'uid': user, 'clicks': 0}
    articles_feats = [get_news_features(article, kind) for article in clicks[user]]
    articles_feats_df = pd.DataFrame(articles_feats)
    cat_counts = articles_feats_df['category'].value_counts()
    cats = cat_counts.index.tolist()
    spaces_to_add = max(3 - len(cats), 0)
    cats = cats + [''] * spaces_to_add
    cat_counts = cat_counts.values.tolist() + [0] * spaces_to_add
    subcat_counts = articles_feats_df['subcategory'].value_counts()
    spaces_to_add = max(3 - len(subcat_counts), 0)
    subcats = subcat_counts.index.tolist() + [''] * spaces_to_add 
    subcat_counts = subcat_counts.values.tolist() + [0] * spaces_to_add
    click_counts = len(articles_feats_df)
    return_val = {'uid': user, 'clicks': click_counts}
    for i in range(3):
        cats[i] = cats[i]# if len(cats) > i else ''
        cat_counts[i] = cat_counts[i]# if len(cat_counts) > i else 0
        subcats[i] = subcats[i]# if len(subcats) > i else ''
        subcat_counts[i] = subcat_counts[i]# if len(subcat_counts) > i else 0
    return_val['cat0'] = cats[0]
    return_val['cat1'] = cats[1]
    return_val['cat2'] = cats[2]
    return_val['cat_counts0'] = cat_counts[0]
    return_val['cat_counts1'] = cat_counts[1]
    return_val['cat_counts2'] = cat_counts[2]
    return_val['subcat0'] = subcats[0]
    return_val['subcat1'] = subcats[1]
    return_val['subcat2'] = subcats[2]
    return_val['subcat_counts0'] = subcat_counts[0]
    return_val['subcat_counts1'] = subcat_counts[1]
    return_val['subcat_counts2'] = subcat_counts[2]
    return return_val


import functools

#
# user features + news article features + impression features | clicked?
def create_data_df(impressions_df, news_df, kind = 'train'):
    examples = []
    user_newsid_histories = {}
    for i, impression in tqdm(impressions_df.iterrows(), total=len(impressions_df)):
        user_id = impression.user_id
        impression_id = impression.impression_id
        user_features = get_user_feats(user_id, kind)
        #impression_feats = get_impression_feats(impression_id)
        impression_feats = {'time': impression['time'], 'impression_id': impression_id, 'numchoices': len(impression['impressions'])}
        if user_id not in user_newsid_histories: user_newsid_histories[user_id] = set()
        # for hist_news in impression.history:
        #     news_id = hist_news
        #     if news_id not in user_newsid_histories[user_id]:
        #         user_newsid_histories[user_id].add(news_id) # make sure to only add a history news item once
        #         news_feats = get_news_features(news_id, kind)
        #         example = {**user_features, **news_feats, **impression_feats, 'clicked': True, 'history': True}
        #         examples.append(example)
        if kind == 'train' or kind == 'val':
            for n, impression_news in enumerate(impression.impressions):
                news_id, clicked = impression_news[:-2], bool(int(impression_news[-1]))
                news_feats = get_news_features(news_id, kind)
                example = {**user_features, **news_feats, **impression_feats, 'clicked': clicked, 'history': False}
                examples.append(example)
        elif kind == 'test':
            for n, impression_news in enumerate(impression.impressions):
                news_id, clicked = impression_news[:-2], bool(int(impression_news[-1]))
                news_feats = get_news_features(news_id, kind)
                example = {**user_features, **news_feats, **impression_feats, 'clicked': False, 'history': False}
                examples.append(example)
    return pd.DataFrame(examples)

print('creating train df...')
df_train = create_data_df(impressions_df, news_df, 'train')
print('created train df')

print('creating test df...')
df_test = create_data_df(impressions_df_te, news_df_te, 'val')
print('created test df')

# df_train, df_test = train_test_split(train_df, test_size=0.2, stratify=train_df.clicked)
print(df_train.head())
print(df_test.head())

print('created df_train, df_test')

wide_cols = [
    'cat0', 'cat1', 'cat2',
    #  'cat_counts0', 'cat_counts1', 
    'history',
    # 'cat_counts2',
     'subcat0', 'subcat1', 'subcat2', 
    # 'subcat_counts0',
    # 'subcat_counts1', 'subcat_counts2', 
    'category', 'subcategory',
    'numchoices'
]
cross_cols = [('cat0', 'cat1'), ('cat0', 'subcat0')]
embed_cols = [
    ('cat0', 16),
    ('cat1', 16),
    ('cat2', 16),
    ('subcat0', 16),
    ('subcat1', 16),
    ('subcat2', 16)
]
cont_cols = ['clicks', 'cat_counts0', 'cat_counts1', 'cat_counts2', 'subcat_counts0', 'subcat_counts1', 'subcat_counts2', 'numchoices']

target_col = 'clicked'
target = df_train['clicked'].values

wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
print('fitting x_wide...')
X_wide = wide_preprocessor.fit_transform(df_train)
print('made X_wide')
wide = Wide(wide_dim = np.unique(X_wide).shape[0], pred_dim=128)
tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
print('making x_tab..')
X_tab = tab_preprocessor.fit_transform(df_train)
print('made X_tab')

print('making x_wide_te...')
X_wide_te = wide_preprocessor.transform(df_test)
print('made X_wide_te')
print('making x_tab_te..')
X_tab_te = tab_preprocessor.transform(df_test)
print('made X_tab_te')



import pickle
with open('X_wide.pkl', 'wb') as f: pickle.dump(X_wide, f)
with open('X_tab.pkl', 'wb') as f: pickle.dump(X_tab, f)
with open('X_wide_te.pkl', 'wb') as f: pickle.dump(X_wide_te, f)
with open('X_tab_te.pkl', 'wb') as f: pickle.dump(X_tab_te, f)
with open('wide_preprocessor.pkl', 'wb') as f: pickle.dump(wide_preprocessor, f)
with open('wide.pkl', 'wb') as f: pickle.dump(wide, f)
with open('tab_preprocessor.pkl', 'wb') as f: pickle.dump(tab_preprocessor, f)
df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')
# import IPython
# IPython.embed()