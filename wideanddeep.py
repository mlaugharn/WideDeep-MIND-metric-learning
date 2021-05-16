import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from numpy.lib.function_base import _i0_1


def generate_predictions_file(df, clicks_true, clicks_pred):
    # df -> list of lists: impression . indexes of rows in that impression
    rows = []
    for impression, impression_group in df.groupby('impression_id'):
        idxs = impression_group.index
        order = np.arange(len(idxs))
        preds = clicks_pred[idxs]
        order = order[np.argsort(preds)][::-1]
        order = ','.join([str(id) for id in order.tolist()])
        order = '[' + order + ']'
        row = f'{impression} {order}\n'
        rows.append(row)
    with open('res/prediction.txt', 'w') as f: f.writelines(rows)
    rows = []
    for impression, impression_group in df.groupby('impression_id'):
        idxs = impression_group.index
        order = np.arange(1, len(idxs)+1)
        preds = clicks_true[idxs]
        order = order[np.argsort(preds)][::-1]
        order = ','.join([str(id) for id in order.tolist()])
        order = '[' + order + ']'
        row = f'{impression} {order}\n'
        rows.append(row)
    with open('ref/truth.txt', 'w') as f: f.writelines(rows)



if __name__ == '__main__':
    __spec__ = None # pdb hack
    from tqdm.autonotebook import tqdm
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import sklearn.metrics
    import pickle
    import torch.nn as nn
    import torch 
    from pytorch_widedeep import Trainer
    from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
    from pytorch_widedeep.models import Wide, TabMlp, WideDeep
    from pytorch_widedeep.metrics import Accuracy
    from pytorch_metric_learning import losses, distances, miners, reducers, samplers


    import pandas as pd

    print('loading data..')
    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')
    print('loaded dataframes')

    # wide_cols = [
    #     'cat0', 'cat1', 'cat2', 'cat_counts0', 'cat_counts1', 'history',
    #     'cat_counts2', 'subcat0', 'subcat1', 'subcat2', 'subcat_counts0',
    #     'subcat_counts1', 'subcat_counts2', 'category', 'subcategory', 'title_entity1', 'title_entity2', 'title_entity3',
    #     'title_entity4', 'abstract_entity1', 'abstract_entity2',
    #     'abstract_entity3', 'abstract_entity4', 'abstract_entity5',
    #     'abstract_entity6', 'abstract_entity7', 'abstract_entity8'
    # ]
    # cross_cols = [('cat0', 'cat1'), ('cat0', 'subcat0')]
    # embed_cols = [
    #     ('uid', 16),
    #     ('cat0', 16),
    #     ('cat1', 16),
    #     ('cat2', 16),
    #     ('title_entity1', 16),
    #     ('title_entity2', 16),
    #     ('title_entity3', 16),
    #     ('title_entity4', 16),
    #     ('abstract_entity1', 16),
    #     ('abstract_entity2', 16),
    #     ('abstract_entity3', 16),
    #     ('abstract_entity4', 16),
    # ]
    cont_cols = ['clicks', 'cat_counts0', 'cat_counts1', 'cat_counts2', 'subcat_counts0', 'subcat_counts1', 'subcat_counts2', 'numchoices']

    target_col = 'clicked'
    target = df_train['clicked'].values

    print('loading train features..')
    with open('wide_preprocessor.pkl', 'rb') as f: wide_preprocessor = pickle.load(f)
    with open('X_wide.pkl', 'rb') as f: X_wide = pickle.load(f)
    with open('wide.pkl', 'rb') as f: wide = pickle.load(f)
    with open('tab_preprocessor.pkl', 'rb') as f: tab_preprocessor = pickle.load(f)
    with open('X_tab.pkl', 'rb') as f: X_tab = pickle.load(f)
    print('train features loaded')

    deeptabular = TabMlp(
        mlp_hidden_dims=[128, 128, 128],
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=cont_cols,
    )

    # head = nn.Identity()
    # head.out_dim = 32
    pred_dim = 128
    model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=pred_dim)
    

    # breakpoint()
    # widedim = wide


    class MetricLoss(nn.Module):
        def __init__(self, loss_func, miner):
            super(MetricLoss, self).__init__()
            self.loss_func = loss_func
            self.miner = miner
        
        def forward(self, embeddings, target):
            target = target.reshape(-1).long()
            hard_pairs = self.miner(embeddings, target)
            return self.loss_func(embeddings, target, hard_pairs)

    from pytorch_widedeep.metrics import Metric
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    import faiss

    class EmbeddingAccuracy(Metric):
        def __init__(self, top_k: int = 1):
            super(EmbeddingAccuracy, self).__init__()

            self.acc_calc = AccuracyCalculator()
            # metric name needs to be defined
            self._name = "NMI"
            self.every_n_iter = 500
            self.last_accuracy = None
            self.i = -1

        def reset(self):
            pass
            # self.queries = np.array()
            # self.total_count = np.array()

        def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
            self.i += 1
            if self.i % self.every_n_iter == 0: 
                # num_classes = y_pred.size(1)

                # if num_classes == 1:
                #     y_pred = y_pred.round()
                #     y_true = y_true
                # elif num_classes > 1:
                #     y_pred = y_pred.topk(self.top_k, 1)[1]
                #     y_true = y_true.view(-1, 1).expand_as(y_pred)

                # self.correct_count += y_pred.eq(y_true).sum().item()
                # self.total_count += len(y_pred)
                # accuracy = float(self.correct_count) / float(self.total_count)
                labels = labels.reshape(-1).long()
                acc_dict = self.acc_calc.get_accuracy(embeddings.detach(), embeddings.detach(), labels.detach(), labels.detach(), embeddings_come_from_same_source=True)
                print(acc_dict)
                accuracy = acc_dict[self._name]
                self.last_accuracy = accuracy
                return accuracy
            else: return self.last_accuracy


    # sampler = samplers.MPerClassSampler(labels, m, batch_size=None, length_before_new_iter=100000)

    metric_loss = MetricLoss(losses.TripletMarginLoss(reducer=reducers.ClassWeightedReducer( torch.tensor([1, 22]) )), miners.MultiSimilarityMiner())
    # metric_loss = MetricLoss(losses.TripletMarginLoss())#, miners.MultiSimilarityMiner())

    print('made model')
    trainer = Trainer(model, objective="binary", metrics=[EmbeddingAccuracy], custom_loss_function=metric_loss)
    print('fitting...')
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=5,
        batch_size=78,
        val_split=None
    )


    print("loading test set..")
    with open('X_wide_te.pkl', 'rb') as f: X_wide_te = pickle.load(f)
    with open('X_tab_te.pkl', 'rb') as f: X_tab_te = pickle.load(f)

    from pytorch_metric_learning import testers


    print("Creating tester")
    tester = testers.GlobalEmbeddingSpaceTester(use_trunk_output=True, dataloader_num_workers=0, accuracy_calculator=AccuracyCalculator(k=3), batch_size=78)

    class HackyDict(dict):
        # allows dataset[i] -> {"X_wide": X_wide[i], "X_tab": X_tab[i]} for dataloader problem
        def __missing__(self, key):
            if isinstance(key, int):
                value = {"X_wide": self["X_wide"][key], "X_tab": self["X_tab"][key]}
                return value
    
    class DS(torch.utils.data.Dataset):
        def __init__(self, X_wide, X_tab, labels):
            self.X_wide = X_wide
            self.X_tab = X_tab
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {"wide": self.X_wide[idx], "deeptabular": self.X_tab[idx]}, self.labels[idx]

    # train_d = HackyDict()
    # val_d = HackyDict()
    # train_d["X_wide"] = X_wide
    # train_d["X_tab"] = X_tab
    # val_d["X_wide"] = X_wide_te
    # val_d["X_tab"] = X_tab_te
    train_target = target
    test_target = df_test[target_col].values
    # test_target = [None] * len(df_test)
    # dataset_dict = {"train": {, "X_tab": X_tab}, 
    #                 "val": {"X_wide": X_wide_te, "X_tab": X_tab_te}}
    # dataset_dict = {"train": train_d, "val": val_d}
    dataset_dict = {"train": DS(X_wide, X_tab, target), "val": DS(X_wide_te, X_tab_te, test_target)}
    print("made dataset_dict")
    print("computing accuracies...")
    all_accuracies = tester.test(dataset_dict, epoch = 0, trunk_model = model)
    print(f"computed  accuracies: {all_accuracies}")
    
    print("computing embeddings...")
    train_embeddings, train_labels = tester.get_all_embeddings(
        dataset_dict['train'],
        trunk_model = model
    )

    test_embeddings, test_labels = tester.get_all_embeddings(
        dataset_dict['val'],
        trunk_model = model, 
    )
    print("computed embeddings")

    from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder

    # matcher = MatchFinder()
    inference_model = InferenceModel(trunk = model.cpu(), embedder = None)
    print("adding train set to index...")
    # inference_model.train_indexer(dataset_dict['train'])
    print("done")
    print("finding nearest neighbors....")
    # from sklearn.neighbors import NearestNeighbors
    print("try the following lines first..")
    # breakpoint()
    # use annoy
    # neigh = NearestNeighbors(n_neighbors=5)
    # neigh.fit(train_embeddings.cpu().numpy(), train_labels.cpu().numpy())
    # distances, indices = neigh.kneighbors(test_embeddings.cpu().numpy(), return_distance=True)
    # breakpoint()
    # indices, distances = inference_model.get_nearest_neighbors(test_embeddings, k=1)
    import annoy

    index = annoy.AnnoyIndex(pred_dim, metric='angular')
    print("adding train set to index")
    for i, training_example in tqdm(enumerate(train_embeddings.cpu().numpy()), total=len(train_embeddings)):
        index.add_item(i, training_example)
    print("building nearest neighbor trees...")
    index.build(10)
    print("built.")
    indices, distances = [], []
    print("finding nearest neighbors for test set..")
    for i, test_example in tqdm(enumerate(test_embeddings.cpu().numpy()), total=len(test_embeddings)):
        nearest, distance = index.get_nns_by_vector(test_example, 1, include_distances=True)
        indices.append(nearest)
        distances.append(distance)
    print("found.")
    indices, distances = np.array(indices), np.array(distances)
    preds = df_train.clicked.values[indices]
    # impressions_results = {imp_id: [news_id, pos/neg, distance]}
    impression_ranks = {}
    print("ranking impression articles...")
    for impression_id, group in tqdm(df_test.groupby('impression_id')):
        pos = []
        neg = []
        for imp_position, (ix, impression) in enumerate(group.iterrows()):
            idx = impression.iloc[0]
            pred = preds[idx]
            dist = distances[idx]
            if pred: pos.append((imp_position, dist))
            else: neg.append((imp_position, dist))
        final_rank = [imp_position for imp_position, dist in sorted(pos, key=lambda xy: xy[1])] + [imp_position for imp_position, dist in sorted(neg, key=lambda xy: xy[1])]
        impression_ranks[impression_id] = final_rank
    
    with open("res/prediction.txt", "w") as f:
        for impression_id, rankings in impression_ranks.items():
            rankings = ",".join([str(rank + 1) for rank in rankings])
            line = f"{impression_id} [{rankings}]\n"
            f.writelines(line)
    import IPython
    IPython.embed()
