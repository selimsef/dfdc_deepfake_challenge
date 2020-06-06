import json

import numpy as np
from sklearn.metrics import log_loss
preds= {"probs" :{}, "targets": {}}
for path in ["5folds/predictions_{}.json".format(i) for i in range(0, 4)]:
    with open(path, "r") as f:
         p = json.load(f)
         preds["probs"].update(p["probs"])
         preds["targets"].update(p["targets"])
probs = preds["probs"]
targets = preds["targets"]
print(len(probs))


def confident_strategy(pred, t=0.87):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > 0.7 * sz:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.6 * sz:
        return np.mean(pred[pred < 0.2])

    else:
        return np.mean(pred)

strategies = [np.mean]

for strategy in strategies:
   
    data_x = []
    data_y = []
    data_vids = []
    losses = []
    for vid, score in probs.items():
        score = np.array(score)
        lbl = targets[vid]
    
        score = strategy(score)
        lbl = np.mean(lbl)
        data_x.append(score)
        data_y.append(lbl)
        data_vids.append(vid)
    y = np.array(data_y)
    x = np.array(data_x)
    fake_idx = y > 0.1
    real_idx = y < 0.1
    fake_loss = log_loss(y[fake_idx], x[fake_idx], labels=[0, 1])
    real_loss = log_loss(y[real_idx], x[real_idx], labels=[0, 1])
    print("fake loss {} real loss {}".format(fake_loss, real_loss))
    for i in range(len(data_x)):
        loss = log_loss(y[i:i+1], np.clip(x[i:i+1], 0.01, .99), labels=[0, 1])
        losses.append(loss)
    losses = np.array(losses)
    data_vids = np.array(data_vids)
    data = []
    for i in reversed(np.argsort(losses)):
        data.append([data_vids[i], losses[i], data_x[i], data_y[i]])
    print("Strategy: {} loss {} ".format(strategy, np.mean(np.sort(losses)[:])))
    import pandas as pd
    pd.DataFrame(data, columns=["id", "loss", "p", "target"]).to_csv("losses.csv", index=False)
