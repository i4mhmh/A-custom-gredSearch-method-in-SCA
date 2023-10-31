#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
* File       : custom_callback.py
* Created    : 2023-10-30 11:43:32
* Author     : M0nk3y
* Version    : 1.1
'''

from sklearn.model_selection import ParameterGrid, train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers


def show_result(rank_times, optmizier, learning_rate, epoch, batch_size, loss):
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.title("optmizer:" + str(optmizier) + '_' + str(learning_rate) + '_epoch:' + str(epoch) + '_batch' + str(batch_size) + '_loss: '+ str(loss))
    x = [x for x in range(0, 2000)]
    ax.plot(x, np.mean(rank_times, axis=0), 'b')
    ax.set(xlabel="number of traces", ylabel="mean rank")
    plt.show()


def mean_rank_score(model, test_data, pred_size, key):
    attack_traces, targets = test_data[0], test_data[1]
    predictions = model.predict(attack_traces, verbose=0)
    predictions = np.log(predictions + 1e-40)
    #! 可能需要修改
    rank_times = np.zeros(pred_size)
    pred = np.zeros(256)
    idx = np.random.randint(predictions.shape[0], size=pred_size)    
    for i, p in enumerate(idx):
        for k in range(pred.shape[0]):
            pred[k] += predictions[p, targets[p, k]]
        ranked = np.argsort(pred)[::-1]
        rank_times[i] = list(ranked).index(key)
    return rank_times

# 自定网格调参callbacks 暂时只修改model的param.
def monkey_search(data, params, model, callbacks, key, shuffle=True, pred_size=2000, verbose=1):
    assert type(callbacks) == list
    train_data, test_data = data['train_data'], data['test_data']
    idx = None
    for i, callback in enumerate(callbacks):
        if callback.__class__.__name__ == 'OneCycleLR':
            idx = i

    candidatas = ParameterGrid(param_grid=params)
    for index, candidata in enumerate(candidatas):
        x_train, x_val, y_train, y_val = train_test_split(train_data[0], train_data[1], test_size=0.2, shuffle=True) if(shuffle == True) else train_test_split(train_data[0], train_data[1], test_size=0.2)
        optimizer = candidata['optimizer']
        learning_rate = candidata['learning_rate']
        epoch = candidata['epoch']
        batch_size = candidata['batch_size']
        loss = candidata['loss']
        print(learning_rate, type(learning_rate))
        # 如果有onecycle要给对应赋值
        if idx != None:
            callbacks[idx].num_samples = x_train.shape[0]
            callbacks[idx].batch_size = batch_size
            callbacks[idx].max_lr = learning_rate
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        model.optimizer.learning_rate.assign(learning_rate)
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks, epochs=epoch, batch_size=batch_size)

        ranks = np.zeros((10, pred_size))
        for i in range(ranks.shape[0]):
            ranks[i] = mean_rank_score(model, test_data, pred_size=pred_size, key=key)
        if verbose == 1:
            show_result(ranks, optimizer, learning_rate, epoch, batch_size, loss)

