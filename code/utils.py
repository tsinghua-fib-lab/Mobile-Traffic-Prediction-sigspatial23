import torch
import numpy as np
import pandas as pd

def evaluate_mdoel(model, loss, data_iter, g_base, etype_base, emd_mx, g_urban, etype_urban, scaler,
                                  device):
    model.eval( )
    count = 0
    l_sum, n = 0.0, 0
    with torch.no_grad( ):
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            # print(x.shape,g_base,etype_base.shape,emd_mx.shape,etype_urban.shape)
            y_pred = model(x, g_base, etype_base, emd_mx, g_urban, etype_urban)
            # print(y.shape,y_pred.shape)
            l = loss(y_pred, y)
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]

            y = y.squeeze(-1)
            y_pred = y_pred.squeeze(-1)
            y_pred = y_pred.unsqueeze(0)
            # print(y.shape)
            y = scaler.inverse_transform(y.detach( ).cpu( ).numpy( ))
            y_pred = scaler.inverse_transform(y_pred.detach( ).cpu( ).numpy( ))
            if count == 0:
                data = y
                data_pred = y_pred
                count += 1
            else:
                data = np.concatenate((data, y), axis = 0)
                data_pred = np.concatenate((data_pred, y_pred), axis = 0)

        data = data.T
        data_pred = data_pred.T
        d = np.abs(data - data_pred)
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE = np.array(mae).mean( )
        RMSE = np.sqrt(np.array(mse).mean( ))

        data = torch.tensor(data)
        data_pred = torch.tensor(data_pred)
        r2 = 1 - torch.sum((data - data_pred) ** 2) / torch.sum((data - torch.mean(data)) ** 2)

    return l_sum / n, MAE, RMSE, r2.item( )

