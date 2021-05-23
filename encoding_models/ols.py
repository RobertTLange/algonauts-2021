import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold


def fit_linear_model(model_config, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    n_scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ols = OLS_pytorch(model_config["lambda_reg"])
        ols.fit(X_train, y_train.T)
        preds = ols.predict(X_test)
        score = np.mean((y_test - preds)**2)
        n_scores.append(score)
    return np.mean(n_scores)


class OLS_pytorch(object):
    def __init__(self, lambda_reg=0.0, use_gpu=False):
        self.coefficients = []
        self.use_gpu = use_gpu
        self.X = None
        self.y = None
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = self._reshape_x(X)
        if len(y.shape) == 1:
            y = self._reshape_x(y)

        X =  self._concatenate_ones(X)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        if self.use_gpu:
            X = X.cuda()
            y = y.cuda()
        XtX = torch.matmul(X.t(),X)
        Xty = torch.matmul(X.t(),y.unsqueeze(2))
        XtX = XtX.unsqueeze(0)
        XtX = torch.repeat_interleave(XtX, y.shape[0], dim=0)
        # Add regularization
        ident_rep = torch.repeat_interleave(
                            torch.eye(n=XtX.shape[1]).unsqueeze(0),
                            y.shape[0], dim=0)
        XtX_lamb = self.lambda_reg * ident_rep

        betas_cholesky, _ = torch.solve(Xty, XtX + XtX_lamb)

        self.coefficients = betas_cholesky
        self.X = X
        self.y = y

    def predict(self, entry):
        if len(entry.shape) == 1:
            entry = self._reshape_x(entry)
        entry =  self._concatenate_ones(entry)
        entry = torch.from_numpy(entry).float()
        if self.use_gpu:
            entry = entry.cuda()
        prediction = torch.matmul(entry,self.coefficients)
        prediction = prediction.cpu().numpy()
        prediction = np.squeeze(prediction).T
        return prediction

    def score(self):
        prediction = torch.matmul(self.X,self.coefficients)
        yhat = prediction
        ybar = (torch.sum(self.y, dim=1, keepdim=True)/self.y.shape[1]).unsqueeze(2)
        ssreg = torch.sum((yhat-ybar)**2,dim=1, keepdim=True)
        print(yhat.shape, ybar.shape)
        print(ssreg/(yhat.shape[0]*yhat.shape[1]))
        sstot = torch.sum((self.y.unsqueeze(2) - ybar)**2,dim=1, keepdim=True)
        score = ssreg / sstot
        return score.cpu().numpy().ravel()

    def _reshape_x(self,X):
        return X.reshape(-1,1)

    def _concatenate_ones(self,X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        return np.concatenate((ones,X),1)