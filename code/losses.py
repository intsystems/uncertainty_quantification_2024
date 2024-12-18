import torch
from torch.nn.functional import cross_entropy
import numpy as np

def get_ohe_ce_risk(y_logit, y_ids):
    '''
        returns ohe risk for a Cross-Entropy loss
    '''
    # y_ids: [batch_size]: Long tensor with ids of right classes
    # y_logit: [batch_size, K]: Float tensor with K classes (LOGITS)
    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]

    chosen_probs = probs[np.arange(len(y_ids)), y_ids] # [batch_size]
    return -torch.log(chosen_probs).mean()

def get_ohe_ce_risk(y_logit, y_ids):
    '''
        returns ohe risk for a Cross-Entropy loss
    '''
    # y_ids: [batch_size]: Long tensor with ids of right classes
    # y_logit: [batch_size, K]: Float tensor with K classes (LOGITS)
    # probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]
    norm_constant = torch.logsumexp(y_logit, dim=1)

    chosen_logits = y_logit[np.arange(len(y_ids)), y_ids] # [batch_size]
    return -(chosen_logits - norm_constant).mean()

def test_ohe_ce_risk():
    # 5 классов, batch_size = 4
    batch_size = 4
    K = 5
    y_ids = torch.LongTensor([1, 3, 2, 4])
    y_logit = torch.randn(size=(4, 5))

    ce_risk = get_ohe_ce_risk(y_logit, y_ids)
    right_ce_risk = cross_entropy(y_logit, y_ids)
    assert torch.allclose(ce_risk, right_ce_risk), f"CE risk: {ce_risk.item():.4f}, right_risk: {right_ce_risk.item():.4f}"
test_ohe_ce_risk()

def get_uniform_approx_ce_addition(y_logit, y_ids, eps=1e-4):
    '''
        according to the formula above
    '''

    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]
    g_grad = 1 + torch.log(probs) # [batch_size, K]

    K = g_grad.shape[1]
    s_vector = torch.zeros_like(probs) - eps / K
    s_vector[np.arange(len(probs)), y_ids] = eps * (1 - 1 / K) # [batch_size, K]

    return (s_vector * g_grad).sum(axis=1).mean()

def test_get_uniform_approx_ce_addition():
    # 5 классов, batch_size = 4
    batch_size = 4
    K = 5
    y_ids = torch.LongTensor([1, 3, 2, 4])
    y_logit = torch.randn(size=(4, 5))

    ce_add = get_uniform_approx_ce_addition(y_logit, y_ids, eps=1e-2)
    print(f"CE_add: {ce_add:.4f}")

test_get_uniform_approx_ce_addition()


def get_stopgrad_ce_addition(y_logit, y_ids, eps=1e-4):
    '''
        according to the formula above
    '''

    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]
    g_grad = 1 + torch.log(probs) # [batch_size, K]

    s_vector = -probs.detach()
    s_vector[np.arange(len(probs)), y_ids] = 1 - s_vector[np.arange(len(probs)), y_ids]
    s_vector *= eps
    # K = g_grad.shape[1]
    # s_vector = torch.zeros_like(probs) - eps / K
    # s_vector[np.arange(len(probs)), y_ids] = eps * (1 - 1 / K) # [batch_size, K]

    return (s_vector * g_grad).sum(axis=1).mean()

def test_get_stopgrad_ce_addition():
    # 5 классов, batch_size = 4
    batch_size = 4
    K = 5
    y_ids = torch.LongTensor([1, 3, 2, 4])
    y_logit = torch.randn(size=(4, 5))

    ce_add = get_stopgrad_ce_addition(y_logit, y_ids, eps=1e-2)
    print(f"CE_add: {ce_add:.4f}")

test_get_stopgrad_ce_addition()



import torch
import numpy as np
from sklearn.metrics import brier_score_loss

def brier_right_risk(y_logit, y_ids, K):
    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]
    y_ohe = torch.nn.functional.one_hot(y_ids, num_classes=K) # [batch_size, K]
    return ((probs - y_ohe) ** 2).sum(dim=1).mean()

def get_ohe_brier_risk(y_logit, y_ids):
    '''
        returns ohe risk for a Cross-Entropy loss
    '''
    # y_ids: [batch_size]: Long tensor with ids of right classes
    # y_logit: [batch_size, K]: Float tensor with K classes (LOGITS)
    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]

    G = -(probs * (1 - probs)).sum(dim=1) # [batch_size]
    G_grad = 2 * probs - 1 # [batch_size, K]

    ohe_risk = (G_grad * probs).sum(dim=1) - G_grad[np.arange(len(y_ids)), y_ids] - G

    return ohe_risk.mean()

def test_ohe_brier_risk():
    # 5 классов, batch_size = 4
    batch_size = 4
    K = 5
    y_ids = torch.LongTensor([1, 3, 2, 4])
    y_logit = torch.randn(size=(4, 5))

    brier_risk = get_ohe_brier_risk(y_logit, y_ids)
    print(f'Brier risk: {brier_risk:.4f}')
    right_risk = brier_right_risk(y_logit, y_ids, K)
    # print(right_risk)
    assert torch.allclose(brier_risk, right_risk), f"Brier risk: {brier_risk.item():.4f}, right_risk: {right_risk.item():.4f}"

test_ohe_brier_risk()



def get_prior_approx_brier_addition(y_logit, y_ids, freqs, eps=1e-4):
    '''
        according to the formula above
    '''

    # freqs: [K] torch.Float with N_k/N


    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]
    g_grad = 2 * probs - 1 # [batch_size, K]

    freqs = freqs.to(probs.device)
    K = g_grad.shape[1]
    s_vector = torch.zeros_like(probs) - eps * freqs[None, :]
    s_vector[np.arange(len(probs)), y_ids] = eps * (1 - freqs[y_ids]) # [batch_size, K]

    return (s_vector * g_grad).sum(axis=1).mean()

def test_get_prior_approx_brier_addition():
    # 5 классов, batch_size = 4
    batch_size = 4
    K = 5
    y_ids = torch.LongTensor([1, 3, 2, 4])
    y_logit = torch.randn(size=(4, 5))

    freqs = torch.FloatTensor([3, 4, 5, 2, 8])
    freqs /= freqs.sum()

    add = get_prior_approx_brier_addition(y_logit, y_ids, freqs, eps=1e-2)
    print(f"prior_add: {add:.4f}")

test_get_prior_approx_brier_addition()


def get_stopgrad_brier_addition(y_logit, y_ids, freqs, eps=1e-4):
    '''
        according to the formula above
    '''

    # freqs: [K] torch.Float with N_k/N


    probs = torch.nn.functional.softmax(y_logit, dim=1) # [batch_size, K]
    g_grad = 2 * probs - 1 # [batch_size, K]

    s_vector = -probs.detach()
    s_vector[np.arange(len(probs)), y_ids] = 1 - s_vector[np.arange(len(probs)), y_ids]
    s_vector *= eps

    return (s_vector * g_grad).sum(axis=1).mean()

def test_get_stopgrad_brier_addition():
    # 5 классов, batch_size = 4
    batch_size = 4
    K = 5
    y_ids = torch.LongTensor([1, 3, 2, 4])
    y_logit = torch.randn(size=(4, 5))

    freqs = torch.FloatTensor([3, 4, 5, 2, 8])
    freqs /= freqs.sum()

    add = get_stopgrad_brier_addition(y_logit, y_ids, freqs, eps=1e-2)
    print(f"prior_add: {add:.4f}")

test_get_stopgrad_brier_addition()



