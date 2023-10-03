import torch

import torch.nn.functional as F


def mixture_bernoulli_loss(
    label,
    theta_logits,
    log_alpha,
    adj_loss_func,
    subgraph_idx,
    reduction="mean",
):
    """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      theta_logits: E X D, see comments above
      log_alpha: E X D, see comments above
      adj_loss_func: BCE loss
      subgraph_idx: E X 1, see comments above
      subgraph_idx_base: B+1, cumulative # of edges in the subgraphs associated with each batch
      num_canonical_order: int, number of node orderings considered
      sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp
        i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
        This is equivalent to the original GRAN loss.
      return_neg_log_prob: boolean, if True also return neg log prob
      reduction: string, type of reduction on batch dimension ("mean", "sum", "none")

    Returns:
      loss (and potentially neg log prob)
    """

    num_subgraph = subgraph_idx.max() + 1
    E = theta_logits.shape[0]
    K = theta_logits.shape[1]
    assert E % C == 0

    adj_loss = torch.stack(
        [adj_loss_func(theta_logits[:, kk], label) for kk in range(K)], dim=1
    )  # E, K, adj loss for each edge

    const = torch.zeros(num_subgraph).to(label.device).unsqueeze(1)  # S

    const = const.scatter_add_(
        0, subgraph_idx, torch.ones_like(subgraph_idx).float()
    )  # nb edges per subgraph

    reduce_adj_loss = torch.zeros((num_subgraph, K)).to(label.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add_(
        0, subgraph_idx.expand(-1, K), adj_loss
    )  # S, K, sum of adj losses for each subgraph

    reduce_log_alpha = torch.zeros((num_subgraph, K)).to(label.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add_(
        0, subgraph_idx.expand(-1, K), log_alpha
    )  # S, K, sum of log alpha for each subgraph

    reduce_log_alpha = reduce_log_alpha / const.view(
        -1, 1
    )  # S, K, average log alpha for each subgraph
    reduce_log_alpha = F.log_softmax(
        reduce_log_alpha, -1
    )  # S, K, log softmax of average log alpha for each subgraph

    log_prob = -reduce_adj_loss + reduce_log_alpha  # S, K, log prob of each subgraph
    log_prob = torch.logsumexp(log_prob, dim=1)  # S, log prob of each subgraph

    if reduction == "mean":
        loss = log_prob.mean()
    elif reduction == "sum":
        loss = log_prob.sum()

    return loss


E = 1000
K = 10
C = 1
B = 10

label = torch.randint(0, 2, (E,)).float()
theta_logits = torch.randn(E, K).float()
log_alpha = torch.randn(E, K).float()
adj_loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")

subgraph_idx = torch.randint(0, B, (E, 1))
# sort by subgraph_idx
subgraph_idx, sort_idx = torch.sort(subgraph_idx, dim=0)

subgraph_idx_base = torch.arange(B + 1) * E // B

num_canonical_order = C

loss = mixture_bernoulli_loss(
    label,
    theta_logits,
    log_alpha,
    adj_loss_func,
    subgraph_idx,
    subgraph_idx_base,
    num_canonical_order,
    sum_order_log_prob=False,
    return_neg_log_prob=False,
    reduction="mean",
)
