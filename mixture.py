import numpy as np
import torch
import torch.nn.functional as F


def log_sum_exp(x: torch.Tensor):
    """ numerically stable log_sum_exp implementation that prevents overflow 
    @param x (torch.Tensor): a tensor with shape (batch_size, time_length, channels)
    """
    m, _ = torch.max(x, -1, keepdim=False)
    m2, _ = torch.max(x, -1, keepdims=True)
    return m + torch.log(torch.sum(torch.exp(x-m2), -1))


def log_prob_from_logits(x: torch.Tensor):
    """ numerically stable log_softmax implementation that prevents overflow 
    @param x (torch.Tensor): a tensor with shape (batch_size, time_length, channels)
    """
    m, _ = torch.max(x, -1, keepdims=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), -1, keepdims=True))


def discretized_mix_logistic_loss(y_hat: torch.Tensor, y: torch.Tensor, num_classes: int, device: torch.device,
                                  log_scale_min=-7.0, reduce=True):
    '''Discretized mix of logistic distributions loss.

    Note that it is assumed that input is scaled to [-1, 1]

    @param y_hat (torch.Tensor): Tensor [batch_size, channels, time_length], predicted output.
    @param y (torch.Tensor): Tensor [batch_size, time_length, 1], Target.
    Returns:
            Tensor loss
    '''

    nr_mix = y_hat.size(1) // 3

    #[Batch_size, time_length, channels]
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. [batch_size, time_length, num_mixtures] x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.max(
        y_hat[:, :, 2 * nr_mix: 3 * nr_mix], torch.Tensor([log_scale_min]).to(device))

    # [batch_size, time_length, 1] -> [batch_size, time_length, num_mixtures]
    y = y * torch.ones(1, 1, nr_mix, dtype=torch.float32, device=device)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = - F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in this code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    log_probs = torch.where(y < -0.999, log_cdf_plus,
                            torch.where(y > 0.999, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5,
                                                    torch.log(torch.max(
                                                        cdf_delta, torch.Tensor([1e-12]).to(device))),
                                                    log_pdf_mid - np.log((num_classes - 1) / 2))))

    #log_probs = log_probs + tf.nn.log_softmax(logit_probs, -1)
    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def sample_from_discretized_mix_logistic(y: torch.Tensor, log_scale_min=-7.):
    '''
    Args:
            y: Tensor, [batch_size, channels, time_length]
    Returns:
            Tensor: sample in range of [-1, 1]
    '''
    nr_mix = y.size(1) // 3

    #[batch_size, time_length, channels]
    y = y.transpose(2, 1)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size()).uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))
    argmax = torch.argmax(temp, -1)

    # [batch_size, time_length] -> [batch_size, time_length, nr_mix]
    one_hot = F.one_hot(argmax, num_classes=nr_mix).float()
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, -1)
    log_scales = torch.max(torch.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, -1), log_scale_min)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8-bit value when sampling
    u = torch.FloatTensor(means.size()).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1 - u))

    return torch.min(torch.max(x, -1.), 1.)
