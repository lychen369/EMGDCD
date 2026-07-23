import itertools
import sys
from os.path import join as opj
from os.path import dirname as opd
import numpy as np
from model.SCE import Spat_Cau_Encoder
from utils.eval_auroc import cosine_similarity, compute_matrix_auroc, shd_distance
from utils.loss_tool import decouple_loss

sys.path.append(opj(opd(__file__), ".."))
import tqdm
import torch
from torch import  nn
from utils.utils_tool import gumbel_softmax, spars_loss, matching_loss
from utils.opt_type import MSCDopt
from model.TCD import Tem_Cau_Decoder
from torchmetrics.classification import BinaryAveragePrecision

METRIC_OUTPUT_NAMES = (
    "full_auroc",
    "common_auroc",
    "specific_auroc",
    "full_auprc",
    "common_auprc",
    "specific_auprc",
    "full_shd",
    "common_shd",
    "specific_shd",
    "full_cs",
    "common_cs",
    "specific_cs",
)


def _binarize_graph(graph):
    graph = np.abs(np.asarray(graph))
    graph[graph != 0] = 1
    return graph.astype(int)


def build_specific_ground_truth(true_causal, true_common):
    true_causal = _binarize_graph(true_causal)
    true_common = _binarize_graph(true_common)
    true_specific = np.clip(true_causal - true_common, 0, 1)
    return true_specific.astype(int)


def _mask_top_by_target(scores, target):
    edge_count = int(np.asarray(target).sum())
    target_shape = np.asarray(target).shape
    flat_mask = np.zeros(np.prod(target_shape, dtype=int), dtype=int)
    if edge_count <= 0:
        return flat_mask.reshape(target_shape)
    if edge_count >= flat_mask.size:
        flat_mask[:] = 1
        return flat_mask.reshape(target_shape)
    flat_scores = np.asarray(scores).reshape(-1)
    if flat_scores.size != flat_mask.size:
        raise ValueError(
            f"Score shape {np.asarray(scores).shape} does not match target shape {target_shape}"
        )
    if not np.isfinite(flat_scores).all():
        raise ValueError("Graph scores contain NaN or infinite values")
    top_idx = np.argsort(-flat_scores, kind="stable")[:edge_count]
    flat_mask[top_idx] = 1
    return flat_mask.reshape(target_shape)


def _graph_metrics(scores, target, n_nodes):
    target = _binarize_graph(target)
    auroc = compute_matrix_auroc(torch.tensor(scores), torch.tensor(target))
    metric = BinaryAveragePrecision()
    auprc = metric(torch.tensor(scores), torch.tensor(target))
    binary_scores = _mask_top_by_target(scores, target)
    shd = shd_distance(binary_scores, target)
    cs = cosine_similarity(torch.tensor(binary_scores), torch.tensor(target), n_nodes)
    return auroc, auprc, shd, cs

def _remove_common_from_full_graph(full_graph, common_graph, true_common):
    specific_graph = np.asarray(full_graph).copy()
    common_mask = _mask_top_by_target(common_graph, true_common).astype(bool)
    specific_graph[common_mask] = 0
    return specific_graph


def _mean_last(values, n=5):
    tail = values[-n:]
    return sum(tail) / len(tail)


def _max_with_smooth_gradient(common, specific, tau=0.2):
    """Use an exact max in the forward pass and a smooth max for gradients."""
    hard = torch.maximum(common, specific)
    smooth = tau * torch.logsumexp(torch.stack((common, specific), dim=0) / tau, dim=0)
    return hard.detach() + (smooth - smooth.detach())


class ST_MSCD_NET(object):
    def __init__(self, args: MSCDopt.MSCDargs, device="cuda"):
        self.args = args
        self.device = device
        self.lambda_1 = self.args.encoder.lambda_1
        self.max_grad_norm = getattr(self.args, "max_grad_norm", 5.0)
        self.max_backward_tau = getattr(self.args.encoder, "max_backward_tau", 0.2)
        self.decoder = nn.ModuleList([Tem_Cau_Decoder(self.args.n_nodes, in_ch=self.args.data_dim,n_layers=self.args.decoder.gru_layers,hidden_ch=self.args.decoder.mlp_hid,shared_weights_decoder=self.args.decoder.shared_weights_decoder,
                                           concat_h=self.args.decoder.concat_h).to(self.device) for _ in range(args.n_sub)])
        self.encoder = Spat_Cau_Encoder(n_sub = self.args.n_sub, n_nodes=self.args.n_nodes, in_dim=args.input_step, hid_dim=args.encoder.hid_dim).to(self.device)
        self.pred_loss = nn.MSELoss()
        self.spar_loss = spars_loss()
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=self.args.decoder.lr_dec,
            weight_decay=self.args.decoder.weight_decay
        )
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=10, gamma=0.9)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.encoder.lr_enc, weight_decay=1e-4)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=2, gamma=0.9)


    def decoder_stage(self, train_batch_data,i):
        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
        self.decoder.train()
        self.decoder_optimizer.zero_grad()
        loss = 0
        causal_s, causal_c, causal_c_all = self.encoder(train_batch_data, i)
        causal_s, causal_c = torch.sigmoid(causal_s), torch.sigmoid(causal_c)
        for m,_ in enumerate(self.decoder):
            batch_data_j = train_batch_data[m][i]
            x, y, t = batch_data_j
            common_expanded = causal_c
            causal_m = torch.max(common_expanded, causal_s[:, m, :, :])
            causal_m = torch.mean(causal_m, dim=0)
            causal_sampled = sample_bernoulli(causal_m, self.args.batch_size)
            y_pred = self.decoder[m](x,causal_sampled)
            loss_j = self.pred_loss(y, y_pred)
            loss += loss_j
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        self.decoder_optimizer.step()
        return loss

    def encoder_stage(self, train_batch_data,i):
        def gumbel_sigmoid_sample(graph, batch_size, tau=1.01):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples
        self.encoder.train()
        self.encoder_optimizer.zero_grad()
        loss,loss_pre,loss_spar = 0.0, 0.0,0.0
        causal_s, causal_c, causal_c_all = self.encoder(train_batch_data,i)
        causal_s, causal_c, causal_c_all = torch.sigmoid(causal_s), torch.sigmoid(causal_c), torch.sigmoid(causal_c_all)
        for m in range(self.args.n_sub):
            batch_data_m = train_batch_data[m][i]
            x, y, t = batch_data_m
            causal_m = _max_with_smooth_gradient(causal_c, causal_s[:, m, :, :], self.max_backward_tau)
            causal_m = torch.mean(causal_m,dim=0)
            causal_sampled = gumbel_sigmoid_sample(causal_m, self.args.batch_size)
            y_pred = self.decoder[m](x, causal_sampled)
            loss_spar_m = self.spar_loss(causal_m)
            loss_pre_m = self.pred_loss(y, y_pred)
            loss_spar += loss_spar_m
            loss_pre += loss_pre_m
        loss = loss_pre + self.lambda_1 * loss_spar
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        self.encoder_optimizer.step()
        return loss, loss_pre, loss_spar

    def train(self, train_batch_data, test_batch_data, true_causal, true_common, true_specific=None):
        full_auroc = 0
        common_auroc = 0
        specific_auroc = 0
        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.args.total_epoch,dynamic_ncols=True)
        true_causal = _binarize_graph(true_causal)
        true_common = _binarize_graph(true_common)
        if true_specific is None:
            true_specific = build_specific_ground_truth(true_causal, true_common)
        else:
            true_specific = _binarize_graph(true_specific)
        metric_history = {name: [] for name in METRIC_OUTPUT_NAMES}
        for epoch_i in range(self.args.total_epoch):
            # train decoder
            for i,_ in enumerate(train_batch_data[0]):
                latent_pred_step += self.args.batch_size
                loss = self.decoder_stage(train_batch_data,i)
                pbar.set_postfix_str(f"d loss={loss.item():.2f}, full={full_auroc:.2f}, c={common_auroc:.2f}")
            self.decoder_scheduler.step()

           # train encoder
            for i, _ in enumerate(train_batch_data[0]):
                graph_discov_step += self.args.batch_size
                loss, loss_pre, loss_spar = self.encoder_stage(train_batch_data,i)
                pbar.set_postfix_str(f"e pl={loss_pre.item():.2f},sl={loss_spar.item():.2f},full={full_auroc:.2f}, c={common_auroc:.2f},s={specific_auroc:.2f}")
            self.encoder_scheduler.step()

            pbar.update(1)
            # Estimate one global graph by averaging scores over all test batches.
            common_graph_sum = np.zeros((self.args.n_nodes, self.args.n_nodes), dtype=float)
            full_graph_sum = np.zeros(
                (self.args.n_sub, self.args.n_nodes, self.args.n_nodes), dtype=float
            )
            test_batch_count = 0
            self.encoder.eval()
            with torch.no_grad():
                for i, _ in enumerate(test_batch_data[0]):
                    causal_s, causal_c, causal_c_all = self.encoder(test_batch_data, i)
                    causal_s, causal_c, causal_c_all = torch.sigmoid(causal_s), torch.sigmoid(causal_c), torch.sigmoid(causal_c_all)
                    causal_c_graph = torch.mean(causal_c, dim=0).detach().cpu().numpy()
                    common_graph_sum += causal_c_graph
                    for m in range(self.args.n_sub):
                        causal_full_m = torch.mean(torch.max(causal_c, causal_s[:, m, :, :]), dim=0).detach().cpu().numpy()
                        full_graph_sum[m] += causal_full_m
                    test_batch_count += 1

            common_graph = common_graph_sum / test_batch_count
            full_graphs = full_graph_sum / test_batch_count
            full_metric_sum = np.zeros(4, dtype=float)
            specific_metric_sum = np.zeros(4, dtype=float)
            for m in range(self.args.n_sub):
                specific_graph = _remove_common_from_full_graph(full_graphs[m], common_graph, true_common)
                # specific_graph = remove_common_from_full(full_graphs[m], common_graph)
                full_metric_sum += np.array(_graph_metrics(full_graphs[m], true_causal[m], self.args.n_nodes))
                specific_metric_sum += np.array(_graph_metrics(specific_graph, true_specific[m], self.args.n_nodes))

            full_auroc, full_auprc, full_shd, full_cs = full_metric_sum / self.args.n_sub
            specific_auroc, specific_auprc, specific_shd, specific_cs = (
                specific_metric_sum / self.args.n_sub
            )
            common_auroc, common_auprc, common_shd, common_cs = _graph_metrics(
                common_graph, true_common, self.args.n_nodes
            )
            epoch_metrics = {
                "full_auroc": full_auroc, "full_auprc": full_auprc,
                "full_shd": full_shd, "full_cs": full_cs,
                "specific_auroc": specific_auroc, "specific_auprc": specific_auprc,
                "specific_shd": specific_shd, "specific_cs": specific_cs,
                "common_auroc": common_auroc, "common_auprc": common_auprc,
                "common_shd": common_shd, "common_cs": common_cs,
            }
            for name in METRIC_OUTPUT_NAMES:
                metric_history[name].append(float(epoch_metrics[name]))
        return tuple(_mean_last(metric_history[name]) for name in METRIC_OUTPUT_NAMES)



def main(train_batch_data,test_batch_data,true_causal,true_common, opt, device="cpu", true_specific=None):
    ST_MSCD = ST_MSCD_NET(opt,device=device)
    metrics = ST_MSCD.train(
        train_batch_data, test_batch_data, true_causal, true_common, true_specific=true_specific
    )
    return tuple(round(metric, 4) for metric in metrics)
