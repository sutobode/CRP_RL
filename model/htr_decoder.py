import torch
import torch.nn as nn
import math
from model.encoder import Encoder, MultiHeadAttention
from model.target_selector import TargetSelector
from model.sampler import TopKSampler, CategoricalSampler
from env.env import Env


class HTRDecoder(nn.Module):
    """
    Hierarchical Target-Then-Relocate decoder.
    
    High-level: TargetSelector selects which stack to clear next.
    Low-level: Attention-based decoder selects where to relocate blockers.
    
    Khác với Decoder gốc:
    - Không dùng env.find_target_stack() (rule-based min priority)
    - Dùng TargetSelector (learned policy) để chọn target
    - TargetSelector nhìn vào working time cost + priority
    """

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tanh_c = args.tanh_c
        self.samplers = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}
        self.sampler = None

        self.encoder = Encoder(args).to(self.device)
        self.target_selector = TargetSelector(args.embed_dim).to(self.device)

        self.W_target = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_global = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_K1 = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_K2 = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_Q = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_V = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.MHA = MultiHeadAttention(args.n_heads, args.embed_dim, is_encoder=False)

        self.online = args.online
        if self.online:
            self.online_known_num = args.online_known_num
            init_mask_token = float(self.online_known_num + 1)
            self.mask_token = nn.Parameter(torch.tensor(init_mask_token, device=self.device))

    def set_sampler(self, decode_type):
        self.sampler = self.samplers[decode_type]

    def select_target(self, node_embeddings, graph_embedding, mask):
        """High-level: chọn target stack dùng learned policy.
        
        Returns:
            target_logp: log probability của target được chọn
            target_embeddings: embedding của target stack
        """
        logits = self.target_selector(node_embeddings, graph_embedding, mask)
        log_p = torch.log_softmax(logits, dim=1)
        actions = self.sampler(log_p)
        logp = torch.gather(log_p, dim=1, index=actions).squeeze(-1)
        target_emb = node_embeddings[torch.arange(node_embeddings.size(0)), actions.squeeze(-1), :]
        return actions, logp, target_emb

    def select_destination(self, target_emb, graph_emb, node_emb, node_keys, node_vals, mask):
        """Low-level: chọn destination stack dùng attention (giống Decoder gốc)."""
        context = (self.W_target(target_emb) + self.W_global(graph_emb)).unsqueeze(1)
        query_ = self.W_Q(self.MHA([context, node_keys, node_vals]))
        logits = torch.matmul(query_, node_keys.permute(0, 2, 1)).squeeze(1) / math.sqrt(query_.size(-1))
        logits = self.tanh_c * torch.tanh(logits)
        logits = logits - mask.squeeze(-1) * 1e9
        log_p = torch.log_softmax(logits, dim=1)
        return log_p

    def forward(self, x, max_retrievals):
        batch, n_bays, n_rows, max_tiers = x.size()
        max_stacks = n_bays * n_rows

        cost = torch.zeros(batch).to(self.device)
        ll = torch.zeros(batch).to(self.device)

        env = Env(self.device, x, max_retrievals)
        cost = cost + env.clear()

        if not self.online:
            encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
        else:
            x_new = env.x.clone()
            mask = x_new > self.online_known_num
            x_new[mask] = self.mask_token
            encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

        node_embeddings, graph_embedding = encoder_output

        # action mask for target selection: non-empty stacks
        target_mask = (env.x.amax(dim=-1) > 0).bool()

        # HTR: High-level target selection
        target_actions, target_logp, target_embeddings = self.select_target(
            node_embeddings, graph_embedding, target_mask)
        env.set_target_stack(target_actions)

        # mask for destination selection (existing logic)
        dest_mask = env.create_mask()

        for i in range(max_stacks * max_tiers * max_tiers):
            assert i < max_stacks * max_tiers * max_tiers - 1

            # prepare keys/values for destination attention
            node_keys = self.W_K1(node_embeddings)
            node_values = self.W_V(node_embeddings)

            # Low-level: select destination
            log_p = self.select_destination(
                target_embeddings, graph_embedding,
                node_embeddings, node_keys, node_values, dest_mask)

            actions = self.sampler(log_p)

            tmp_log_p = log_p.clone()
            tmp_log_p[(env.empty | env.early_stopped), :] = 0
            ll = ll + torch.gather(input=tmp_log_p, dim=1, index=actions).squeeze(-1).to(self.device)

            cost = cost + env.step(dest_index=actions)

            if env.all_terminated():
                break

            if not self.online:
                encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
            else:
                x_new = env.x.clone()
                mask = x_new > self.online_known_num
                x_new[mask] = self.mask_token
                encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

            node_embeddings, graph_embedding = encoder_output

            # HTR: re-select target after each step
            target_mask = (env.x.amax(dim=-1) > 0).bool()
            target_actions_h, target_logp_h, target_embeddings = self.select_target(
                node_embeddings, graph_embedding, target_mask)
            target_logp = target_logp + target_logp_h  # accumulate for training
            env.set_target_stack(target_actions_h)

            dest_mask = env.create_mask()

        return cost, ll, target_logp  # return target_logp for auxiliary loss
