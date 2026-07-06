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
    
    Khác với Decoder gốc (chỉ dùng min-priority stack làm context):
    - Dùng TWO contexts cho destination selection:
      1. Current target (env.find_target_stack() — min priority)
      2. Long-term goal (TargetSelector — learned, cost-aware)
    
    Context = W_target(current_target_emb) + W_goal(longterm_goal_emb) + W_global(graph_emb)
    
    TargetSelector được train bằng REINFORCE (cùng objective với decoder).
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
        self.W_goal = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
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

    def select_goal(self, node_embeddings, graph_embedding, mask):
        """Select long-term goal stack via learned policy."""
        logits = self.target_selector(node_embeddings, graph_embedding, mask)
        log_p = torch.log_softmax(logits, dim=1)
        actions = self.sampler(log_p)
        logp = torch.gather(log_p, dim=1, index=actions).squeeze(-1)
        goal_emb = node_embeddings[torch.arange(node_embeddings.size(0)), actions.squeeze(-1), :]
        return actions, logp, goal_emb

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

        # Current target: min-priority stack (rule-based, giống original)
        current_target_emb = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]

        # Long-term goal: learned (cost-aware)
        target_mask = (env.x.amax(dim=-1) > 0).bool()
        goal_actions, goal_logp, goal_embeddings = self.select_goal(
            node_embeddings, graph_embedding, target_mask)

        # Accumulate goal selection log probs for REINFORCE
        goal_ll = goal_logp.clone()

        dest_mask = env.create_mask()

        for i in range(max_stacks * max_tiers * max_tiers):
            assert i < max_stacks * max_tiers * max_tiers - 1

            # prepare keys/values for destination attention
            node_keys = self.W_K1(node_embeddings)
            node_values = self.W_V(node_embeddings)

            # Dual-context destination selection
            context = (self.W_target(current_target_emb)
                       + self.W_goal(goal_embeddings)
                       + self.W_global(graph_embedding)).unsqueeze(1)

            query_ = self.W_Q(self.MHA([context, node_keys, node_values]))

            logits = torch.matmul(query_, node_keys.permute(0, 2, 1)).squeeze(1) / math.sqrt(query_.size(-1))
            logits = self.tanh_c * torch.tanh(logits)
            logits = logits - dest_mask.squeeze(-1) * 1e9
            log_p = torch.log_softmax(logits, dim=1)

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

            # Update current target (rule-based, từ env sau step+clear)
            current_target_emb = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]

            # Re-select long-term goal
            target_mask = (env.x.amax(dim=-1) > 0).bool()
            goal_actions_h, goal_logp_h, goal_embeddings = self.select_goal(
                node_embeddings, graph_embedding, target_mask)
            goal_ll = goal_ll + goal_logp_h  # accumulate for training

            dest_mask = env.create_mask()

        return cost, ll, goal_ll
