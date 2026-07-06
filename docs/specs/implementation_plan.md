# Implementation Plan: Hierarchical Target-Then-Relocate (HTR)

## Chiến Lược Phát Triển

**Laptop (Windows):** Code + Smoke test — đảm bảo chạy đúng luồng, đúng kết quả
**Colab (Linux + GPU):** Training thật + Experiments

Mọi code phải chạy được cả trên Windows laptop lẫn Linux Colab.

---

## Phase 1: High-Level Environment (Laptop, ~2 ngày)

### File: `env/high_level_env.py` (mới)

High-level MDP với action = target stack index.

```python
class HighLevelCRPSPEnv:
    """
    Khác với baseline env:
    - Action: stack index (0..S_y-1)
    - step() thực hiện target selection + blocker relocation + transfer closure
    - Reward: -số relocations trong macro-step này
    """
```

**Các method cần implement:**

| Method | Mô tả | Smoke test |
|--------|-------|------------|
| `__init__` | Nhận low-level solver (heuristic/A*/beam), config | — |
| `reset(instance)` | Reset yard, vessel, chạy transfer closure | 1 instance |
| `step(action)` | Dọn target stack → reward + next state + done | 1 macro-step |
| `_clear_target(stack_idx)` | Core logic: tìm target, gọi low-level solver | 1-2 cases |
| `_is_transferable(c)` | Kiểm tra container có thể transfer ngay | Vài TH đơn giản |
| `encode()` | State matrix (giống baseline) | — |
| `action_mask()` | Mask các stack rỗng | — |

**Kiến trúc:**

```python
class HighLevelCRPSPEnv:
    def __init__(self, low_level_solver="heuristic", terminal_bonus=10.0):
        self.low_level_solver = low_level_solver  # "heuristic" | "astar" | "beam"
        self.terminal_bonus = terminal_bonus

    def reset(self, instance):
        self.inst = instance
        self.yard = [list(s) for s in instance.yard]
        self.vessel = [[] for _ in range(instance.s_v)]
        self.slot_of = slot_map(instance)
        self.n_relocations = 0
        self.n_transfers = 0
        self._transfer_closure()
        self.done = all(not s for s in self.yard)
        return self._obs(), self._mask()

    def step(self, action):
        # action = stack index to clear
        stack_idx = action
        # Find target = top container of chosen stack
        if not self.yard[stack_idx]:
            raise ValueError(f"Stack {stack_idx} is empty")
        
        target = self.yard[stack_idx][-1]
        
        if self._is_transferable(target):
            # Direct transfer, no relocation needed
            self._transfer(target)
            reward = 0
        else:
            # Need to relocate blockers above target
            blockers = self.yard[stack_idx][:-1]  # all except top
            n_blk = len(blockers)
            # Find destination for each blocker using low-level solver
            if self.low_level_solver == "heuristic":
                result = solve_blocker_relocation_heuristic(
                    target, self.yard, self.slot_of, self.inst
                )
            elif self.low_level_solver == "astar":
                result = solve_blocker_relocation_astar(
                    target, self.yard, self.slot_of, self.inst
                )
            elif self.low_level_solver == "beam":
                result = solve_blocker_relocation_beam(
                    target, self.yard, self.slot_of, self.inst
                )
            # Apply relocation sequence
            n_reloc = len(result)
            for s, d in result:
                self.yard[d].append(self.yard[s].pop())
            self.n_relocations += n_reloc
            reward = -n_reloc

        # Transfer closure after relocation
        self._transfer_closure()
        self.done = all(not s for s in self.yard)
        
        if self.done:
            reward += self.terminal_bonus

        return self._obs(), self._mask(), reward, self.done, False, {}
```

---

## Phase 2: Low-Level Subproblem Solvers (Laptop, ~2 ngày)

### File: `env/subproblem.py` (mới)

Ba solver cho blocker relocation subproblem.

### 2.1 Greedy Heuristic Solver

```python
def solve_blocker_relocation_heuristic(target, yard, slot_of, inst):
    """
    Ý tưởng: mỗi blocker, chọn destination stack ít tạo blocking pairs nhất.
    Score = số container trong destination phải precede blocker.
    """
```

**Thuật toán:**
1. Tìm stack s chứa target
2. Với mỗi blocker b từ gần target nhất đi lên:
   - Với mỗi destination d ≠ s, yard[d] chưa đầy:
     - Score = count(c in yard[d] | must_precede(c, b))
   - Chọn d có score thấp nhất → relocate b → d
3. Trả về list các cặp (s, d)

**Smoke test:** 5 instances, so sánh kết quả với baseline heuristic.

### 2.2 A* Solver

```python
def solve_blocker_relocation_astar(target, yard, slot_of, inst):
    """
    Dùng A* của baseline để giải subproblem.
    Subproblem = chỉ gồm blocker containers + target container.
    Node limit = 500.
    """
```

**Thuật toán:**
1. Tạo subproblem instance (copy yard, loại bỏ containers không liên quan)
2. Gọi `solve_astar()` từ baseline với node_limit=500
3. Trả về trajectory

**Smoke test:** 5 instances, verify output format.

### 2.3 Beam Search Solver

```python
def solve_blocker_relocation_beam(target, yard, slot_of, inst, beam_width=5):
    """
    Beam search: mỗi level = một blocker cần đặt.
    Score = lower_bound sau khi apply partial assignment.
    """
```

**Smoke test:** 5 instances, beam_width = 1 (greedy) phải match heuristic.

---

## Phase 3: High-Level Policy + PPO Training (Laptop + Colab)

### File: `model/htr_agent.py` (mới)

```python
class HTRAgent:
    """
    High-level policy network.
    Input: yard matrix (S_y x T_y) + vessel heights (S_v)
    Network: RowSelfAttention (giống baseline) → flatten → Linear(128) → ReLU → Linear(S_y)
    Output: action logits (S_y)
    """
```

**Smoke test (laptop):** forward pass, action sampling, mask.

### File: `trainer.py` (sửa — thêm HTR mode)

PPO training loop cho HTR:
- Kế thừa `compute_gae()` từ baseline PPO
- Thay `CRPSPEnv` bằng `HighLevelCRPSPEnv`
- Action dim = S_y (thay vì S_y·(S_y-1))
- Thêm supervised warm-start phase

```python
def train_htr(cfg, device):
    # Phase 1: Supervised warm-start (CPU, nhanh)
    policy = HTRAgent(...)
    optimizer = Adam(policy.parameters(), lr=cfg.lr)
    
    print("Phase 1: Generating A* demonstrations...")
    demo_buffer = generate_demonstrations(n_instances=500, solver="astar")
    # demo_buffer[i] = (obs, target_stack_idx)
    
    print("Phase 2: Behavioral cloning...")
    for epoch in range(50):
        loss = supervised_loss(policy, demo_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Phase 3: PPO fine-tune
    print("Phase 3: PPO fine-tune...")
    env = HighLevelCRPSPEnv(low_level_solver="heuristic")
    for iteration in range(cfg.iterations):
        rollout_buffer = collect_rollouts(policy, env, cfg.instances_per_iter)
        advantages, targets = compute_gae(rollout_buffer, cfg.gamma, cfg.gae_lambda)
        # PPO update (same as baseline)
        update_policy(policy, rollout_buffer, advantages, targets, cfg)
```

**Smoke test (laptop):**
- 5 iterations PPO với N=5, S_y=3 (tiny)
- Kiểm tra loss giảm, không crash
- ~30 giây

**Full training (Colab):**
- N=15, S_y=5, 2000 iterations
- ~2 GPU hours

---

## Phase 4: Subproblem Evaluation (Laptop)

### File: `benchmarks/evaluate_subproblem.py`

So sánh 3 low-level solvers:

| Solver | Quality (gap vs optimal A*) | Time (per subproblem) |
|--------|---------------------------|----------------------|
| Greedy heuristic | ~? | ~μs |
| Beam width=3 | ~? | ~μs |
| Beam width=5 | ~? | ~μs |
| A* (node_limit=500) | 0% (optimal) | ~ms |

**Smoke test (laptop):** 20 subproblems, verify tất cả solvers chạy được, output đúng format.

---

## Phase 5: Full Comparison with Baseline (Colab)

### File: `benchmarks/benchmark_htr.py`

Chạy tất cả experiments so sánh HTR với baseline:

| Experiment | Scale | Baseline methods | Metrics |
|-----------|-------|-----------------|---------|
| Table 4 replication | N=15, S_y=5 | PPO, A*, Voting | Gap, time |
| Scaling test | N=30, S_y=6 | A*, PPO | Gap, solve rate |
| Large scale | N=50, S_y=8 | A* (partial), PPO | Gap, solve rate |
| Real scale | N=100, S_y=11 | None (first!) | Gap lower bound, time |

### 5.1 Ablation Studies

| Ablation | Configs | Questions |
|----------|---------|----------|
| Low-level solver | Greedy vs A* vs Beam(3) vs Beam(5) | Quality-speed tradeoff? |
| High-level | HTR vs Random vs Priority-rule | Does RL help? |
| Training solver | heuristic vs A* | Asymmetric training effect? |
| Warm-start | With vs without BC | How much does warm-start help? |

---

## Phase 6: Colab Deployment Script

### File: `colab_train.ipynb`

Script chạy trên Colab:

```python
# 1. Clone repo
!git clone https://github.com/sutobode/CRP_RL.git
%cd CRP_RL

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run HTR training (N=15)
!python trainer.py --mode htr --n 15 --s_y 5 --s_v 5 --t_y 5 \
    --iterations 2000 --instances_per_iter 10 --lr 5e-4 \
    --gamma 0.4 --gae_lambda 0.9 --clip_eps 0.15 \
    --warm_start --warm_start_instances 500 \
    --low_level_solver heuristic --out checkpoints/htr_n15.pt

# 4. Run benchmark comparison
!python benchmarks/benchmark_htr.py --ckpt checkpoints/htr_n15.pt \
    --scales "15,5,5,5" "30,6,6,6" "50,8,8,6" "100,11,11,6" \
    --output results/htr_results.csv

# 5. Run ablations
!python benchmarks/benchmark_htr.py --ablations \
    --ckpt checkpoints/htr_n15.pt --output results/ablations.csv
```

---

## Tổng Quan Files Cần Tạo/Sửa

| File | Trạng thái | Dòng | Laptop/Colab |
|------|-----------|------|--------------|
| `env/high_level_env.py` | **Mới** | ~120 | Laptop |
| `env/subproblem.py` | **Mới** | ~120 | Laptop |
| `model/htr_agent.py` | **Mới** | ~80 | Laptop |
| `trainer.py` | **Sửa** | Thêm HTR mode ~100 | Cả hai |
| `benchmarks/evaluate_subproblem.py` | **Mới** | ~60 | Laptop |
| `benchmarks/benchmark_htr.py` | **Mới** | ~150 | Colab |
| `colab_train.ipynb` | **Mới** | ~50 | Colab |

**Total code mới:** ~680 dòng
**Baseline code tái sử dụng (không sửa):** ~1500 dòng (instance.py, lower_bound.py, transfer.py, astar.py, heuristic.py, model/*, trainer.py base)

---

## Timeline

| Phase | Nội dung | Thiết bị | Thời gian |
|-------|----------|---------|-----------|
| 1 | env/high_level_env.py | Laptop | 2 ngày |
| 2 | env/subproblem.py + smoke test | Laptop | 2 ngày |
| 3 | model/htr_agent.py + trainer.py HTR mode | Laptop | 2 ngày |
| 4 | evaluate_subproblem.py benchmark | Laptop | 1 ngày |
| 5 | benchmark_htr.py + colab script | Laptop | 1 ngày |
| 6 | Full training + experiments | Colab | ~2 ngày |
| 7 | Analysis + paper writing | Laptop | ~5 ngày |
| **Total** | | | **~15 ngày** |

---

## Lưu Ý Kỹ Thuật

1. **Cross-platform:** Dùng `os.path.join`, `pathlib.Path`, tránh hardcode Windows path
2. **CUDA check:** `device = "cuda" if torch.cuda.is_available() else "cpu"` — tự động dùng GPU trên Colab
3. **Random seed:** Fix seed cho reproducibility trên cả hai môi trường
4. **Checkpoint:** Save model ở cuối mỗi phase, load được trên Colab
5. **Dependency:** Chỉ dùng PyTorch + numpy (giống baseline), không thêm thư viện mới
