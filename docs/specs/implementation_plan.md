# Implementation Plan: HTR cho CRP_RL (Shin et al. 2026, TR-C)

## 1. Phân Tích Baseline

### 1.1 Codebase CRP_RL

CRP_RL giải bài toán Container Retrieval Problem với mục tiêu tối thiểu hóa **working time**:

```
Working time = travel_time + acceleration_time + pickup_deposit_time
             = (bay_diff * t_bay + row_diff * t_row + t_acc) + t_pd
```

### 1.2 Kiến Trúc Hiện Tại

```
Input: yard matrix (batch, n_bays, n_rows, n_tiers) với container priorities
  │
  ▼
Encoder (LSTM + Self-Attention) → node_embeddings (mỗi stack) + graph_embedding (global)
  │
  ▼
Decoder loop:
  1. env.find_target_stack() → chọn stack có priority nhỏ NHẤT (RULE-BASED)
  2. env.set_target_stack(target) 
  3. Attention over stacks → chọn destination stack (LEARNED)
  4. env.step(dest) → relocate + clear (tự động retrieve)
  5. Re-encode state → lặp
```

### 1.3 Hạn Chế Chính

| Hạn chế | Mô tả |
|---------|-------|
| **Target selection là rule-based** | `find_target_stack()` luôn chọn min-priority, không quan tâm working time cost |
| **Không có cost-awareness** | Có thể chọn target xa crane hơn dù có stack gần hơn với priority gần tương đương |
| **Không thể tối ưu working time qua target** | Chỉ tối ưu destination, không tối ưu target |

### 1.4 Các Baseline Có Sẵn

- Kim (2016)
- Lin (2015)  
- Durasevic (2025)
- Leveling

Tất cả đều dùng priority-based target selection. Không có baseline nào dùng learned target selection.

---

## 2. Phương Pháp Đề Xuất: HTR (Hierarchical Target-Then-Relocate)

### 2.1 Key Insight

CRP có hai quyết định ở hai level khác nhau:

| Level | Quyết định | Hiện tại | Proposed |
|-------|-----------|----------|----------|
| **High** | Chọn stack nào làm target? | Rule-based (min priority) | **Learned policy** |
| **Low** | Relocate top container đi đâu? | Learned (attention decoder) | Learned (giữ nguyên) |

**Target selection hiện tại bỏ qua working time.** Một learned policy có thể học trade-off giữa:
- **Urgency:** Container priority càng nhỏ càng cần retrieve sớm
- **Travel cost:** Crane di chuyển đến stack đó tốn bao nhiêu working time
- **Relocation cost:** Clear target đó cần relocate bao nhiêu container

### 2.2 Kiến Trúc HTR

```diff
- Decoder hiện tại:
    env.find_target_stack() [RULE] → decoder chọn dest

+ HTR Decoder:
    TargetSelector [LEARNED] → env.set_target(target) → decoder chọn dest
```

```
Input: yard matrix
  │
  ▼
Encoder (giữ nguyên) → node_embeddings + graph_embedding
  │
  ▼
HIGH-LEVEL: TargetSelector(net(node_emb, graph_emb)) → target_stack_idx
  │           MLP: [embed_dim*2 → 128 → 1] → softmax → sample
  │
  ▼
env.set_target_stack(target_idx)
  │
  ▼
LOW-LEVEL: Attention decoder (giữ nguyên) → destination_idx
  │
  ▼
env.step(dest) → relocate + clear → next state
  │
  ▼
Re-encode state → lặp HIGH-LEVEL
```

### 2.3 TargetSelector Network

```python
class TargetSelector(nn.Module):
    """
    Input:  node_embeddings (batch, n_stacks, embed_dim)
            graph_embedding (batch, embed_dim)
            mask (batch, n_stacks) — non-empty stacks
    Output: logits (batch, n_stacks) — target scores
    """
    embed_dim → concat(embed, global) → Linear(2*embed_dim, 128) → ReLU → Linear(128, 1)
```

### 2.4 Training

Joint training (cả target selector và destination decoder cùng được train):

```
forward() returns (cost, dest_ll, target_ll)
  - cost = total working time của episode
  - dest_ll = sum of log probs của destination actions (từ decoder)
  - target_ll = sum of log probs của target actions (từ target selector)

loss = REINFORCE(cost, dest_ll + target_ll)
  - Dùng POMO baseline (giống trainer.py hiện tại)
  - total_ll = dest_ll + target_ll → backprop qua cả hai network
```

Loss function chi tiết (POMO):
```
obj_reshaped = cost.view(batch/pomo, pomo)          # reshape theo POMO groups
obj_mean = obj_reshaped.mean(dim=1)                 # baseline = mean của pomo trajectories
obj_std = obj_reshaped.std(dim=1)                   # std normalization
advantage = (obj_reshaped - obj_mean) / (obj_std + 1e-8)
loss = (advantage * total_ll).mean()                # REINFORCE loss
```

---

## 3. So Sánh Với Baseline

| Yếu tố | CRP_RL gốc (Shin 2026) | HTR (Proposed) |
|--------|------------------------|----------------|
| **Target selection** | Rule-based: argmin priority | **Learned: cost-aware policy** |
| **Destination selection** | Learned: attention decoder | Learned: attention decoder (giữ nguyên) |
| **Working time cost** | Chỉ dùng làm reward | **Dùng làm training signal cho cả target + dest** |
| **Novelty** | — | **First learned target selection cho CRP working time** |

---

## 4. Files Cần Tạo/Sửa

### Files mới:

| File | Dòng | Mục đích |
|------|------|----------|
| `model/target_selector.py` | ~20 | Target selection network |
| `model/htr_decoder.py` | ~140 | Decoder với HTR target selection |

### Files sửa:

| File | Dòng sửa | Mục đích |
|------|---------|----------|
| `env/env.py` | +4 | Thêm `set_target_stack()` method |
| `model/model.py` | ~20 | Thêm HTR mode switch |
| `trainer.py` | ~20 | Thêm HTR training (handle target_logp) |
| `main.py` | +1 | Thêm `--htr` flag |

### Files không đụng:

`env/`, `generator/`, `baselines/`, `benchmarks/`, `model/encoder.py`, `model/sampler.py`, `model/decoder.py`

---

## 5. Kế Hoạch Thực Thi

### Phase 1: Laptop (code + smoke test)

| Task | File | Thời gian |
|------|------|-----------|
| 1.1 Tạo TargetSelector | `model/target_selector.py` | ~15 phút |
| 1.2 Sửa env | `env/env.py` (thêm set_target_stack) | ~5 phút |
| 1.3 Tạo HTRDecoder | `model/htr_decoder.py` | ~1 giờ |
| 1.4 Sửa Model wrapper | `model/model.py` | ~10 phút |
| 1.5 Sửa trainer | `trainer.py` (handle target_logp) | ~15 phút |
| 1.6 Smoke test | Chạy 5 epochs N=35, batch=16 | ~5 phút |

**Smoke test kỳ vọng:**
- Code chạy không crash
- Loss giảm dần (từ ~1000 xuống ~500 sau 5 epochs)
- Target selector chọn target khác min-priority ít nhất 20% steps

### Phase 2: Colab (training + benchmark)

| Task | Config | Thời gian |
|------|--------|-----------|
| 2.1 Full training | 1000 epochs, batch=128, pomo=16 | ~12 giờ |
| 2.2 Benchmark | Lee instances + Shin instances | ~1 giờ |
| 2.3 So sánh baselines | Kim, Lin, Durasevic, Leveling | ~2 giờ |

---

## 6. Rủi Ro

| Rủi ro | Khả năng | Giảm thiểu |
|--------|---------|------------|
| Target selector học được policy giống hệt rule-based (luôn chọn min priority) | Cao | Nếu xảy ra → ablation check: so sánh target selector vs rule-based |
| Thêm network làm tăng training time | Trung bình | TargetSelector chỉ ~10K params, không đáng kể so với encoder |
| HTRDecoder không tương thích POMO training | Thấp | Log prob được accumulate giống dest_ll, cùng REINFORCE loss |

---

## 7. Expected Results

| Experiment | Expected improvement over baseline |
|-----------|----------------------------------|
| Lee benchmark (1 bay, 70 containers) | Working time giảm 5-10% |
| Large instance (6 bays, 570 containers) | Working time giảm 10-15% (travel cost chiếm tỉ trọng lớn hơn) |
| Upside-down instances | Working time giảm 3-5% (ít cơ hội tối ưu) |
