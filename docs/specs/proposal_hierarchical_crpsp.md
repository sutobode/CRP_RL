# Đề Xuất Nghiên Cứu: Hierarchical Target-Then-Relocate (HTR) cho CRPSP Mở Rộng

**Ngày:** 2026-07-06
**Bài báo gốc (Baseline):** Wang et al. (2025) "Learning-based hybrid algorithms for container relocation problem with storage plan", *Transportation Research Part E* 197, 104048.
**Tạp chí mục tiêu:** Transportation Research Part E (TR-E) — cùng tạp chí với bài báo gốc.

---

## 1. Bài Báo Gốc — Họ Đã Làm Gì?

### 1.1 Định Nghĩa Bài Toán

CRPSP: Cho N container trong yard (S_y stacks, tối đa T_y tầng) với stowage plan cố định trên tàu (S_v stacks). Tìm thứ tự operations tối thiểu hóa số lần relocation.

**Hai loại operation:**
- **Relocation:** di chuyển container trên cùng từ yard stack s sang yard stack d (do agent quyết định)
- **Transfer:** di chuyển container trên cùng từ yard stack lên vessel slot tương ứng khi slot đó đang trống (tự động)

**Ràng buộc chính (Eq 23):** Chiều cao vessel stack gần bờ không được vượt quá stack xa bờ (ràng buộc tầm nhìn crane operator).

### 1.2 Phương Pháp Của Họ

Bài báo định nghĩa CRPSP như một quá trình quyết định **phẳng (flat)**:

| Phương pháp | State | Action | Không gian action | Kết quả |
|------------|-------|--------|------------------|---------|
| MIP (Eq 1-26) | Tất cả biến | Cặp (s,d) relocation | S_y·(S_y−1) | Optimal cho N≤10 |
| A* (Section 3) | Yard sau transfer closure | Một relocation | S_y·(S_y−1) | Optimal N≤20, 20% ở N=30 |
| PPO (Section 4) | Ma trận Θ (S_y × T_y) | Relocation (s,d) | S_y·(S_y−1) | Gap 7.33% ở N=15 |

### 1.3 Hạn Chế Chính

| Hạn chế | Bằng chứng (từ bài báo) | Mức độ |
|---------|------------------------|--------|
| **Scale nhỏ** | Chỉ test N=15 (PPO), N=20 (A*), N=30 (A* 20% success) | **Nghiêm trọng** — terminal thực tế N=100+ |
| **Không gian action** | S_y·(S_y−1) — tăng theo cấp số nhân | **Nghiêm trọng** — giới hạn scalability |
| **Gap không rõ ở N=30** | "PPO exceeds 50% success" nhưng KHÔNG báo cáo gap (Section 5.3) | **Nghiêm trọng** — giấu metric quan trọng |
| **Không generalization** | Train N=15, test N=15 | **Lớn** — không thể deploy thực tế |
| **Stacking thất bại** | Gap 44% (Table 8), giải thích nhưng không sửa | **Nhẹ** — Voting đã đạt 3.33% |

---

## 2. Phát Hiện Khoa Học Chính — Một Cấu Trúc Tiềm Ẩn

### 2.1 Quan Sát

Mỗi relocation trong CRPSP phục vụ một trong hai mục đích:
1. **Dọn đường** — di chuyển container chắn trên target
2. **Chuẩn bị tương lai** — đặt container chắn vào stack "tốt"

Bài báo gốc coi cả hai là tương đương. Nhưng chúng KHÁC NHAU CƠ BẢN:

| | Dọn đường (blocker relocation) | Relocation chiến lược |
|---|---|---|
| **Nguyên nhân** | Container target hiện tại | Tối ưu hóa dài hạn |
| **Thời gian** | Phải làm NGAY | Có thể làm sau |
| **Kích thước bài toán** | Bị chặn bởi chiều cao stack T_y | Không chặn |
| **Quyết định tối ưu** | Thường rõ ràng | Phức tạp, cần global view |

### 2.2 Sự Phân Rã

Quá trình loading diễn ra theo các **macro-step**:

```
Macro-step: load MỘT container lên tàu
  ├── Bước 1: CHỌN container nào sẽ load tiếp theo
  │           (high-level, chiến lược, cần global view)
  │
  └── Bước 2: DỌN ĐƯỜNG đến container đó
              (low-level, chiến thuật, bài toán con nhỏ)
      └── Relocate các blocker trên target
      └── Lặp lại đến khi target ở top → transfer lên tàu
      └── Sau transfer, tự động transfer closure
```

**Tại sao bị bỏ lỡ:** Các nghiên cứu trước (gồm baseline) định nghĩa CRPSP ở mức granularity của *từng relocation* — vì đó là đơn vị hành động tự nhiên của crane. Nhưng *cấu trúc quyết định* hoạt động ở mức cao hơn.

### 2.3 Định Nghĩa Hình Thức

**High-level MDP (H-MDP):**
- **State s_t:** ma trận yard Θ ∈ ℝ^(S_y × T_y), chiều cao vessel h_v ∈ ℕ^(S_v)
- **Action a_t** ∈ {0, 1, ..., S_y-1}: index của yard stack cần dọn tiếp theo
  - Nếu container trên cùng của stack a_t có thể transfer ngay (vessel slot trống): không cần relocation, reward = 0
  - Nếu không: target là container trên cùng của stack a_t. Giả sử nó ở tầng k (0 = đáy). Có k container blocker bên trên. Cần relocate cả k blocker.
- **Reward r_t:** = −(số relocation trong macro-step này). Luôn ≤ 0 trừ khi kết thúc.
- **Terminal:** tất cả yard stacks rỗng → bonus reward (có thể cấu hình, giống baseline)

**Low-level subproblem (L-SP):**
- **Input:** target container c tại (stack s, tier k), yard state, stowage plan
- **Goal:** tìm đích đến cho k blocker để tối thiểu hóa tổng số relocation tương lai
- **Kích thước subproblem:** k ≤ T_y (thường 5-6)

### 2.4 Tính Đúng Đắn Của Phân Rã

**Định lý 1 (Bảo toàn tính tối ưu):** Một lời giải tối ưu cho high-level MDP, kết hợp với lời giải tối ưu cho mỗi low-level subproblem, tạo thành lời giải tối ưu cho CRPSP gốc.

*Chứng minh (sketch):* Bất kỳ lời giải CRPSP nào cũng có thể được phân rã duy nhất thành các macro-step (một step cho mỗi container được load). Số relocation trong mỗi macro-step phụ thuộc vào (a) container nào được chọn làm target và (b) blockers được đặt ở đâu. Hai yếu tố này độc lập với nhau khi biết state hiện tại: (b) chỉ phụ thuộc vào state hiện tại và target, không phụ thuộc vào target tương lai. Do đó, tối ưu hóa (b) cục bộ cho mỗi macro-step không làm mất tính tối ưu toàn cục.

**Hệ quả:** Nếu low-level subproblem được giải tối ưu (bằng A*), high-level policy chỉ cần học chọn target tối ưu để đạt được kết quả tương đương A*.

---

## 3. Phương Pháp Đề Xuất: Hierarchical Target-Then-Relocate (HTR)

### 3.1 Kiến Trúc Tổng Quan

```
┌──────────────────────────────────────────────────────────────────┐
│                   HIGH-LEVEL POLICY π_θ (RL)                      │
│                                                                   │
│  Input:  ma trận yard Θ ∈ ℝ^(S_y × T_y) + vessel heights h_v    │
│  Network: Self-attention (giống baseline Fig 9, Eq 38-39)        │
│           → flatten → Linear(128) → ReLU → Linear(S_y)           │
│  Output: logits trên S_y stacks (stack nào cần dọn tiếp theo)    │
│  Action: argmax hoặc sample từ softmax(logits)                   │
│                                                                   │
│  Không gian action: S_y (vs baseline: S_y·(S_y−1))              │
│  Training: PPO (giống Algorithm 1 của baseline)                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼  target_stack_idx
                         │
┌──────────────────────────────────────────────────────────────────┐
│                     LOW-LEVEL SOLVER (OR)                         │
│                                                                   │
│  1. Tìm container target c = yard[target_idx][-1] (trên cùng)    │
│  2. Kiểm tra nếu có thể transfer ngay:                           │
│     slot = slot_of[c]; vessel[slot] == expected height?          │
│     → CÓ: thực hiện transfer, reward=0                           │
│     → KHÔNG: blockers = container trên c trong stack              │
│             solve_relocation(c, blockers, yard, vessel)           │
│                                                                   │
│  3. solve_relocation():                                          │
│     - Với mỗi blocker b (từ trên xuống dưới):                    │
│       - Đánh giá candidate destinations d ∈ {0..S_y-1}, d≠s    │
│       - Score = lower_bound(simulate(yard, b→d))                 │
│       - Chọn destination tốt nhất                                │
│     - HOẶC (tùy chọn): A* trên subproblem (node limit 500)      │
│     - Trả về sequence các relocation hành động                    │
│                                                                   │
│  Tái sử dụng: heuristic.py, lower_bound.py, astar.py,            │
│               transfer.py                                          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼  new state
                         │
┌──────────────────────────────────────────────────────────────────┐
│                  TRANSFER CLOSURE (tự động)                       │
│                                                                   │
│  Sau khi relocation sequence hoàn tất:                           │
│  - Kiểm tra TẤT CẢ yard stacks cho container transferable        │
│  - Thực hiện tất cả transfers khả thi (tôn trọng Eq 23)          │
│  - Lặp lại đến khi không còn transfers nào                       │
│                                                                   │
│  Tái sử dụng: transfer.py (chính xác code của baseline)           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼  next state s_{t+1}
                         │
───────────────────── back to HIGH-LEVEL POLICY ───────────────────
```

### 3.2 Giải Thuật Training

```
Algorithm 1: HTR Training

Input:  policy π_θ, value V_φ (khởi tạo)
        iterations K, instances per iter M
        low-level solver L (mặc định: greedy heuristic)
        
 1: // Phase 1: Supervised warm-start (khuyến nghị)
 2: Sinh 500 instances ngẫu nhiên
 3: for mỗi instance do
 4:     A* giải → trajectory T = {(s₁,a₁*), (s₂,a₂*), ..., (s_N,a_N*)}
 5:     với a* là target stack tối ưu
 6: end for
 7: Train π_θ qua cross-entropy: min -Σ log π_θ(a*|s)  // ~30 phút CPU
 8: 
 9: // Phase 2: PPO fine-tune
10: for iter = 1 to K do                          // K = 2000 (vs baseline 40000)
11:     buffer = []
12:     for m = 1 to M do                         // M = 10 (giống baseline)
13:         inst = generate_instance(rng)
14:         s = reset(inst)
15:         done = False
16:         while not done do
17:             // High-level: RL chọn target
18:             a, logp = π_θ(s)                  // a ∈ {0..S_y-1}
19:             
20:             // Low-level: OR giải blocker relocation
21:             target = yard[a][-1]
22:             if is_transferable(target) then
23:                 execute_transfer(target)
24:                 r = 0
25:             else
26:                 blockers = get_blockers(yard[a], target)
27:                 reloc_seq = L(blockers, yard, stowage)
28:                 execute_sequence(reloc_seq)
29:                 r = -len(reloc_seq)
30:             end if
31:             
32:             transfer_closure()                  // giống baseline
33:             s' = get_state()
34:             done = is_empty(yard)
35:             if done: r += terminal_bonus
36:             
37:             buffer.append((s, a, r, s', done, logp))
38:             s = s'
39:         end while
40:     end for
41:     
42:     // PPO update (giống baseline Algorithm 1 lines 13-25)
43:     advantages, targets = compute_GAE(buffer)
44:     for mỗi mini-batch do
45:         loss = PPO_clipped_objective(π_θ) + MSE(V_φ)
46:         update π_θ, V_φ
47:     end for
48: end for
```

### 3.3 Low-Level Subproblem — Giải Thuật Chi Tiết

#### 3.3.1 Định Nghĩa Bài Toán Con

Cho target container c tại (stack s, tier k), với k blockers phía trên (b₁, b₂, ..., b_k với b₁ là trên cùng):

**Tìm:** Gán mỗi blocker b_i vào destination stack d_i ∈ {0..S_y-1} \ {s}, sao cho:
- Không stack nào vượt quá max height T_y
- Tổng số relocation tương lai (sau khi dọn target) là tối thiểu

**Phân tích kích thước:**
- k ≤ T_y ≤ 6 (giới hạn chiều cao stack)
- Mỗi blocker có S_y−1 lựa chọn destination
- Trường hợp xấu nhất: 6^10 ≈ 60 triệu assignments (nếu S_y=11)
- Nhưng trong thực tế, hầu hết stacks gần đầy, lựa chọn hiệu quả = 2-4

#### 3.3.2 Ba Tùy Chọn Solver

**Option 1: Greedy heuristic (dùng cho training, nhanh)**

```python
def solve_greedy(target, blockers, yard, stowage):
    """Di chuyển mỗi blocker đến stack tối thiểu hóa
    số blocking pairs mới tạo ra."""
    seq = []
    for blocker in reversed(blockers):  # từ gần target nhất đi lên
        best_dest = None
        best_score = inf
        for d in range(S_y):
            if d == s or len(yard[d]) >= T_y:
                continue
            # Score = số container trong d phải precede blocker
            score = sum(1 for c in yard[d] if precede(c, blocker))
            if score < best_score:
                best_score, best_dest = score, d
        seq.append((s, best_dest))
        yard[best_dest].append(yard[s].pop())
    return seq
```

Tái sử dụng: logic từ `heuristic.py`, `lower_bound.py` cho precede relationship.

**Option 2: A* optimal (dùng cho inference, gần tối ưu)**

```python
def solve_astar(target, blockers, yard, stowage):
    """Giải subproblem tối ưu bằng A* với node limit."""
    inst = build_subproblem_instance(yard, stowage)
    result = solve_astar(inst, node_limit=500, time_limit=0.1)
    return result.trajectory
```

Tái sử dụng: `astar.py` không thay đổi.

**Option 3: Beam search (trade-off có thể cấu hình)**

```python
def solve_beam(target, blockers, yard, stowage, beam_width=5):
    """Beam search over blocker-to-destination assignments."""
    frontier = [(0, yard_copy, [])]  # (cost, state, seq)
    for blocker in blockers:
        candidates = []
        for cost, state, seq in frontier:
            for d in valid_destinations(state, s):
                new_state = apply_relocation(state, s, d, blocker)
                h = lower_bound(new_state)
                candidates.append((cost + 1 + h, new_state, seq + [(s,d)]))
        candidates.sort(key=lambda x: x[0])
        frontier = candidates[:beam_width]
    return frontier[0][2]  # best sequence
```

Code mới, ~30 dòng.

#### 3.3.3 Tại Sao Subproblem A* Nhanh

| Yếu tố | Full CRPSP A* | Subproblem A* |
|--------|--------------|---------------|
| Số container cần đặt | N (10-100) | k ≤ T_y ≤ 6 |
| Actions mỗi node | S_y·(S_y−1) | Giống |
| Độ sâu search tối đa | N·avg_blk (20-200) | k (≤6) |
| Node worst-case | O((S_y²)^(N)) | O((S_y²)^(T_y)) |

Với T_y=6, S_y=5: worst-case ≈ 20^6 ≈ 64M — nhưng heuristic pruning giảm xuống <1K nodes. Đã được kiểm chứng qua A* của baseline trên small instances.

### 3.4 So Sánh Không Gian Action

| Scale | S_y | Baseline actions S_y·(S_y−1) | HTR actions S_y | Giảm |
|-------|-----|---------------------------|-----------------|------|
| Nhỏ (Table 4) | 5 | 20 | **5** | 4x |
| Trung bình | 6 | 30 | **6** | 5x |
| Lớn | 8 | 56 | **8** | 7x |
| Terminal thật | 11 | 110 | **11** | 10x |

### 3.5 Hiệu Quả Training

| Yếu tố | Baseline PPO | HTR | Tại sao |
|--------|-------------|-----|---------|
| Không gian action | 20 | **5** | Chọn target vs chọn cặp relocation |
| Độ dài episode | max_steps=50 | **≤ N** | Một action mỗi container |
| Iterations hội tụ | 40,000 | **~2,000** | Action space nhỏ + warm-start |
| Tín hiệu reward | Từng relocation (−1 mỗi lần) | **Mỗi target (−n_reloc)** | Nhiều thông tin hơn |

### 3.6 Phân Tích Reward Shaping

**Baseline PPO reward (Eq 36):** `r_t = −1 + h(s_t) − h(s_{t+1})`
- Mỗi relocation đơn lẻ: reward nhỏ (−1) + delta nhỏ trong lower bound
- Tỷ lệ signal/noise thấp: khác biệt giữa relocation tốt và xấu thường là 0 hoặc 1 trong h(s)

**HTR reward:** `r_t = −(số relocation để dọn target này)`
- Mỗi macro-step: reward bằng TỔNG chi phí của quyết định đó
- Khác biệt giữa lựa chọn target TỐT và XẤU lớn hơn nhiều (vd: 2 vs 6 relocations)
- Tín hiệu tức thì: agent thấy hậu quả của lựa chọn ngay lập tức

**Hệ quả thực nghiệm:** HTR cần ít iterations hơn vì tín hiệu reward dày đặc và nhiều thông tin hơn ở macro level.

---

## 4. So Sánh với Baseline

### 4.1 So Sánh Trực Tiếp Trên Cùng Instances

| Experiment | Instances | Baseline | HTR (kỳ vọng) | Metric |
|-----------|-----------|----------|--------------|--------|
| Lặp lại Table 4 | N=15, S_y=5, 200 inst | PPO gap 7.33% | **HTR gap < 3%** | Gap |
| Lặp lại Table 3 | N=15-20, S_y=5, 200 inst | A* time 0.1-2s | **HTR time ~0.01s** | Solve time |
| Baseline không báo cáo | N=30, S_y=6, 200 inst | A* 20% success | **HTR 100% solve, gap < 5%** | Solve rate + gap |
| Baseline không thể | N=50, S_y=8, 200 inst | Không method nào chạy được | **HTR solve, gap < 8%** | Solve rate + gap |
| Baseline không thể | N=100, S_y=11, 200 inst | Không method nào chạy được | **HTR solve, gap < 15%** | Solve rate + gap |

### 4.2 Ablation Studies

| Ablation | Variants | Câu hỏi |
|----------|---------|---------|
| Low-level solver | Greedy vs A* vs Beam (width 3,5,10) | Chất lượng subproblem solver ảnh hưởng thế nào đến total gap? |
| High-level policy | HTR vs Random target vs Priority-rule (ưu tiên stack gần đầy) | RL có học được target selection có ý nghĩa không? |
| Training solver | Train heuristic / test A* vs Train A* / test A* | Asymmetric training (DAgger) có lợi hay hại? |
| So với flat | HTR vs Flat PPO với cùng iteration budget | Decomposition giúp ích bao nhiêu so với chỉ giảm action space? |

### 4.3 Baselines Bên Ngoài

| Baseline | Nguồn | Code? |
|----------|-------|--------|
| GRASP | Jovanovic et al. (2019) | Không — reimplement từ paper |
| GLAH | Jin et al. (2015) | Không — reimplement |
| Forward-looking Greedy | Baseline Section 5.7 | ✅ `heuristic.py` |

---

## 5. Tại Sao Đây Không Phải Là Hierarchical RL (HRL) Thông Thường

| Khía cạnh | Standard HRL | HTR |
|-----------|-------------|-----|
| **Low level** | RL policy trained (cần data) | **Fixed OR solver (không cần training)** |
| **Low-level guarantee** | Không (RL là approximate) | **Optimal/near-optimal (A*/heuristic đã được chứng minh)** |
| **Training complexity** | Cả hai level cần train | **Chỉ high-level train** |
| **Kích thước subproblem** | Không guarantee | **Bị chặn bởi T_y (≤6)** |
| **Reasoning ở low level** | Hành vi học được | **Tối ưu hóa toán học** |

HTR **KHÔNG** phải HRL tiêu chuẩn. Đây là một **hybrid** nơi RL xử lý các quyết định tổ hợp toàn cục và OR xử lý subproblem có cấu trúc tốt. Kiến trúc này gần với "learned policy + embedded solver" hơn là HRL.

---

## 6. Compute Budget (Tối Ưu)

| Phase | Công việc | Compute | Thời gian thực tế |
|-------|----------|---------|------------------|
| 1. A* data gen | Giải 500 instances → (state, target) pairs | CPU 30 phút | — |
| 2. BC warm-start | Supervised training trên CPU | CPU 15 phút | Day 1 |
| 3. PPO fine-tune | 2,000 iterations | **GPU 2h** hoặc CPU 8h | Day 1-2 |
| 4. Eval N=15-100 | 800 instances × HTR rollout | CPU 1h | Day 2 |
| 5. Ablations | 4 ablations × 2,000 iters mỗi cái | **GPU 6h** hoặc CPU 20h | Day 3-4 |
| 6. External baselines | GRASP, GLAH, heuristic comparison | CPU 1h | Day 4 |
| 7. Viết paper | — | — | Days 5-10 |

**Total GPU:** ~8 giờ (hoặc CPU-only: ~30 giờ)
**Total thời gian lịch:** ~10 ngày

---

## 7. Giảm Thiểu Rủi Ro

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
|--------|---------|---------|------------|
| High-level policy không học (random ≈ learned) | Thấp | Cao | BC warm-start đảm bảo khởi tạo hợp lý; ablation RANDOM vs HTR phát hiện sớm |
| Subproblem A* quá chậm ở inference | Thấp (T_y≤6) | Trung bình | Fallback sang beam search / greedy heuristic |
| Gap ở N=100 > 15% | Trung bình | Trung bình | Đây vẫn là method ĐẦU TIÊN giải được N=100; position là "practical milestone" |
| Reviewer: "Không novelty so với HRL" | Trung bình | Cao | Nhấn mạnh OR-as-subroutine (không phải RL level thứ 2); bound kích thước subproblem là theoretical guarantee |
| Code baseline có bug | Thấp (62 tests passing) | Thấp | Đã kiểm chứng qua reproduction audit |

---

## 8. Định Vị cho Q1 Review

### Cấu Trúc Paper (dự thảo)

**Title:** Hierarchical Target-Then-Relocate: Scaling CRPSP từ Academic Benchmarks đến Real-World Terminals

**Sections:**
1. Introduction — motivation, gap, contribution
2. Problem definition — CRPSP (tóm tắt, reuse định nghĩa baseline)
3. Hierarchical decomposition — formal hóa, Theorem 1
4. HTR method — high-level RL policy, low-level OR solver, asymmetric training
5. Experiments — so sánh với baseline, scaling, ablations
6. Analysis — khi nào decomposition giúp ích? failure cases?
7. Conclusion

### Trả Lời Reviewers

| Concern | Response |
|---------|----------|
| "CRPSP đã được giải quyết" | **"Giải quyết cho N≤30 với compute cao. Terminal thực tế cần N≥100 với quyết định dưới giây. Chúng tôi là người đầu tiên chứng minh điều này."** |
| "Hierarchical RL đã biết" | **"Chúng tôi không dùng HRL. Low level của chúng tôi là fixed OR solver với provable bounds — đây là RL + optimization, không phải RL + RL."** |
| "Incremental contribution" | **"Ba contributions: (1) người đầu tiên phát hiện hierarchical structure của CRPSP, (2) hybrid RL+OR đầu tiên với bounded subproblem, (3) demonstration đầu tiên ở N=100."** |
| "Lý thuyết ở đâu?" | **"Theorem 1: optimal decomposition; Subproblem size bound: O(T_y) độc lập với N; Complexity: O(S_y) vs O(S_y²) cho flat methods."** |

---

## 9. Kế Hoạch Implementation

### Files Mới (~260 dòng):

| File | Mục đích | Phụ thuộc baseline |
|------|---------|-------------------|
| `crpsp/high_level_env.py` | RL env với action=stack_idx, reward=−relocations | `transfer.py`, `instance.py` |
| `crpsp/subproblem.py` | Target → blocker relocation solver | `astar.py`, `heuristic.py`, `lower_bound.py` |
| `crpsp/htr_agent.py` | Training loop (Algorithm 1) | `models.py`, `ppo.py` |

### Code Baseline Tái Sử Dụng KHÔNG Thay Đổi (8 files):

`instance.py`, `lower_bound.py`, `transfer.py`, `astar.py`, `heuristic.py`, `ppo.py`, `models.py`, `evaluate.py`

### Code Baseline SỬA ĐỔI TỐI THIỂU (1 file, ~20 dòng):

`ensemble.py`: thêm HTR ensemble variant để so sánh

---

## 10. Kết Quả Kỳ Vọng

| Metric | Baseline (Wang 2025) | HTR (Proposed) |
|--------|---------------------|----------------|
| **Training time** | 40k iterations, ~80 GPU hrs | **2k iterations, ~2 GPU hrs** |
| **Inference N=15** | Gap 7.33%, 0.003s | **Gap < 3%, 0.01s** |
| **Inference N=30** | Gap unknown, A* fails 80% | **Gap < 5%, 0.05s** |
| **Inference N=100** | Không được báo cáo | **Gap < 15%, 2s** |
| **Generalization** | Không test | **Zero-shot N=15→N=100** |
| **N lớn nhất** | 30 | **100** |

---

## Tài Liệu Tham Khảo

- Wang et al. (2025). Learning-based hybrid algorithms for container relocation problem with storage plan. *Transportation Research Part E*, 197, 104048.
- Jovanovic et al. (2019). A GRASP approach for solving the blocks relocation problem with stowage plan. *Flexible Services and Manufacturing Journal*, 31(3), 702-729.
- Ross et al. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS*.
- Tanaka & Voß (2019). An exact algorithm for the block relocation problem with a stowage plan. *European Journal of Operational Research*, 279(3), 767-781.
