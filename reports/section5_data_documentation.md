# Section 5: Data Documentation

## 5.1 Data Source

**Simulator**: CityFlow v1.0 (https://cityflow-project.github.io/)
- **Type**: Microscopic traffic simulation engine
- **Source**: Open-source, developed by MSRA CNSM group
- **License**: Apache 2.0
- **Modality**: Discrete-time traffic simulation with 1-second timesteps

**Rationale**: CityFlow provides a realistic, reproducible traffic environment with:
- Accurate vehicle dynamics (acceleration, deceleration, gap-keeping)
- Traffic signal control interface
- Deterministic simulation (seeded for reproducibility)

---

## 5.2 Dataset Size and Structure

### Training Data
- **Simulation episodes per model**: 50 episodes (baseline experiments)
- **Steps per episode**: 1,000 timesteps (1,000 seconds simulated time)
- **Total experiences per model**: ~50,000 state-action-reward tuples
- **Replay buffer size**: 5,000 most recent experiences
- **Independent runs per experiment**: 10 runs with different random seeds (42-51)

### Evaluation Data
- **Evaluation episodes**: 1 episode per trained model (post-training evaluation)
- **Evaluation length**: 1,000 timesteps
- **Seeds**: Match training seeds for paired statistical comparison

### No Explicit Train/Val/Test Split
**Reasoning**:
- RL operates on online learning (no pre-collected dataset)
- Each episode generates new experiences during training
- Evaluation performed on fresh environment instances post-training
- Statistical validity achieved through multiple independent runs (n=10)

---

## 5.3 State Space

### H1-Basic and H2-MaxPressure (Single-Agent)
**State representation**: Normalized lane waiting vehicle counts

```
State s_t = [w_1/w_max, w_2/w_max, ..., w_n/w_max]

where:
  w_i = waiting vehicle count on lane i
  w_max = max(w_1, ..., w_n) at time t
  n = number of lanes (56 for roadnet-adv.json)
```

**Dimensionality**:
- Single intersection (roadnet-adv): 56-dimensional
- 2-intersection corridor: varies by agent

**Normalization**: Min-max normalization per timestep (scale to [0, 1])

### H3-MultiAgent (Independent Agents)
**State representation**: Local lane waiting counts per intersection

```
State s_t^i = [w_1^i/w_max^i, ..., w_k^i/w_max^i]

where:
  i = intersection ID
  k = number of lanes controlled by intersection i
  w_max^i = local maximum for intersection i
```

**Dimensionality per agent**:
- intersection_1: 56-dimensional (roadnet-adv)
- intersection_1 (corridor): 1-dimensional
- intersection_2 (corridor): 2-dimensional

**Coordination**: Implicit (no communication between agents)

---

## 5.4 Action Space

### Discrete Traffic Signal Phase Control
**Action a_t ∈ {0, 1, ..., |P|-1}**

where:
- P = set of available signal phases for intersection
- |P| = 8 (roadnet-adv), 2 (corridor network)

**Phase Definitions** (roadnet-adv, 8-phase):
- Phases 0, 2, 4, 6: Green phases (30s default duration)
- Phases 1, 3, 5, 7: Yellow/all-red transitions (5s default)

**Action Effect**: Sets traffic light to specified phase immediately

**Constraints**:
- No minimum green time enforcement (DRL can switch freely)
- Yellow time handled by environment (fixed 5s in baseline phases)

---

## 5.5 Reward Structure

### H1-Basic (Queue Minimization)
```python
r_t = -mean(queue_lengths_t)
```
- Range: [-50, 0] typical (depends on traffic density)
- Interpretation: Negative average waiting vehicles across all lanes
- Goal: Maximize reward → minimize average queue length

### H2-MaxPressure (Normalized Total Queue)
```python
r_t = -sum(queue_lengths_t) / num_lanes
```
- Range: [-50, 0] typical (after normalization fix)
- Interpretation: Negative total queue pressure, normalized by network size
- Goal: Minimize total system congestion

**Critical Fix**: Original H2 used unnormalized `-sum(queue_lengths)`, resulting in rewards 50x larger than H1 (~-300 to -600 vs -5 to -10). Fixed by dividing by `num_lanes` to match H1's scale.

### H3-MultiAgent (Local Queue Minimization)
```python
r_t^i = -mean(queue_lengths_t^i)  # Per intersection i
```
- Range: [-30, 0] typical per agent
- Interpretation: Each agent minimizes local queue length
- Total reward: Sum of all agent rewards

---

## 5.6 Traffic Scenarios

### Low-Variance Scenario
**File**: `scenarios/configs/config_low_variance.json`

**Traffic characteristics**:
- 12 vehicle flows from 4 directions
- Arrival intervals: 4.52s - 5.47s (low variance)
- Mean flow rate: ~200 vehicles/hour per direction
- Pattern: Stable, predictable traffic

**Distributions**:
- Vehicle inter-arrival time: ~N(5.0, 0.3) seconds
- Queue length: Stationary distribution after warmup

### High-Variance Scenario
**File**: `scenarios/configs/config_high_variance.json`

**Traffic characteristics**:
- Same 12 flows, different intervals
- Arrival intervals: Variable (high variance)
- Includes surge periods (3x normal flow)
- Pattern: Fluctuating demand

**Distributions**:
- Vehicle inter-arrival time: Higher variance
- Queue length: Non-stationary, periodic surges

### 2-Intersection Corridor
**File**: `scenarios/configs/config_2intersection_corridor.json`

**Traffic characteristics**:
- 1 flow through both intersections
- Arrival interval: 5.0s
- Tests coordination between sequential intersections
- Queue propagation from intersection_1 → intersection_2

---

## 5.7 Data Collection Methodology

### Training Loop
```python
for episode in range(50):
    state = env.reset()  # Fresh traffic generation
    for step in range(1000):
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, next_state, done))

        if len(buffer) > batch_size:
            agent.update(batch_size)  # Sample from buffer
```

### Experience Replay
- **Buffer type**: Uniform random sampling
- **Buffer size**: 5,000 (circular, overwrites oldest)
- **Batch size**: 128 experiences per update
- **Update frequency**: Every timestep (if buffer > 128)

### Evaluation Protocol
```python
agent.model.eval()  # Disable training mode
state = env.reset()
for step in range(1000):
    action = agent.greedy_action(state)  # No exploration
    state, reward, done, _ = env.step(action)
```

---

## 5.8 Normalization and Preprocessing

### State Normalization
**Method**: Per-timestep min-max scaling

```python
def normalize_state(lane_counts):
    max_count = max(lane_counts)
    if max_count == 0:
        return lane_counts  # All zeros
    return lane_counts / max_count
```

**Properties**:
- Ensures state values in [0, 1]
- Adaptive to current traffic conditions
- No global statistics required

**Limitation**: Scale changes over time (not batch-normalized)

### Reward Normalization
**H1 and H3**: None (raw queue counts)
**H2**: Division by `num_lanes` (scale alignment fix)

**No reward clipping**: Values unbounded (can be arbitrarily negative)

### No Augmentation
**Reasoning**:
- Traffic simulation is deterministic given seed
- State space is continuous (no discrete augmentation)
- Environment dynamics provide natural variance

---

## 5.9 Known Biases and Limitations

### Network Topology Bias
**Issue**: Majority of experiments on **single intersection**
- roadnet-adv.json: 1 controllable + 4 virtual boundary nodes
- Limits generalization to multi-intersection coordination

**Mitigation**:
- Created 2-intersection corridor for scalability testing
- H3 designed for multi-intersection (tested on corridor)

### Traffic Pattern Bias
**Issue**: Synthetic traffic flows (not real-world data)
- Uniform vehicle properties (length, speed, acceleration)
- Deterministic arrival patterns (fixed intervals)
- No driver heterogeneity (reaction times, aggression)

**Impact**:
- Results may not transfer to real-world traffic
- Simplified dynamics easier to learn (optimistic performance)

### Reward Engineering Bias
**Issue**: Hand-crafted rewards (queue minimization)
- Assumes minimizing queues = good traffic management
- Ignores other metrics (throughput, fairness, fuel consumption)
- Different reward designs favor different policies

**Examples**:
- H1 favors balanced queues (mean)
- H2 favors total reduction (sum)
- Could use multi-objective: `-α*queue - β*wait_time - γ*stops`

### Exploration Bias
**Issue**: Epsilon-greedy exploration only
- May miss optimal policies in sparse reward regions
- 50 episodes insufficient for full state space coverage

**Evidence**: Epsilon still at 0.74 after 50 episodes (74% random actions!)

### Replay Buffer Bias
**Issue**: Small buffer (5,000) forgets 90% of experiences
- 50 episodes × 1,000 steps = 50,000 experiences
- Buffer only stores 10% (most recent)
- Early episode experiences lost

**Impact**: Learning unstable, catastrophic forgetting

### Evaluation Bias
**Issue**: Single evaluation episode per model
- High variance in single-run evaluation
- Better: Average over 10+ evaluation episodes

**Mitigation**: 10 independent training runs for statistical validity

### Simulation Determinism
**Benefit**: Reproducible (seeded runs)
**Limitation**: No stochasticity in vehicle behavior
- Real-world: Driver randomness, sensor noise, weather
- Simulation: Perfect information, deterministic dynamics

---

## 5.10 Data Availability

### Public Artifacts
- **Code**: Available in project repository
- **Configurations**: `scenarios/configs/*.json`
- **Roadnet files**: `data/roadnet*.json`
- **Results**: `results/*.json` (all experimental outputs)

### Reproducibility
**Seeds**: 42-51 (10 runs)
**CityFlow version**: 1.0
**PyTorch version**: 2.x (CPU)
**Platform**: macOS (Darwin 24.3.0)

### Replication Instructions
```bash
# Install dependencies
pip install cityflow torch scipy matplotlib

# Run experiments
python experiments/run_h1.py  # H1 baseline
python experiments/run_h2.py  # H2 MaxPressure
python experiments/run_h3.py  # H3 Multi-agent

# Results saved to results/*.json
```

---

## 5.11 Summary Statistics

### Dataset Size
| Metric | Value |
|--------|-------|
| Total episodes (all models) | 600 (60 runs × 10 seeds) |
| Total timesteps simulated | 600,000 |
| Total experiences collected | ~600,000 |
| Experiences used (buffer) | ~5,000 per model |
| Evaluation episodes | 60 |

### Traffic Statistics (Low-Variance)
| Metric | Mean | Std |
|--------|------|-----|
| Vehicle arrival rate | 200/hr | 20/hr |
| Queue length (steady state) | 15-25 vehicles | 5-10 |
| Wait time per vehicle | 30-60s | 10-20s |

### State/Action Distributions
- **State values**: [0, 1] (normalized)
- **Action distribution**: Uniform during exploration, learned during exploitation
- **Reward distribution**:
  - H1: Mean=-6.0, Std=2.0
  - H2: Mean=-5.0, Std=2.5 (after fix)
  - Fixed-Time: Mean=-4.67, Std=0 (deterministic)

---

**Last Updated**: 2025-11-30
**Experimental Suite**: H1, H2, H3 with baselines
**Total Compute**: ~10 hours (all experiments)
