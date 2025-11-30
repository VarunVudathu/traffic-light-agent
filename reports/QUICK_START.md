# Quick Start Guide

Get up and running with the traffic light DQN experiments in 5 minutes.

## ✅ Pre-Flight Checklist

Before running experiments, ensure:

```bash
# 1. Check CityFlow works
python test_h1_quick.py

# 2. Verify scenarios exist
ls scenarios/configs/*.json

# 3. Check Python packages
python -c "import torch, scipy, numpy, matplotlib; print('All packages installed!')"
```

If any fail, see [Installation](#installation) below.

---

## 🚀 Running Your First Experiment

### Option 1: Super Quick Test (2 minutes)

```bash
python test_h1_quick.py
```

**What it does**: Tests that H1-Basic and Fixed-Time baseline work correctly.

**Expected output**:
```
Testing Fixed-Time baseline...
✅ Fixed-Time works! Mean reward: -0.69

Testing H1-Basic...
✅ H1-Basic works! Final reward: -15.21

All models working correctly! ✅
```

---

### Option 2: H1 Quick Validation (15 minutes)

```bash
python experiments/run_h1_quick.py
```

**What it does**:
- Runs 3 training runs for H1-Basic and H1-Enhanced
- 30 episodes per run
- Tests on low-variance and extreme surge scenarios
- Outputs statistical comparison

**Expected output**:
```
Experiment 1: H1-Basic vs Fixed-Time (Low Variance)
  H1-Basic:   -7653.25 ± 955.15
  Fixed-Time: -4.67 ± 0.00
  t=-11.325, p=0.0077, d=-6.538
  Result: Fixed-Time ✅

Experiment 2: H1-Enhanced vs H1-Basic (Extreme Surge)
  H1-Enhanced: -9633.70 ± 1595.05
  H1-Basic:    -8343.25 ± 1213.70
  t=-1.211, p=0.3495, d=-0.699
  Result: No significant difference ⚠️
```

---

### Option 3: Full H1 Experiments (30-60 minutes)

```bash
python experiments/run_h1.py
```

**What it does**:
- 10 training runs per configuration
- 50 episodes per run
- Comprehensive statistical analysis
- Generates plots

**Outputs**:
- `results/h1_results.json` - Statistical results
- `results/plots/h1_comparison.png` - Visualization

---

## 📝 Installation (If Needed)

### CityFlow Path Issue?

If you see `ModuleNotFoundError: No module named 'cityflow'`:

**Fix**: Edit the path in model files to point to your CityFlow build:

```python
# In models/h1_basic.py (and other model files)
sys.path.insert(0, "/path/to/your/CityFlow/build")
```

### Missing Python Packages?

```bash
pip install torch numpy scipy matplotlib pandas
```

---

## 🧪 Testing Individual Models

### Train H1-Basic

```python
from models import H1BasicAgent, CityFlowEnv

env = CityFlowEnv("scenarios/configs/config_low_variance.json")
agent = H1BasicAgent(env)
rewards = agent.train(episodes=30)
print(f"Final reward: {rewards[-1]:.2f}")
```

### Train H1-Enhanced

```python
from models import H1EnhancedAgent, CityFlowEnvEnhanced

env = CityFlowEnvEnhanced("scenarios/configs/config_extreme_surge.json")
agent = H1EnhancedAgent(env)
rewards = agent.train(episodes=30)
print(f"Final reward: {rewards[-1]:.2f}")
```

### Train H2-MaxPressure

```python
from models import H2MaxPressureAgent, CityFlowEnvMaxPressure

env = CityFlowEnvMaxPressure("scenarios/configs/config_low_variance.json")
agent = H2MaxPressureAgent(env)
rewards = agent.train(episodes=30)
print(f"Final reward: {rewards[-1]:.2f}")
```

### Train H3-Multi-Agent

```python
from models import H3MultiAgent, MultiIntersectionEnv

# Independent mode (no coordination)
env = MultiIntersectionEnv("scenarios/configs/config_low_variance.json",
                           coordination_mode='independent')
agent = H3MultiAgent(env)
rewards = agent.train(episodes=30)
print(f"System reward: {rewards[-1]:.2f}")

# Shared-phase mode (with coordination)
env = MultiIntersectionEnv("scenarios/configs/config_low_variance.json",
                           coordination_mode='shared_phase')
agent = H3MultiAgent(env)
rewards = agent.train(episodes=30)
print(f"System reward: {rewards[-1]:.2f}")
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Sanity Check

**Goal**: Verify everything works before long experiments

```bash
# 1. Test models work
python test_h1_quick.py

# 2. Quick validation
python experiments/run_h1_quick.py

# 3. If successful, proceed to full experiments
```

**Time**: ~15 minutes

---

### Workflow 2: Full H1 Validation

**Goal**: Get complete H1 results with statistics

```bash
# 1. Run full H1 experiments
python experiments/run_h1.py

# 2. Check results
cat results/h1_results.json

# 3. View plots
open results/plots/h1_comparison.png  # macOS
# OR
xdg-open results/plots/h1_comparison.png  # Linux
```

**Time**: ~60 minutes

---

### Workflow 3: Custom Training

**Goal**: Train a specific model with custom parameters

```python
from models import H1BasicAgent, CityFlowEnv

# Create environment
env = CityFlowEnv("scenarios/configs/config_extreme_surge.json",
                   frame_skip=1,
                   max_steps=1000)

# Create agent
agent = H1BasicAgent(env, learning_rate=5e-4)

# Train with custom hyperparameters
rewards = agent.train(
    episodes=100,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    decay=0.99,
    batch_size=256
)

# Evaluate
eval_rewards = agent.evaluate(episodes=10)

# Save model
agent.save("my_model.pth")
```

---

## 🔍 Inspecting Results

### View JSON Results

```bash
# Pretty print results
python -m json.tool results/h1_results.json
```

### Plot Training Curves

```python
import matplotlib.pyplot as plt
import json

# Load results
with open('results/h1_results.json') as f:
    results = json.load(f)

# Get H1-Basic means
h1_means = results['experiment_1']['h1_basic_means']

# Plot
plt.plot(h1_means)
plt.xlabel('Run')
plt.ylabel('Mean Reward')
plt.title('H1-Basic Performance Across Runs')
plt.show()
```

---

## 🐛 Troubleshooting

### Problem: Training is stuck / very slow

**Solutions**:
1. Reduce `max_steps`: `env = CityFlowEnv(config, max_steps=500)`
2. Reduce `episodes`: `agent.train(episodes=20)`
3. Use fewer runs: Edit experiment scripts to use `num_runs=3`

### Problem: Rewards are very negative

**This is normal!** Rewards are negative because:
- Reward = -mean_waiting_vehicles
- More waiting = more negative reward
- Goal is to maximize (make less negative)

Example: -4.67 is better than -7653.25

### Problem: Models don't learn / rewards don't improve

**Possible causes**:
1. **Too few episodes**: Try 100+ episodes for better learning
2. **Hyperparameters**: Try lower learning rate (5e-4)
3. **Exploration**: Increase epsilon_end to 0.2 for more exploration
4. **Scenario too hard**: Start with `config_low_variance.json`

---

## 📊 Understanding Output

### Training Output

```
[H1-Basic] Ep 10/30: Reward=-1177.41, Avg(10)=-1348.36, ε=0.951
```

- **Ep 10/30**: Episode 10 out of 30
- **Reward=-1177.41**: Total episode reward
- **Avg(10)=-1348.36**: Average reward of last 10 episodes
- **ε=0.951**: Current exploration rate (epsilon)

### Statistical Output

```
t=-11.325, p=0.0077, d=-6.538
Result: Fixed-Time ✅
```

- **t=-11.325**: t-statistic (measures difference)
- **p=0.0077**: p-value (probability of chance, < 0.05 = significant)
- **d=-6.538**: Cohen's d (effect size, 0.8+ = large)
- **Result**: Winner if p < 0.05

---

## 🎓 Next Steps

After running experiments:

1. **Analyze Results**: Check `results/h1_results.json`
2. **View Plots**: Look at `results/plots/*.png`
3. **Tune Models**: Adjust hyperparameters based on results
4. **Run H2/H3**: Implement experiment runners for other hypotheses
5. **Write Report**: Summarize findings with statistical evidence

---

## 💡 Tips

- **Start small**: Test with 10-20 episodes first
- **Use quick validation**: Run `run_h1_quick.py` before full experiments
- **Monitor progress**: Watch console output during training
- **Save models**: Use `agent.save()` to keep trained models
- **Compare scenarios**: Try same model on different traffic patterns

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Check `IMPLEMENTATION_PLAN.md` for design decisions
3. Check `IMPLEMENTATION_SUMMARY.md` for what was built
4. Look at code comments in model files

---

**Quick Command Reference**:

```bash
# Test everything works
python test_h1_quick.py

# Quick validation (15 min)
python experiments/run_h1_quick.py

# Full H1 experiments (60 min)
python experiments/run_h1.py

# Generate scenarios
python scenarios/generate_variance.py
python scenarios/generate_surge.py

# Validate scenarios
python scenarios/validate_scenarios.py
```

---

**Last Updated**: 2025-11-29
