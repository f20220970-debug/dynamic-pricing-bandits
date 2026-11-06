# Dynamic Pricing with Contextual Bandits

[![tests](https://github.com/f20220970-debug/dynamic-pricing-bandits/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/dynamic-pricing-bandits/actions/workflows/tests.yml)

An A/B test finds the best price but loses money along the way: for weeks, a fifth of the traffic sits on each bad price. A **bandit** moves traffic toward better prices while it is still learning. This repo tests whether that actually pays off in a logistics marketplace, where a price changes **both** booking (customers) and fulfilment (drivers).

> **Simulated market with known ground truth:** 12 zones × peak/off-peak = 24 contexts, 5 price levels (0.85×–1.25×), 60,000 quotes, 3 random seeds. Each policy is scored against an oracle that knows the true best price in every context. No company data is used.

Run it: `pip install -r requirements.txt && python bandits.py && pytest -q` (about 20 seconds).

## Results (mean of 3 seeds)

| Policy | % of oracle contribution | Cumulative regret (INR) | Regret per quote, last 10% |
| --- | --- | --- | --- |
| Fixed price 1.0× (status quo) | 87.4% | 192,757 | 3.25 |
| A/B test for 20% of traffic, then commit | 95.4% | 72,683 | 0.67 |
| Epsilon-greedy (10%) | 95.6% | 74,204 | 0.85 |
| UCB1 | 91.0% | 137,252 | 1.99 |
| Thompson sampling, one bandit per context | 94.5% | 89,177 | 1.24 |
| **Linear Thompson sampling (one shared model)** | **98.4%** | **28,960** | **0.41** |

## What this shows

1. **The status quo leaves 12.6% of contribution on the table.** In the simulation, peak hours should be priced up in every zone (drivers are scarce, so a higher price lifts fulfilment), and off-peak prices should differ by zone.
2. **Bandits aren't automatically better.** With 24 separate contexts and noisy book-or-not rewards, a bandit per context (Thompson, UCB) does **worse** than a plain A/B test. Each context gets only about 2,500 quotes, too few to learn alone.
3. **Sharing information is what wins.** One linear model of reward over price, peak and supply tightness learns from every quote at once. It cuts cumulative regret **60% vs the A/B test** and reaches **98.4%** of the oracle.
4. **What to do in practice:** run pricing as a shared-model bandit with guardrails (price bounds, a fulfilment floor, a small always-on holdout), not as a series of separate A/B tests per city.

## Files

```
bandits.py     market simulator (ground truth) + 6 policies + evaluation
results.json   headline numbers, including the oracle's best price per context
tests/         oracle is an upper bound; LinTS beats the A/B test, which beats fixed pricing; regret >= 0
```

Companion repos: [marketplace-pricing-experiments](https://github.com/f20220970-debug/marketplace-pricing-experiments) (designing and reading a price test) and [porter-category-growth-lab](https://github.com/f20220970-debug/porter-category-growth-lab).
