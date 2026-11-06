"""Dynamic pricing with contextual bandits on a simulated logistics marketplace.

Problem: each (zone, hour-type) context has an unknown best price multiplier. A classic A/B test spends weeks
splitting traffic evenly across prices, losing money on bad arms, then commits. A bandit shifts traffic toward
good prices while it is still learning.

Policies compared on the same demand stream (same random seed), scored against an oracle that knows the truth:
  - Fixed price 1.0 (status quo)
  - Explore-then-commit (A/B test for the first 20% of rounds, then the winner per context)
  - Epsilon-greedy (10% exploration)
  - UCB1 per context
  - Thompson sampling per context (Gaussian posterior on reward)
  - LinTS: linear Thompson sampling sharing information across contexts through features

Reward per round = contribution from one quote: P(book | price) x P(fulfil | price, context) x fare x price x take rate.
"""
import json
import numpy as np

ARMS = np.array([0.85, 0.95, 1.05, 1.15, 1.25])
TAKE = 0.18
FARE = 400


class Market:
    """Ground truth. Contexts = zones x {off-peak, peak}. Elasticity and supply tightness vary by context."""

    def __init__(self, n_zones=12, seed=0):
        r = np.random.default_rng(seed)
        self.n_ctx = n_zones * 2
        self.peak = np.tile([0, 1], n_zones)
        self.tight = np.repeat(r.uniform(-.5, .5, n_zones), 2) + .9 * self.peak          # supply tightness
        self.elast = np.where(self.peak, r.uniform(-.9, -.5, self.n_ctx), r.uniform(-2.0, -1.3, self.n_ctx))
        self.rng = np.random.default_rng(seed + 100)

    def p_book(self, c, a):
        return np.clip(.45 * ARMS[a] ** self.elast[c], 0, 1)

    def p_fulfil(self, c, a):
        return 1 / (1 + np.exp(-(1.3 - 1.4 * self.tight[c] + 3.0 * (ARMS[a] - 1))))

    def expected(self, c, a):
        return self.p_book(c, a) * self.p_fulfil(c, a) * FARE * ARMS[a] * TAKE

    def oracle(self):
        return np.array([[self.expected(c, a) for a in range(len(ARMS))] for c in range(self.n_ctx)])

    def pull(self, c, a):
        booked = self.rng.random() < self.p_book(c, a)
        done = booked and self.rng.random() < self.p_fulfil(c, a)
        return FARE * ARMS[a] * TAKE * done

    def features(self, c, a):
        p = ARMS[a] - 1
        return np.array([1, p, p * p, self.peak[c], self.peak[c] * p, self.tight[c], self.tight[c] * p])


# ---------------- policies ----------------
class Fixed:
    name = "Fixed price 1.0"
    def __init__(self, m, T, rng): pass
    def choose(self, c, t): return 2
    def update(self, c, a, r): pass


class ExploreCommit:
    name = "A/B test then commit (20%)"
    def __init__(self, m, T, rng):
        self.T0, self.rng = int(.2 * T), rng
        self.s = np.zeros((m.n_ctx, len(ARMS))); self.n = np.zeros_like(self.s)
    def choose(self, c, t):
        if t < self.T0:
            return self.rng.integers(len(ARMS))
        return int(np.argmax(self.s[c] / np.maximum(self.n[c], 1)))
    def update(self, c, a, r):
        self.s[c, a] += r; self.n[c, a] += 1


class EpsGreedy(ExploreCommit):
    name = "Epsilon-greedy (10%)"
    def choose(self, c, t):
        if self.rng.random() < .1 or self.n[c].min() == 0:
            return self.rng.integers(len(ARMS))
        return int(np.argmax(self.s[c] / self.n[c]))


class UCB(ExploreCommit):
    name = "UCB1"
    def choose(self, c, t):
        if self.n[c].min() == 0:
            return int(np.argmin(self.n[c]))
        mean = self.s[c] / self.n[c]
        bonus = FARE * TAKE * 1.25 * np.sqrt(2 * np.log(self.n[c].sum()) / self.n[c])
        return int(np.argmax(mean + bonus))


class Thompson(ExploreCommit):
    name = "Thompson sampling"
    def __init__(self, m, T, rng):
        super().__init__(m, T, rng); self.ss = np.zeros_like(self.s)
    def choose(self, c, t):
        n = self.n[c]
        mean = np.where(n > 0, self.s[c] / np.maximum(n, 1), 40.0)
        var = np.where(n > 1, (self.ss[c] / np.maximum(n, 1) - mean ** 2).clip(1), 60.0 ** 2)
        draw = self.rng.normal(mean, np.sqrt(var / np.maximum(n, 1)))
        return int(np.argmax(draw))
    def update(self, c, a, r):
        super().update(c, a, r); self.ss[c, a] += r * r


class LinTS:
    """Bayesian linear regression on (context, price) features: learns one model for all contexts."""
    name = "Linear Thompson sampling (shared model)"
    def __init__(self, m, T, rng, lam=1.0, noise=60.0):
        self.m, self.rng, self.noise = m, rng, noise
        d = len(m.features(0, 0))
        self.A = lam * np.eye(d); self.b = np.zeros(d)
        self.X = {c: np.array([m.features(c, a) for a in range(len(ARMS))]) for c in range(m.n_ctx)}
    def choose(self, c, t):
        A_inv = np.linalg.inv(self.A)
        theta = self.rng.multivariate_normal(A_inv @ self.b, self.noise ** 2 * A_inv)
        return int(np.argmax(self.X[c] @ theta))
    def update(self, c, a, r):
        x = self.X[c][a]; self.A += np.outer(x, x); self.b += r * x


POLICIES = [Fixed, ExploreCommit, EpsGreedy, UCB, Thompson, LinTS]


def run(policy_cls, T=60000, seed=0):
    m = Market(seed=seed)
    opt = m.oracle()
    best = opt.max(axis=1)
    rng = np.random.default_rng(seed + 7)
    ctx_stream = np.random.default_rng(seed + 9).integers(0, m.n_ctx, T)
    pol = policy_cls(m, T, rng)
    regret = np.zeros(T); reward = 0.0
    for t, c in enumerate(ctx_stream):
        a = pol.choose(c, t)
        r = m.pull(c, a)
        pol.update(c, a, r)
        reward += r
        regret[t] = best[c] - opt[c, a]
    final_correct = np.mean([pol.choose(c, T) == np.argmax(opt[c]) for c in range(m.n_ctx)]) if policy_cls is not Fixed else \
        np.mean(np.argmax(opt, 1) == 2)
    return dict(policy=pol.name, cum_regret=float(regret.sum()), reward=reward,
                regret_last_10pct=float(regret[-T // 10:].mean()), final_best_arm_share=float(final_correct),
                oracle_reward=float(best[ctx_stream].sum()))


def main(T=60000, seeds=(0, 1, 2)):
    rows = {}
    for P in POLICIES:
        res = [run(P, T, s) for s in seeds]
        rows[P.name] = dict(
            pct_of_oracle_contribution=round(100 * np.mean([r["reward"] / r["oracle_reward"] for r in res]), 1),
            cumulative_regret=round(float(np.mean([r["cum_regret"] for r in res]))),
            regret_per_quote_last_10pct=round(float(np.mean([r["regret_last_10pct"] for r in res])), 2),
            contexts_on_best_price_at_end=round(100 * float(np.mean([r["final_best_arm_share"] for r in res])), 1))
    m = Market()
    opt = m.oracle()
    summary = dict(rounds=T, seeds=len(seeds), contexts=m.n_ctx, arms=ARMS.tolist(),
                   oracle_best_price_by_context_type=dict(
                       offpeak=[float(ARMS[i]) for i in np.argmax(opt, 1)[m.peak == 0]],
                       peak=[float(ARMS[i]) for i in np.argmax(opt, 1)[m.peak == 1]]),
                   policies=rows)
    base = rows["A/B test then commit (20%)"]["cumulative_regret"]
    for k in ("Thompson sampling", "Linear Thompson sampling (shared model)"):
        summary[f"regret_cut_vs_ab_{k}"] = round(100 * (1 - rows[k]["cumulative_regret"] / base), 1)
    with open("results.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
