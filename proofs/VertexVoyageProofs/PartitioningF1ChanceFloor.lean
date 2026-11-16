import VertexVoyageProofs.PartitioningF1NodeAveraged

/-!
# The chance floor of a fixed-budget reconstruction

Formalizes the "Erdős–Rényi sweep" section of
`results/E08-f1-decomposition/README.md`.

`reconstruct(k, embedding, nodes)` in `vertex_voyage/reconstruction.py` emits the `k` globally
closest embedding pairs, and the caller passes `k := pg.number_of_edges()`. So the reconstruction
budget is a *global* pair budget, not the per-vertex budget assumed by
`PartitioningF1.NodeAveraged.count_subsets_containing_fixed_element_eq`. Under a global budget the
no-skill baseline for within-partition recall is the *density of the scope being reconstructed*,
and the quantity that measures reconstruction skill is the lift of the score over that baseline,
not the score itself.

Three facts come out of this:

1. `fixed_element_subset_fraction`: a uniformly random `budget`-subset of a `pairs`-element pair
   set contains any fixed pair with probability exactly `budget / pairs` — so the chance floor of
   recall, precision and F1 alike is the density `|E| / C(|V|, 2)` of the reconstructed scope.
2. `score_ratio_factors`: a raw score ratio `Q'/Q` between two partition counts factors into a
   *skill* ratio and a *chance-floor* ratio. A raw ratio above `1` therefore does not by itself
   witness any gain in reconstruction quality.
3. `ratio_le_inv_of_le_one` / `ratio_eq_inv_iff`: since a recall is at most `1`, the raw ratio is
   capped by `1 / Q`, with equality exactly at saturation `Q' = 1`. A saturated `Q'` makes the
   ratio a function of `Q` alone, carrying no information about the intervention that produced it.

A fourth fact covers the degenerate cells of the sweep: `label_propagation_partitioner` bin-packs
the communities `asyn_lpa_communities` finds, so when label propagation returns a single community
every requested partition count collapses to one non-empty block.
`cut_fraction_eq_zero_of_constant_partition` and `condMean_within_eq_total_of_constant_partition`
say such a cell measures the unpartitioned pipeline exactly.
-/

open scoped Classical

noncomputable section

namespace PartitioningF1.ChanceFloor

open PartitioningF1.NodeAveraged

section GlobalBudget

/-- The no-skill baseline of a reconstruction that emits `budget` of the `pairs` candidate pairs:
the fraction of the candidate pairs it is allowed to name. -/
def chanceFloor (budget pairs : ℕ) : ℝ := (budget : ℝ) / (pairs : ℝ)

/-- Edge density of a graph with `edges` edges on `vertices` vertices. -/
def density (edges vertices : ℕ) : ℝ := (edges : ℝ) / (vertices.choose 2 : ℝ)

/-- **A uniformly random global budget hits a fixed pair with probability `budget / pairs`.** The
`(k+1)`-subsets of an `(n+1)`-element set containing a fixed element number `n.choose k`, against
`(n+1).choose (k+1)` subsets in total, and the quotient is exactly `(k+1)/(n+1)`. Unlike
`count_subsets_containing_fixed_element_eq`, which is stated for a per-vertex budget among a pool
of candidate *neighbours*, here the pool is the set of candidate *pairs* and the budget is global,
matching `reconstruct`. -/
theorem fixed_element_subset_fraction (n k : ℕ) (hk : k ≤ n) :
    (n.choose k : ℝ) / ((n + 1).choose (k + 1) : ℝ) = ((k : ℝ) + 1) / ((n : ℝ) + 1) := by
  have hchoose : (0 : ℕ) < (n + 1).choose (k + 1) := Nat.choose_pos (by omega)
  have hden : ((n + 1).choose (k + 1) : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hchoose.ne'
  have hn : ((n : ℝ) + 1) ≠ 0 := by positivity
  rw [div_eq_div_iff hden hn]
  have hnat : (n + 1) * n.choose k = (n + 1).choose (k + 1) * (k + 1) :=
    Nat.add_one_mul_choose_eq n k
  have hcast : ((n : ℝ) + 1) * (n.choose k : ℝ)
      = ((n + 1).choose (k + 1) : ℝ) * ((k : ℝ) + 1) := by exact_mod_cast hnat
  linear_combination hcast

/-- The chance floor of the whole-graph reconstruction, whose budget is the graph's own edge
count, is the graph's density. -/
theorem chanceFloor_eq_density (edges vertices : ℕ) :
    chanceFloor edges (vertices.choose 2) = density edges vertices := rfl

end GlobalBudget

section Lift

/-- Reconstruction skill: how many times the chance floor a score achieves. -/
def lift (score floor : ℝ) : ℝ := score / floor

theorem lift_mul_floor_eq_score (score floor : ℝ) (hfloor : floor ≠ 0) :
    lift score floor * floor = score := div_mul_cancel₀ score hfloor

/-- **A raw score ratio factors into a skill ratio and a chance-floor ratio.** Comparing a score
at two partition counts conflates the two: a raw ratio above `1` is consistent with skill having
*fallen*, provided the chance floor rose faster. In the Erdős–Rényi sweep this is what happens —
label propagation's cut raises the within-partition density, so the floor rises by more than the
raw score does. -/
theorem score_ratio_factors {score score' floor floor' : ℝ}
    (hscore : score ≠ 0) (hfloor : floor ≠ 0) (hfloor' : floor' ≠ 0) :
    score' / score = lift score' floor' / lift score floor * (floor' / floor) := by
  unfold lift
  field_simp

/-- **The raw ratio is capped by the reciprocal of the baseline score.** Any recall is at most
`1`, so `Q' / Q ≤ 1 / Q`. -/
theorem ratio_le_inv_of_le_one {score score' : ℝ} (hscore : 0 < score) (hle : score' ≤ 1) :
    score' / score ≤ 1 / score := by gcongr

/-- **At saturation the raw ratio is a function of the baseline alone.** `Q' / Q = 1 / Q` exactly
when `Q' = 1`, so a sweep whose scores saturate reports `1 / Q(1)` whatever the intervention did,
and the ratio's variation across settings measures only the variation of `Q(1)`. -/
theorem ratio_eq_inv_iff {score score' : ℝ} (hscore : 0 < score) :
    score' / score = 1 / score ↔ score' = 1 := by
  constructor
  · intro h
    field_simp at h
    exact h
  · intro h
    rw [h]

end Lift

section CollapsedPartition

variable {Ω : Type*} [Fintype Ω]

theorem prob_eq_zero_of_never (w : Ω → ℝ) (P : Ω → Prop) (hnever : ∀ x, ¬ P x) :
    prob w P = 0 := by
  unfold prob
  refine Finset.sum_eq_zero fun x _ => ?_
  exact iteD_neg (hnever x) (w x) 0

theorem probMean_congr (w : Ω → ℝ) (f : Ω → ℝ) {P Q : Ω → Prop} (hiff : ∀ x, P x ↔ Q x) :
    probMean w f P = probMean w f Q := by
  unfold probMean
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases hP : P x
  · rw [iteD_pos hP, iteD_pos ((hiff x).1 hP)]
  · rw [iteD_neg hP, iteD_neg (fun hQ => hP ((hiff x).2 hQ))]

theorem condMean_congr (w : Ω → ℝ) (f : Ω → ℝ) {P Q : Ω → Prop} (hiff : ∀ x, P x ↔ Q x) :
    condMean w f P = condMean w f Q := by
  unfold condMean
  rw [probMean_congr w f hiff, prob_congr w hiff]

variable {V ι : Type*} [Fintype V]

/-- **A single-community partitioning has zero edge cut.** `label_propagation_partitioner` bin-packs
the communities `asyn_lpa_communities` returns into the requested number of bins, so one community
means one non-empty bin: `δ = 0` no matter how many partitions were requested. -/
theorem cut_fraction_eq_zero_of_constant_partition (w : V × V → ℝ) (π : V → ι) (c : ι)
    (hconst : ∀ v, π v = c) : prob w (fun x => π x.1 ≠ π x.2) = 0 :=
  prob_eq_zero_of_never w _ fun x => by simp [hconst x.1, hconst x.2]

/-- **Such a cell measures the unpartitioned pipeline.** Conditioning on "same partition" is
conditioning on a certain event, so every within-partition statistic equals its whole-graph
counterpart: the requested partition count has no effect to measure. -/
theorem condMean_within_eq_total_of_constant_partition (w : V × V → ℝ) (f : V × V → ℝ)
    (π : V → ι) (c : ι) (hconst : ∀ v, π v = c) :
    condMean w f (fun x => π x.1 = π x.2) = condMean w f (fun _ => True) :=
  condMean_congr w f fun x => by simp [hconst x.1, hconst x.2]

end CollapsedPartition

end PartitioningF1.ChanceFloor
