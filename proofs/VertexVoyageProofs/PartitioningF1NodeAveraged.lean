import Mathlib
import VertexVoyageProofs.PartitioningF1

/-!
# Node-averaged F1 score

Formalizes the "Node-level expectation" and "Decomposing F1 score expectation" sections of
`results/E08-f1-decomposition/README.md`, in an elementary
finite probability model (a nonnegative weight function on a `Fintype` summing to `1`), since the
note works entirely with `Finset` cardinalities (`|Sᵢ|/|V|`) rather than a continuous measure.

Three corrections come out of the formalization:

1. The node-level formula `E[F₁] = S·(α + (1-α)·Σᵢ (|Sᵢ|/|V|)·p(π v = i | π u = i))` **is** correct,
   but only once the implicit assumption "`u` is drawn uniformly from `V`" is stated explicitly
   (`hmarg` below) — the note never says what `u` and `v` are random over.
2. The "Decomposing F1 score expectation" step, `E[F₁ | π u = π v] = Σᵢ (|Sᵢ|/|V|)·E[F₁ | π u = π v = i]`,
   uses the *unconditional* partition share `|Sᵢ|/|V|` as the mixing weight. The correct
   (Bayes) weight is the *conditional* probability `P(π u = i | π u = π v)`, not `P(π u = i)`;
   the two agree only when `P(π u = π v | π u = i)` is the same for every partition `i`
   (`condProb_partition_eq_share_iff` below makes this precise). The note's version is therefore
   only valid under this unstated homogeneity assumption.
3. In that same section, the denominator term `p(uv ∈ E | uv ∈ E)·p(uv ∈ E)` is *definitionally*
   just `p(uv ∈ E)` again (conditioning on an event given itself is certain), not the intended
   `p(uv ∈ Ē)`. `F1_eq_two_TP_div` gives the correct, conditioning-free identity
   `F1 = 2·P(A∩B) / (P(A)+P(B))`.
-/

open scoped Classical

noncomputable section

namespace PartitioningF1.NodeAveraged

section FiniteProbability

/-- An `ite` whose `Decidable` instance is always resolved by *directly applying*
`Classical.propDecidable`, never by typeclass search. Typeclass search would otherwise pick a
*structural* instance (e.g. `instDecidableAnd`, composed from deciding each conjunct separately)
whenever the proposition's head symbol is visible (like `P x ∧ Q x`), but fall back to
`Classical.propDecidable` applied to the *whole* proposition when it is hidden behind an abstract
variable (like a bound `P : Ω → Prop`) — two propositionally-equal but *not definitionally equal*
`Decidable` terms for the same proposition. Every `ite` in this file goes through `iteD` so that
never happens: the same proposition always gets the same, directly-applied instance. -/
def iteD (p : Prop) (a b : ℝ) : ℝ := @ite ℝ p (Classical.propDecidable p) a b

theorem iteD_pos {p : Prop} (h : p) (a b : ℝ) : iteD p a b = a := if_pos h

theorem iteD_neg {p : Prop} (h : ¬p) (a b : ℝ) : iteD p a b = b := if_neg h

variable {Ω : Type*} [Fintype Ω]

/-- Probability of an event, under the finite weight `w`. -/
def prob (w : Ω → ℝ) (P : Ω → Prop) : ℝ := ∑ x, iteD (P x) (w x) 0

/-- Unnormalized mean of `f` over an event, under the finite weight `w`. -/
def probMean (w : Ω → ℝ) (f : Ω → ℝ) (P : Ω → Prop) : ℝ := ∑ x, iteD (P x) (w x * f x) 0

/-- Conditional probability of `Q` given `R`. -/
def condProb (w : Ω → ℝ) (Q R : Ω → Prop) : ℝ := prob w (fun x => R x ∧ Q x) / prob w R

/-- Conditional mean of `f` given `R`. -/
def condMean (w : Ω → ℝ) (f : Ω → ℝ) (R : Ω → Prop) : ℝ := probMean w f R / prob w R

variable {ι : Type*} [Fintype ι]

/-- Two logically equivalent events have equal probability (proved pointwise via `if_pos`/`if_neg`,
so it never has to unify `Decidable` instances syntactically). -/
theorem prob_congr (w : Ω → ℝ) {P Q : Ω → Prop} (h : ∀ x, P x ↔ Q x) :
    prob w P = prob w Q := by
  unfold prob
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases hP : P x
  · rw [iteD_pos hP, iteD_pos ((h x).mp hP)]
  · rw [iteD_neg hP, iteD_neg (fun hQ => hP ((h x).mpr hQ))]

/-- **Law of total probability**, decomposing any event `P` over the fibers of `π`. -/
theorem prob_eq_sum_prob_inter (w : Ω → ℝ) (π : Ω → ι) (P : Ω → Prop) :
    prob w P = ∑ i : ι, prob w (fun x => P x ∧ π x = i) := by
  unfold prob
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  have key := Finset.sum_eq_single_of_mem (s := (Finset.univ : Finset ι))
    (f := fun i => iteD (P x ∧ π x = i) (w x) 0) (π x) (Finset.mem_univ _)
    (fun j _ hj => iteD_neg (fun h => hj.symm h.2) _ _)
  rw [key]
  by_cases hP : P x
  · rw [iteD_pos hP, iteD_pos ⟨hP, rfl⟩]
  · rw [iteD_neg hP, iteD_neg (fun h => hP h.1)]

/-- Same decomposition for an unnormalized mean. -/
theorem probMean_eq_sum_probMean_inter (w : Ω → ℝ) (π : Ω → ι) (f : Ω → ℝ) (P : Ω → Prop) :
    probMean w f P = ∑ i : ι, probMean w f (fun x => P x ∧ π x = i) := by
  unfold probMean
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  have key := Finset.sum_eq_single_of_mem (s := (Finset.univ : Finset ι))
    (f := fun i => iteD (P x ∧ π x = i) (w x * f x) 0) (π x) (Finset.mem_univ _)
    (fun j _ hj => iteD_neg (fun h => hj.symm h.2) _ _)
  rw [key]
  by_cases hP : P x
  · rw [iteD_pos hP, iteD_pos ⟨hP, rfl⟩]
  · rw [iteD_neg hP, iteD_neg (fun h => hP h.1)]

theorem prob_nonneg (w : Ω → ℝ) (hnonneg : ∀ x, 0 ≤ w x) (P : Ω → Prop) : 0 ≤ prob w P :=
  Finset.sum_nonneg fun x _ => by
    by_cases hP : P x
    · rw [iteD_pos hP]; exact hnonneg x
    · rw [iteD_neg hP]

theorem prob_inter_le_prob (w : Ω → ℝ) (hnonneg : ∀ x, 0 ≤ w x) (Q R : Ω → Prop) :
    prob w (fun x => R x ∧ Q x) ≤ prob w R := by
  apply Finset.sum_le_sum
  intro x _
  by_cases hR : R x <;> by_cases hQ : Q x
  · rw [iteD_pos ⟨hR, hQ⟩, iteD_pos hR]
  · rw [iteD_neg (fun h => hQ h.2), iteD_pos hR]; exact hnonneg x
  · rw [iteD_neg (fun h => hR h.1), iteD_neg hR]
  · rw [iteD_neg (fun h => hR h.1), iteD_neg hR]

/-- **Bayes' rule**, as the identity that makes it division-free: `P(R∧Q) = P(R)·P(Q|R)`, valid
even when `P(R) = 0` (both sides are then `0`, using `R ∧ Q ⊆ R` and nonnegative weights). -/
theorem prob_inter_eq_prob_mul_condProb (w : Ω → ℝ) (hnonneg : ∀ x, 0 ≤ w x) (Q R : Ω → Prop) :
    prob w (fun x => R x ∧ Q x) = prob w R * condProb w Q R := by
  unfold condProb
  by_cases hR : prob w R = 0
  · have hsub := prob_inter_le_prob w hnonneg Q R
    have hz : prob w (fun x => R x ∧ Q x) = 0 :=
      le_antisymm (hR ▸ hsub) (prob_nonneg w hnonneg _)
    simp [hz, hR]
  · field_simp

end FiniteProbability

section NodeLevel

/-!
### Node-level decomposition of `p(π u = π v)` (correction 1)

`Ω := V × V` is the joint law of the pair `(u, v)`. The note's formula needs `u`'s marginal to
be uniform over `V` (`hmarg`); nothing is assumed about `v`.
-/

variable {V : Type*} [Fintype V] {ι : Type*} [Fintype ι]

/-- The share of vertices in partition `i`, `|Sᵢ| / |V|`. -/
def partShare (π : V → ι) (i : ι) : ℝ :=
  (Finset.univ.filter (fun x => π x = i)).card / (Fintype.card V : ℝ)

theorem sum_partShare_eq_one (π : V → ι) [Nonempty V] : ∑ i : ι, partShare π i = 1 := by
  have hcard : (∑ i : ι, (Finset.univ.filter (fun x => π x = i)).card : ℕ) = Fintype.card V := by
    have h := Finset.card_eq_sum_card_fiberwise
      (s := (Finset.univ : Finset V)) (t := (Finset.univ : Finset ι)) (f := π)
      (fun x _ => Finset.mem_univ (π x))
    simpa using h.symm
  unfold partShare
  rw [← Finset.sum_div, ← Nat.cast_sum, hcard]
  have : (Fintype.card V : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  field_simp

/-- **Node-level total-probability decomposition**: if `u`'s marginal law matches uniformly
sampling a vertex (`hmarg`), then `p(π u = π v)` is the partition-share-weighted average of
`p(π v = i | π u = i)`. This is the identity behind the note's "Node-level expectation" formula. -/
theorem prob_samePart_eq_sum_partShare_mul_condProb (π : V → ι) (w : V × V → ℝ)
    (hnonneg : ∀ p, 0 ≤ w p)
    (hmarg : ∀ i : ι, prob w (fun p : V × V => π p.1 = i) = partShare π i) :
    prob w (fun p : V × V => π p.1 = π p.2)
      = ∑ i : ι, partShare π i * condProb w (fun p : V × V => π p.2 = i)
          (fun p : V × V => π p.1 = i) := by
  rw [prob_eq_sum_prob_inter w (fun p : V × V => π p.1) (fun p : V × V => π p.1 = π p.2)]
  refine Finset.sum_congr rfl fun i _ => ?_
  show prob w (fun p : V × V => π p.1 = π p.2 ∧ π p.1 = i) = _
  rw [prob_congr w (P := fun p : V × V => π p.1 = π p.2 ∧ π p.1 = i)
      (Q := fun p : V × V => π p.1 = i ∧ π p.2 = i)
      (fun p => ⟨fun ⟨h1, h2⟩ => ⟨h2, h2 ▸ h1.symm⟩, fun ⟨h1, h2⟩ => ⟨h1.trans h2.symm, h1⟩⟩)]
  rw [prob_inter_eq_prob_mul_condProb w hnonneg
    (fun p : V × V => π p.2 = i) (fun p : V × V => π p.1 = i), hmarg i]

end NodeLevel

section AlphaForm

/-- Bridges the two ways of writing the closed form: the note's node-level formula
`S·(α + (1-α)·p)` is the same function as `expectedScore S α δ = S·(1-(1-α)·δ)` evaluated at
`δ = 1 - p` (pure algebra, confirming the node-level substitution is consistent with
`PartitioningF1.expected_closed_form`). -/
theorem expectedScore_eq_alpha_form (S α p : ℝ) :
    PartitioningF1.expectedScore S α (1 - p) = S * (α + (1 - α) * p) := by
  unfold PartitioningF1.expectedScore; ring

end AlphaForm

section Decomposing

/-!
### Total expectation with the correct (Bayes) weights (correction 2)
-/

variable {Ω : Type*} [Fintype Ω] {ι : Type*} [Fintype ι]

/-- **Correct total expectation**: conditional on `P`, the mean of `f` is the mixture of the
per-fiber conditional means weighted by the *conditional* fiber probabilities `P(π = i | P)`,
not the bare fiber shares `P(π = i)`. -/
theorem condMean_eq_sum_condProb_mul_condMean [Nonempty ι] (w : Ω → ℝ) (π : Ω → ι) (f : Ω → ℝ)
    (P : Ω → Prop) (hPBi : ∀ i : ι, 0 < prob w (fun x => P x ∧ π x = i)) :
    condMean w f P
      = ∑ i : ι, condProb w (fun x => π x = i) P * condMean w f (fun x => P x ∧ π x = i) := by
  have hP : 0 < prob w P := by
    rw [prob_eq_sum_prob_inter w π P]
    exact Finset.sum_pos (fun i _ => hPBi i) Finset.univ_nonempty
  unfold condMean
  rw [probMean_eq_sum_probMean_inter w π f P, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  have ha : prob w (fun x => P x ∧ π x = i) ≠ 0 := (hPBi i).ne'
  have hb : prob w P ≠ 0 := hP.ne'
  unfold condProb
  field_simp

/-- **When the note's shortcut is valid**: using the bare partition share `P(π = i)` as the
mixing weight (as the note does) agrees with the correct Bayes weight `P(π = i | P)` for a given
`i` *iff* `P` is independent of that fiber, i.e. `P(P | π = i) = P(P)`. In particular the note's
formula `Σᵢ P(π=i)·E[f|P∧π=i]` needs this to hold for *every* `i` — a homogeneity assumption the
note never states. -/
theorem condProb_partition_eq_share_iff (w : Ω → ℝ) (π : Ω → ι) (P : Ω → Prop) (i : ι)
    (hBi : 0 < prob w (fun x => π x = i)) (hP : 0 < prob w P) :
    condProb w (fun x => π x = i) P = prob w (fun x => π x = i)
      ↔ condProb w P (fun x => π x = i) = prob w P := by
  unfold condProb
  rw [show prob w (fun x => (fun x => π x = i) x ∧ P x) = prob w (fun x => P x ∧ π x = i) from
    prob_congr w (fun x => and_comm), div_eq_iff hP.ne', div_eq_iff hBi.ne']
  constructor <;> intro h <;> nlinarith [h]

/-! ### Micro- vs macro-averaging across partitions -/

/-- `1` on `A`, `0` off it — the bridge between an event's probability and the mean of a
real-valued score. -/
def indicator (A : Ω → Prop) : Ω → ℝ := fun x => iteD (A x) 1 0

theorem probMean_indicator_eq_prob (w : Ω → ℝ) (A R : Ω → Prop) :
    probMean w (indicator A) R = prob w (fun x => R x ∧ A x) := by
  unfold probMean prob indicator
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases hR : R x
  · rw [iteD_pos hR]
    by_cases hA : A x
    · rw [iteD_pos hA, mul_one, iteD_pos (⟨hR, hA⟩ : R x ∧ A x)]
    · rw [iteD_neg hA, mul_zero, iteD_neg (fun h : R x ∧ A x => hA h.2)]
  · rw [iteD_neg hR, iteD_neg (fun h : R x ∧ A x => hR h.1)]

theorem condMean_indicator_eq_condProb (w : Ω → ℝ) (A R : Ω → Prop) :
    condMean w (indicator A) R = condProb w A R := by
  unfold condMean condProb; rw [probMean_indicator_eq_prob]

/-- Applying the total-expectation theorem to the indicator of an event turns it into a
decomposition of that event's *probability* — the same law, just specialized. -/
theorem condProb_eq_sum_condProb_mul_condProb [Nonempty ι] (w : Ω → ℝ) (π : Ω → ι)
    (P A : Ω → Prop) (hPBi : ∀ i : ι, 0 < prob w (fun x => P x ∧ π x = i)) :
    condProb w A P
      = ∑ i : ι, condProb w (fun x => π x = i) P * condProb w A (fun x => P x ∧ π x = i) := by
  rw [← condMean_indicator_eq_condProb w A P,
    condMean_eq_sum_condProb_mul_condMean w π (indicator A) P hPBi]
  exact Finset.sum_congr rfl fun i _ => by rw [condMean_indicator_eq_condProb]

/-- **Micro-averaging is exact; macro-averaging is not.** `TP`, `P(A|P)` and `P(B|P)` each
decompose exactly into a `condProb`-weighted sum over partitions (three uses of
`condProb_eq_sum_condProb_mul_condProb`, applied to `A∧B`, `A`, `B`). Pooling those weighted
numerator and denominator sums *before* dividing (micro-averaging, the right-hand side here)
recovers `F1(P)` exactly. Dividing within each partition first and *then* taking the weighted
average of the resulting ratios (macro-averaging) does not — that is exactly what
`F1_does_not_distribute_over_mixture` refutes. -/
theorem F1_micro_average [Nonempty ι] (w : Ω → ℝ) (π : Ω → ι) (P A B : Ω → Prop)
    (hPBi : ∀ i : ι, 0 < prob w (fun x => P x ∧ π x = i)) :
    2 * condProb w (fun x => A x ∧ B x) P / (condProb w A P + condProb w B P)
      = (2 * ∑ i : ι, condProb w (fun x => π x = i) P
            * condProb w (fun x => A x ∧ B x) (fun x => P x ∧ π x = i))
        / ∑ i : ι, condProb w (fun x => π x = i) P
            * (condProb w A (fun x => P x ∧ π x = i) + condProb w B (fun x => P x ∧ π x = i)) := by
  rw [condProb_eq_sum_condProb_mul_condProb w π P (fun x => A x ∧ B x) hPBi,
    condProb_eq_sum_condProb_mul_condProb w π P A hPBi,
    condProb_eq_sum_condProb_mul_condProb w π P B hPBi]
  congr 1
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

end Decomposing

section F1Identity

/-!
### The F1 identity, done correctly (correction 3)

`A := uv ∈ E` (true edge), `B := uv ∈ Ē` (predicted edge). `F1 = 2TP/(2TP+FP+FN)` simplifies,
*without any conditional-probability rewriting*, to `2·P(A∩B) / (P(A)+P(B))` — the harmonic mean
of precision `P(A|B)` and recall `P(B|A)`. The note's attempt to reach this via `p(A|A)·P(A)` in
the denominator is flagged separately: that term is just `P(A)` again (`condProb_self` below),
not `P(B)`, so it cannot stand in for `P(A) + P(B)`.
-/

variable {Ω : Type*} [Fintype Ω]

/-- Conditioning an event on itself is certain: `p(A | A) = 1` whenever `P(A) > 0`. So the note's
denominator term `p(A|A)·P(A)` is definitionally just `P(A)` — it can never substitute for
`P(B)` unless `P(A) = P(B)`, which is not assumed anywhere in the note. -/
theorem condProb_self (w : Ω → ℝ) (A : Ω → Prop) (hA : 0 < prob w A) : condProb w A A = 1 := by
  unfold condProb
  have : (fun x => A x ∧ A x) = A := by funext x; exact propext and_self_iff
  rw [this]
  exact div_self hA.ne'

theorem condProb_self_mul_prob_eq_prob (w : Ω → ℝ) (A : Ω → Prop) (hA : 0 < prob w A) :
    condProb w A A * prob w A = prob w A := by rw [condProb_self w A hA]; ring

/-- **The correct F1 identity**: with `TP = P(A∩B)`, `FP = P(B) - TP`, `FN = P(A) - TP`,
`2·TP + FP + FN = P(A) + P(B)` exactly, so `F1 = 2·TP/(P(A)+P(B))` needs no conditional
probability at all. -/
theorem two_TP_add_FP_add_FN_eq (w : Ω → ℝ) (A B : Ω → Prop) :
    2 * prob w (fun x => A x ∧ B x) + (prob w B - prob w (fun x => A x ∧ B x))
        + (prob w A - prob w (fun x => A x ∧ B x))
      = prob w A + prob w B := by ring

/-- The same identity holds *conditionally* on any `R` — a fixed partition `πu=πv=i` — since it
is a pure algebraic identity in the three reals `TP`, `P(A)`, `P(B)`, whatever they mean. So the
correct per-partition F1 formula must condition *every* probability on the same `R`; conditioning
only some of them (as a naive transcription might) is inconsistent. -/
theorem condProb_two_TP_add_FP_add_FN_eq (w : Ω → ℝ) (A B R : Ω → Prop) :
    2 * condProb w (fun x => A x ∧ B x) R + (condProb w B R - condProb w (fun x => A x ∧ B x) R)
        + (condProb w A R - condProb w (fun x => A x ∧ B x) R)
      = condProb w A R + condProb w B R := by ring

/-- **F1 does not distribute over the Bayes mixture across partitions.** `TP(Same)`, `P(A|Same)`,
`P(B|Same)` each decompose *exactly* as a `condProb`-weighted sum of their per-partition values
(three instances of `condMean_eq_sum_condProb_mul_condMean`, applied to the indicators of `A`, `B`,
`A∧B`). But `F1 := 2·TP/(P(A)+P(B))` is a *ratio*, and a ratio of weighted sums is not the weighted
sum of the ratios (the mediant inequality) — so "`F1(Same) = Σᵢ wᵢ·F1ᵢ`" is *not* a further exact
identity, even once the weights and the per-partition conditioning are both fixed. Concretely, with
equal weights `w₁=w₂=1/2` and two partitions realizing `(TP,P(A),P(B)) = (1/2,1/2,1/2)` (perfect
reconstruction, `F1₁=1`) and `(1/20,1/10,1/10)` (`F1₂=1/2`): the exact mixture gives
`F1(Same) = 11/12`, not the naively mixed `Σᵢwᵢ·F1ᵢ = 3/4`. -/
theorem F1_does_not_distribute_over_mixture :
    (2 * ((1/2:ℝ) * (1/2) + (1/2:ℝ) * (1/20))) / ((1/2:ℝ) * 1 + (1/2:ℝ) * (1/5))
      ≠ (1/2:ℝ) * (2 * (1/2) / 1) + (1/2:ℝ) * (2 * (1/20) / (1/5)) := by norm_num

end F1Identity

section MacroAveraging

/-!
### Macro-averaging across vertices (the "Global scores" line, correction 4)

`vertex_voyage/reconstruction.py`'s `get_f1_score` — the codebase's node-averaged "Global scores"
log line — computes something different from every `F1` in this file so far. For each vertex `u`
it forms `recall(u) = TP(u)/A(u)` (`0` if `A(u) = 0`) and `precision(u) = TP(u)/B(u)` (excluded,
not zeroed, if `B(u) = 0`), takes the *plain arithmetic mean* of each across vertices, and only
then takes *one* harmonic mean of the two resulting numbers. That is: divide first (per vertex),
average second, combine third — the reverse order from `F1_micro_average`, which sums first and
divides once at the very end. Concretely, for two vertices with `(TP,A,B) = (2,2,2)` (perfect,
`recall=precision=1`) and `(1,10,1)` (`recall=1/10`, `precision=1`): macro recall `= 11/20`, macro
precision `= 1`, so macro `F1 = 22/31`, whereas the micro/pooled `F1 = 2·3/(12+3) = 2/5`. This
happens with a *single* vertex set and *no* partitioning at all — it is a strictly more basic
failure than `F1_does_not_distribute_over_mixture`, which needs at least two partitions to exhibit
the mediant gap; here two vertices already suffice, because macro-averaging never pools raw counts
across vertices in the first place. -/
theorem macroF1_ne_microF1 :
    (2 * ((((2:ℝ)/2 + 1/1)/2) * (((2:ℝ)/2 + 1/10)/2)))
        / ((((2:ℝ)/2 + 1/1)/2) + (((2:ℝ)/2 + 1/10)/2))
      ≠ 2 * ((2:ℝ) + 1) / ((2 + 10) + (2 + 1)) := by norm_num

end MacroAveraging

section MacroTrend

/-!
### When macro-recall can rise as the partition count grows

The CITESEER/AstroPh sweep in `note.md` shows `recall_cross(u) = 0` for *every* vertex, at every
partition count tried (no cross-partition edge is ever reconstructed — embeddings trained
per-partition never place a node close to one in a different partition's embedding). Under that
hypothesis, `recall(u) = p(same|E,u) * recall_within(u)` exactly (the `diff` term of the
`Decomposing` section's total-probability split vanishes, leaving only the `same` term). This
section gives the general, `k`-free tool needed to explain the two datasets' opposite trends as
`k` grows: a *product* of a shrinking factor (`p`, the retention share — more partitions cut more
of a vertex's edges) and a factor that need not shrink (`recall_within`, reconstruction quality on
the *surviving* edges) does not decrease, from one `k` to the next, whenever the second factor's
relative growth outpaces the first factor's relative shrinkage — and pointwise dominance at every
vertex propagates to the macro-averaged mean by plain monotonicity of the sum. -/

/-- **Monotonicity of the unnormalized conditional mean**: if `f ≤ g` pointwise on `R`, then so is
their `R`-mean, for any nonnegative weight `w`. The order-comparison half of what turns a
*pointwise* per-vertex inequality into an inequality of the *macro-averaged* (or any `w`-weighted)
statistic. -/
theorem probMean_mono {Ω : Type*} [Fintype Ω] (w : Ω → ℝ) (hnonneg : ∀ x, 0 ≤ w x)
    (f g : Ω → ℝ) (R : Ω → Prop) (h : ∀ x, R x → f x ≤ g x) :
    probMean w f R ≤ probMean w g R := by
  unfold probMean
  refine Finset.sum_le_sum fun x _ => ?_
  by_cases hR : R x
  · rw [iteD_pos hR, iteD_pos hR]
    exact mul_le_mul_of_nonneg_left (h x hR) (hnonneg x)
  · rw [iteD_neg hR, iteD_neg hR]

/-- Same monotonicity for the normalized conditional mean (dividing by the same positive `R`-mass
on both sides preserves the inequality). Taking `w` uniform (`w x = 1 / Fintype.card Ω`, `R` the
always-true event) specializes this to plain macro-averaging over vertices. -/
theorem condMean_mono {Ω : Type*} [Fintype Ω] (w : Ω → ℝ) (hnonneg : ∀ x, 0 ≤ w x)
    (f g : Ω → ℝ) (R : Ω → Prop) (hR : 0 < prob w R) (h : ∀ x, R x → f x ≤ g x) :
    condMean w f R ≤ condMean w g R := by
  unfold condMean
  exact (div_le_div_iff_of_pos_right hR).mpr (probMean_mono w hnonneg f g R h)

/-- **Elasticity criterion**: a product `p * q` of two positive reals does not decrease, going to
`p' * q'`, iff `q`'s growth ratio `q'/q` dominates `p`'s *shrinkage* ratio `p/p'` — i.e. iff the
relative gain in `q` outpaces the relative loss in `p`. This is exactly the algebraic content of
"reconstruction quality on surviving edges can rise fast enough to offset more edges being cut":
apply it at `p := p(same|E,u)` (shrinks as partitions get finer), `q := recall_within(u)` (need not
shrink — see the note's density hypothesis), to see when `recall(u)` itself does not decrease. -/
theorem product_le_iff_ratio_le {p p' q q' : ℝ} (hp' : 0 < p') (hq : 0 < q) :
    p * q ≤ p' * q' ↔ p / p' ≤ q' / q := by
  rw [div_le_div_iff₀ hp' hq, mul_comm q' p']

/-- **A pool-size tailwind, independent of embedding quality.** The number of `(k+1)`-subsets of
an `(n+1)`-set containing a fixed element is `n.choose k` (drop the fixed element, choose the rest
from what remains), against `(n+1).choose (k+1)` subsets in total — so *even a uniformly random,
no-skill* choice of which `k+1` candidates to reconstruct as edges lands on a fixed true neighbor
with probability exactly `(k+1)/(n+1)`, whatever that neighbor's degree. Shrinking the pool `n+1`
(a smaller partition) raises this baseline for a fixed per-vertex budget `k+1` — a real, always-
present effect, but (per the AstroPh numbers, where `recall_within` does not rise) not by itself
sufficient to explain the observed divergence; the density-dependent embedding-quality hypothesis
above is doing the rest of the work. A renamed instance of Mathlib's `Nat.add_one_mul_choose_eq`,
stated at this note's own `n`/`k` roles for direct citability. -/
theorem count_subsets_containing_fixed_element_eq (n k : ℕ) :
    (n + 1) * n.choose k = (n + 1).choose (k + 1) * (k + 1) :=
  Nat.add_one_mul_choose_eq n k

end MacroTrend

end PartitioningF1.NodeAveraged
