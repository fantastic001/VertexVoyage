import VertexVoyageProofs.ErdosRenyi

/-!
# Partitioning strategies for Erdős–Rényi graphs

The notes point out that `G(n, p)` has "no hubs" and is a "small world": intuitively, it has no
built-in community structure, unlike e.g. a planted-partition / stochastic block model. Here we
make one aspect of that precise: for any bipartition of the vertex set into a block `A` and its
complement, the *expected* number of edges crossing the cut and the *expected* number of edges
inside each block only depend on the block sizes, not on which vertices happen to be in `A`. This
is exactly what "no community structure" means for the edge-independent Gilbert model, and it is
also the fact underlying why a *balanced* bipartition maximizes the expected cut size (relevant to
random-partition Max-Cut heuristics).
-/

open MeasureTheory ProbabilityTheory unitInterval SimpleGraph
open scoped ENNReal NNReal

noncomputable section

namespace ErdosRenyi

variable {n : ℕ} {p : I}

instance instIsProbabilityMeasure : IsProbabilityMeasure (ErdosRenyi n p) := by
  rw [ErdosRenyi]; infer_instance

lemma measurableSet_adj (x y : Fin n) :
    MeasurableSet {H : SimpleGraph (Fin n) | H.Adj x y} := by
  have hmeas : Measurable fun H : SimpleGraph (Fin n) ↦ H.Adj x y := by fun_prop
  have heq : {H : SimpleGraph (Fin n) | H.Adj x y} = (fun H ↦ H.Adj x y) ⁻¹' {True} := by
    ext H; simp
  rw [heq]
  exact hmeas (by measurability)

/-- The indicator random variable of the event "`x` and `y` are adjacent". -/
def adjIndicator (x y : Fin n) : SimpleGraph (Fin n) → ℝ :=
  {H : SimpleGraph (Fin n) | H.Adj x y}.indicator (fun _ ↦ 1)

lemma integrable_adjIndicator (x y : Fin n) :
    Integrable (adjIndicator (n := n) x y) (ErdosRenyi n p) :=
  (integrable_const 1).indicator (measurableSet_adj x y)

/-- Every potential edge contributes probability `p` in expectation, matching
`ErdosRenyi.real_apply_adj`. -/
theorem integral_adjIndicator {x y : Fin n} (hxy : x ≠ y) :
    ∫ H, adjIndicator x y H ∂(ErdosRenyi n p) = (p : ℝ) := by
  rw [adjIndicator, integral_indicator (measurableSet_adj x y), setIntegral_const, smul_eq_mul,
    mul_one]
  exact real_apply_adj p hxy

/-- **Expected crossing-edge count.** For any bipartition of the vertices into a block `A` and its
complement, the expected number of edges crossing the cut is `|A| * |Aᶜ| * p` — it depends only on
the sizes of the two sides, not on which vertices are chosen. This is the ER-graph analogue of "no
partitioning strategy reveals community structure". -/
theorem integral_crossing_count (A : Finset (Fin n)) (p : I) :
    ∫ H, ∑ x ∈ A, ∑ y ∈ Aᶜ, adjIndicator x y H ∂(ErdosRenyi n p)
      = (A.card : ℝ) * (Aᶜ.card : ℝ) * (p : ℝ) := by
  have hstep : ∀ x ∈ A, ∫ H, ∑ y ∈ Aᶜ, adjIndicator x y H ∂(ErdosRenyi n p)
      = (Aᶜ.card : ℝ) * (p : ℝ) := by
    intro x hx
    rw [integral_finsetSum Aᶜ (fun y _ ↦ integrable_adjIndicator x y)]
    have hxy : ∀ y ∈ Aᶜ, x ≠ y := by
      intro y hy hxy
      exact (Finset.mem_compl.1 hy) (hxy ▸ hx)
    rw [Finset.sum_congr rfl fun y hy ↦ integral_adjIndicator (hxy y hy), Finset.sum_const,
      nsmul_eq_mul]
  rw [integral_finsetSum A (fun x hx ↦ integrable_finsetSum Aᶜ (fun y _ ↦
    integrable_adjIndicator x y)), Finset.sum_congr rfl hstep, Finset.sum_const, nsmul_eq_mul]
  ring

/-- **Expected within-block (ordered) adjacency count.** Summing over *ordered* pairs
`x ≠ y` inside a block `A` (so every edge inside `A` is counted twice), the expected count is
`|A| * (|A| - 1) * p`, again depending only on the block size. -/
theorem integral_within_count (A : Finset (Fin n)) (p : I) :
    ∫ H, ∑ x ∈ A, ∑ y ∈ A \ {x}, adjIndicator x y H ∂(ErdosRenyi n p)
      = (A.card : ℝ) * ((A.card - 1 : ℕ) : ℝ) * (p : ℝ) := by
  have hstep : ∀ x ∈ A, ∫ H, ∑ y ∈ A \ {x}, adjIndicator x y H ∂(ErdosRenyi n p)
      = ((A.card - 1 : ℕ) : ℝ) * (p : ℝ) := by
    intro x hx
    rw [integral_finsetSum (A \ {x}) (fun y _ ↦ integrable_adjIndicator x y)]
    have hxy : ∀ y ∈ A \ {x}, x ≠ y := by
      intro y hy
      simp only [Finset.mem_sdiff, Finset.mem_singleton] at hy
      exact fun h ↦ hy.2 h.symm
    have hc : (A \ {x}).card = A.card - 1 := by
      rw [Finset.card_sdiff_of_subset (Finset.singleton_subset_iff.2 hx), Finset.card_singleton]
    rw [Finset.sum_congr rfl fun y hy ↦ integral_adjIndicator (hxy y hy), Finset.sum_const,
      nsmul_eq_mul, hc]
  rw [integral_finsetSum A (fun x hx ↦ integrable_finsetSum (A \ {x}) (fun y _ ↦
    integrable_adjIndicator x y)), Finset.sum_congr rfl hstep, Finset.sum_const, nsmul_eq_mul]
  ring

/-- **Balanced bipartitions maximize the expected cut.** As a function of the block size `k`, the
expected crossing count `k * (n - k) * p` is largest when the split is as balanced as possible:
it never exceeds `n² / 4 * p`. This is the standard reason random-partition Max-Cut heuristics on
`G(n, p)` aim for a balanced split. -/
theorem integral_crossing_count_le (A : Finset (Fin n)) (p : I) :
    ∫ H, ∑ x ∈ A, ∑ y ∈ Aᶜ, adjIndicator x y H ∂(ErdosRenyi n p) ≤ (n : ℝ) ^ 2 / 4 * (p : ℝ) := by
  rw [integral_crossing_count]
  have hcard : (Aᶜ.card : ℝ) = (n : ℝ) - (A.card : ℝ) := by
    have h := Finset.card_compl A
    have hle : A.card ≤ n := by simpa using A.card_le_univ
    rw [h, Fintype.card_fin, Nat.cast_sub hle]
  rw [hcard]
  have hp : 0 ≤ (p : ℝ) := nonneg p
  nlinarith [sq_nonneg (2 * (A.card : ℝ) - n), hp]

end ErdosRenyi
