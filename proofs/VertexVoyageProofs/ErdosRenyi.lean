import Mathlib

/-!
# Erdős–Rényi random graphs

Based on the Erdős–Rényi entry of `doc/glossary.md`.

The notes describe the *Gilbert model* `G(n, p)`: start with `n` labelled vertices and connect
every pair independently with probability `p`. Mathlib already formalizes exactly this as
`SimpleGraph.binomialRandom`, the measure `G(V, p)` on `SimpleGraph V` (Mathlib calls it the
"binomial random graph" rather than Erdős–Rényi, since Erdős and Rényi's original model,
`G(n, N)`, samples a uniformly random graph with a *fixed* number `N` of edges instead of an
independent probability per edge). We specialize it to `V = Fin n` and reprove a couple of the
basic facts stated in the notes.
-/

open MeasureTheory ProbabilityTheory unitInterval SimpleGraph
open scoped ENNReal NNReal

noncomputable section

/-- The Erdős–Rényi / Gilbert random graph on `n` labelled vertices, connecting each of the
`n.choose 2` possible edges independently with probability `p`. -/
def ErdosRenyi (n : ℕ) (p : I) : Measure (SimpleGraph (Fin n)) :=
  binomialRandom (Fin n) p

namespace ErdosRenyi

variable {n : ℕ} {p : I}

/-- With `p = 0` no edges are ever drawn: the random graph is a.s. edgeless. -/
theorem eq_dirac_bot_of_zero (n : ℕ) : ErdosRenyi n 0 = Measure.dirac ⊥ :=
  binomialRandom_zero (Fin n)

/-- With `p = 1` every edge is drawn: the random graph is a.s. complete. -/
theorem eq_dirac_top_of_one (n : ℕ) : ErdosRenyi n 1 = Measure.dirac ⊤ :=
  binomialRandom_one (Fin n)

/-- The probability of observing a specific graph `H` on `n` vertices is
`p ^ |E(H)| * (1 - p) ^ (C(n,2) - |E(H)|)`: each of the `C(n,2)` possible edges independently
either appears (probability `p`) or not (probability `1 - p`). -/
theorem real_singleton_eq (p : I) (H : SimpleGraph (Fin n)) :
    (ErdosRenyi n p).real {H} =
      (p : ℝ) ^ H.edgeSet.ncard * (1 - p : ℝ) ^ (n.choose 2 - H.edgeSet.ncard) := by
  have hcard : (Nat.card (Fin n)).choose 2 = n.choose 2 := by simp
  rw [measureReal_def, ErdosRenyi, binomialRandom_singleton, hcard, ENNReal.toReal_mul,
    ENNReal.toReal_pow, ENNReal.toReal_pow, ENNReal.coe_toReal, ENNReal.coe_toReal,
    coe_toNNReal, coe_toNNReal, coe_symm_eq]

/-- **From the notes:** "let x, y be nodes of graph, then probability of their connection is
`P(x, y ∈ E) = p`." Each individual potential edge is present with probability exactly `p`. -/
theorem real_apply_adj {x y : Fin n} (p : I) (hxy : x ≠ y) :
    (ErdosRenyi n p).real {H : SimpleGraph (Fin n) | H.Adj x y} = (p : ℝ) := by
  classical
  set e₀ : Sym2 (Fin n) := s(x, y) with he₀
  have hnotdiag : e₀ ∉ Sym2.diagSet := by simpa [he₀, Sym2.mk_isDiag_iff] using hxy
  have hmem : e₀ ∈ Sym2.diagSetᶜ := hnotdiag
  -- The image, under `edgeSet`, of the event "`x` and `y` are adjacent" is exactly the set of
  -- edge-sets that avoid the diagonal and contain `e₀`.
  have himg : edgeSet '' {H : SimpleGraph (Fin n) | H.Adj x y}
      = {t : Set (Sym2 (Fin n)) | e₀ ∈ t ∧ t ⊆ Sym2.diagSetᶜ} := by
    ext t
    simp only [Set.mem_image, Set.mem_ofPred_eq]
    constructor
    · rintro ⟨H, hH, rfl⟩
      exact ⟨(mem_edgeSet H).2 hH, edgeSet_subset_compl_diagSet H⟩
    · rintro ⟨ht0, hsub⟩
      refine ⟨fromEdgeSet t, (fromEdgeSet_adj t).2 ⟨ht0, hxy⟩, ?_⟩
      rw [edgeSet_fromEdgeSet]
      ext e
      simp only [Set.mem_sdiff]
      exact ⟨fun h ↦ h.1, fun h ↦ ⟨h, hsub h⟩⟩
  have key : ErdosRenyi n p {H : SimpleGraph (Fin n) | H.Adj x y}
      = setBer(Sym2.diagSetᶜ, p) {t | e₀ ∈ t} := by
    rw [ErdosRenyi, binomialRandom_apply', himg]
    exact (setBernoulli_apply_eq_apply_subsets Sym2.diagSetᶜ p {t | e₀ ∈ t}).symm
  have hmarg : setBer(Sym2.diagSetᶜ, p) {t | e₀ ∈ t} = (toNNReal p : ℝ≥0∞) := by
    rw [setBernoulli_eq_map, Measure.map_apply (by fun_prop) (by measurability)]
    have hpre : (fun f : Sym2 (Fin n) → Prop ↦ {i | f i}) ⁻¹' {t | e₀ ∈ t} = {f | f e₀} := by
      ext f; simp
    rw [hpre]
    have heval : {f : Sym2 (Fin n) → Prop | f e₀} = (fun f ↦ f e₀) ⁻¹' {True} := by
      ext f; simp
    rw [heval, ← Measure.map_apply (measurable_pi_apply e₀) (by measurability),
      Measure.infinitePi_map_eval]
    simp [hmem]
  rw [measureReal_def, key, hmarg, ENNReal.coe_toReal, coe_toNNReal]

end ErdosRenyi
