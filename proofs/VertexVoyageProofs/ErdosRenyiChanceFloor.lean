import VertexVoyageProofs.ErdosRenyiPartitions

/-!
# Erdős–Rényi graphs offer no density gradient to partition along

Formalizes the "Erdős–Rényi sweep" section of
`results/E08-f1-decomposition/README.md`.

`ErdosRenyiPartitions` already shows that expected crossing and within-block edge counts depend
only on block *sizes*. Here that is pushed to the two statements the experiment needs:

* `integral_within_density_eq`: the *expected density* of the subgraph induced on a block is `p`,
  for every block of at least two vertices. Since the no-skill baseline of a fixed-budget
  reconstruction is the density of the scope being reconstructed
  (`PartitioningF1.ChanceFloor.fixed_element_subset_fraction`), partitioning an Erdős–Rényi graph
  along any edge-independent cut leaves that baseline unchanged. There is no "smaller pool, easier
  problem" tailwind to collect, so the pool-size effect of
  `PartitioningF1.NodeAveraged.count_subsets_containing_fixed_element_eq` — stated for a *fixed*
  per-vertex budget — does not apply to this pipeline, whose budget shrinks with the block.
* `expected_cut_fraction_eq`: the expected edge-cut fraction of a bipartition does not depend on
  `p` at all, only on the block sizes, and a balanced bipartition cuts about half the edges. A
  partitioner that cuts *less* than that on an Erdős–Rényi graph is fitting the sample's own random
  fluctuations, not a density gradient of the model — and it pays for it by raising the
  within-block density above `p`, which raises the chance floor.
-/

open MeasureTheory ProbabilityTheory unitInterval SimpleGraph
open scoped ENNReal NNReal

noncomputable section

namespace ErdosRenyi

variable {n : ℕ} {p : I}

/-- **Expected induced density is block-independent.** Dividing the expected within-block ordered
adjacency count by the number of ordered pairs inside the block gives `p` for every block of at
least two vertices — in particular the same value for the whole vertex set and for any block of
any partition. -/
theorem integral_within_density_eq (A : Finset (Fin n)) (p : I) (hA : 2 ≤ A.card) :
    (∫ H, ∑ x ∈ A, ∑ y ∈ A \ {x}, adjIndicator x y H ∂(ErdosRenyi n p))
      / ((A.card : ℝ) * ((A.card - 1 : ℕ) : ℝ)) = (p : ℝ) := by
  have hcard : (A.card : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have hpred : ((A.card - 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  rw [integral_within_count]
  field_simp

/-- **Any two blocks look equally dense in expectation.** The scale invariance that makes an
Erdős–Rényi sweep unable to reproduce a within-partition density gradient: the chance floor of the
within-block reconstruction is `p` whichever block, and however many blocks, one reconstructs. -/
theorem integral_within_density_eq_of_two_blocks (A B : Finset (Fin n)) (p : I)
    (hA : 2 ≤ A.card) (hB : 2 ≤ B.card) :
    (∫ H, ∑ x ∈ A, ∑ y ∈ A \ {x}, adjIndicator x y H ∂(ErdosRenyi n p))
        / ((A.card : ℝ) * ((A.card - 1 : ℕ) : ℝ))
      = (∫ H, ∑ x ∈ B, ∑ y ∈ B \ {x}, adjIndicator x y H ∂(ErdosRenyi n p))
        / ((B.card : ℝ) * ((B.card - 1 : ℕ) : ℝ)) := by
  rw [integral_within_density_eq A p hA, integral_within_density_eq B p hB]

/-- **The expected cut fraction of a bipartition is free of `p`.** Expected crossing edges over
expected total edges is `2|A||Aᶜ| / (n(n-1))`, a purely combinatorial quantity: an edge-independent
cut of an Erdős–Rényi graph cuts the same *share* of edges at every density. -/
theorem expected_cut_fraction_eq (A : Finset (Fin n)) (p : I) (hp : (p : ℝ) ≠ 0) (hn : 2 ≤ n) :
    (∫ H, ∑ x ∈ A, ∑ y ∈ Aᶜ, adjIndicator x y H ∂(ErdosRenyi n p))
      / ((∫ H, ∑ x ∈ (Finset.univ : Finset (Fin n)), ∑ y ∈ Finset.univ \ {x},
            adjIndicator x y H ∂(ErdosRenyi n p)) / 2)
      = 2 * (A.card : ℝ) * (Aᶜ.card : ℝ) / ((n : ℝ) * ((n : ℝ) - 1)) := by
  have huniv : (Finset.univ : Finset (Fin n)).card = n := by
    rw [Finset.card_univ, Fintype.card_fin]
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have hpred : ((n : ℝ) - 1) ≠ 0 := by
    have : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    intro h; linarith
  rw [integral_crossing_count, integral_within_count, huniv]
  have hcast : (((n - 1 : ℕ) : ℝ)) = (n : ℝ) - 1 := by
    have : (1 : ℕ) ≤ n := by omega
    push_cast [this]
    ring
  rw [hcast]
  field_simp

end ErdosRenyi
