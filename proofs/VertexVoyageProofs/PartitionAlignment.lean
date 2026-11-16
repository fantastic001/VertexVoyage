import Mathlib
import VertexVoyageProofs.PartitioningF1

/-!
# Aligning per-partition embeddings

Formalizes `results/E10-partition-alignment/` and `doc/theory/partition-alignment.md`.

Model: nodes `V`, an embedding `f : V → E` into a metric space, a contrastive loss over a set of
positive pairs and a set of negative pairs, and a triplet loss over a set of
(anchor, positive, negative) triples.

The formalization exposes the points the note relies on:

* both losses see the embedding only through pairwise distances, so pairs inside one partition
  are blind to any per-partition isometry: the alignment signal lives in the cut pairs;
* the contrastive loss is frustrated by every open wedge (`m ^ 2 / 3`) and the constant map is a
  minimiser of its positive part;
* both losses are positively homogeneous, so a margin is meaningless without a fixed scale;
* the skip-gram score is invariant under `GL(d)`, a random isometry moves a point by
  `2 ‖z‖ ^ 2` in expectation, and a residual ReLU block of width `2d` can still represent any
  linear map;
* an isometric alignment can raise the expected score by at most `δ (1 - crossMean)`;
* on the unit sphere distance is cosine, margins are bounded by `2`, the only linear maps that
  keep unit vectors unit are isometries, and the best unit representative of several replicas of
  one node is their normalised sum.
-/

open MeasureTheory

noncomputable section

namespace PartitionAlignment

section Losses

variable {V P E : Type*} [PseudoMetricSpace E]

def contrastiveLoss (edges nonEdges : Finset (V × V)) (m : ℝ) (f : V → E) : ℝ :=
  ∑ e ∈ edges, dist (f e.1) (f e.2) ^ 2
    + ∑ e ∈ nonEdges, max 0 (m - dist (f e.1) (f e.2)) ^ 2

def tripletLoss (triplets : Finset (V × V × V)) (α : ℝ) (f : V → E) : ℝ :=
  ∑ t ∈ triplets, max (dist (f t.1) (f t.2.1) - dist (f t.1) (f t.2.2) + α) 0

variable {edges nonEdges : Finset (V × V)} {triplets : Finset (V × V × V)} {m α : ℝ}
  {f : V → E}

theorem contrastiveLoss_comp_isometry {T : E → E} (hT : Isometry T) :
    contrastiveLoss edges nonEdges m (T ∘ f) = contrastiveLoss edges nonEdges m f := by
  simp [contrastiveLoss, hT.dist_eq]

theorem tripletLoss_comp_isometry {T : E → E} (hT : Isometry T) :
    tripletLoss triplets α (T ∘ f) = tripletLoss triplets α f := by
  simp [tripletLoss, hT.dist_eq]

theorem contrastiveLoss_partitionwise_isometry (π : V → P) (T : P → E → E)
    (hT : ∀ p, Isometry (T p)) (hE : ∀ e ∈ edges, π e.1 = π e.2)
    (hN : ∀ e ∈ nonEdges, π e.1 = π e.2) :
    contrastiveLoss edges nonEdges m (fun v => T (π v) (f v))
      = contrastiveLoss edges nonEdges m f := by
  unfold contrastiveLoss
  congr 1
  · refine Finset.sum_congr rfl fun e he => ?_
    beta_reduce
    rw [hE e he, (hT _).dist_eq]
  · refine Finset.sum_congr rfl fun e he => ?_
    beta_reduce
    rw [hN e he, (hT _).dist_eq]

theorem tripletLoss_partitionwise_isometry (π : V → P) (T : P → E → E)
    (hT : ∀ p, Isometry (T p))
    (hS : ∀ t ∈ triplets, π t.2.1 = π t.1 ∧ π t.2.2 = π t.1) :
    tripletLoss triplets α (fun v => T (π v) (f v)) = tripletLoss triplets α f := by
  unfold tripletLoss
  refine Finset.sum_congr rfl fun t ht => ?_
  beta_reduce
  rw [(hS t ht).1, (hS t ht).2, (hT _).dist_eq, (hT _).dist_eq]

theorem contrastiveLoss_union [DecidableEq V] {cutEdges cutNonEdges : Finset (V × V)}
    (hE : Disjoint edges cutEdges) (hN : Disjoint nonEdges cutNonEdges) :
    contrastiveLoss (edges ∪ cutEdges) (nonEdges ∪ cutNonEdges) m f
      = contrastiveLoss edges nonEdges m f + contrastiveLoss cutEdges cutNonEdges m f := by
  unfold contrastiveLoss
  rw [Finset.sum_union hE, Finset.sum_union hN]
  ring

theorem contrastiveLoss_const (x : E) (hm : 0 ≤ m) :
    contrastiveLoss edges nonEdges m (fun _ => x) = nonEdges.card * m ^ 2 := by
  simp [contrastiveLoss, max_eq_right hm]

theorem contrastiveLoss_const_of_no_negatives (x : E) :
    contrastiveLoss edges ∅ m (fun _ => x) = 0 := by
  simp [contrastiveLoss]

theorem contrastiveLoss_nonneg : 0 ≤ contrastiveLoss edges nonEdges m f :=
  add_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _)
    (Finset.sum_nonneg fun _ _ => sq_nonneg _)

end Losses

section Frustration

theorem wedge_lower_bound {a b c m : ℝ} (hm : 0 ≤ m) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hc0 : 0 ≤ c) (hc : c ≤ a + b) :
    m ^ 2 / 3 ≤ a ^ 2 + b ^ 2 + max 0 (m - c) ^ 2 := by
  rcases le_total (m - c) 0 with h | h
  · rw [max_eq_left h]
    nlinarith [sq_nonneg (a - b), mul_nonneg (by linarith : (0 : ℝ) ≤ a + b - m)
      (by linarith : (0 : ℝ) ≤ a + b + m), sq_nonneg m]
  · rw [max_eq_right h]
    nlinarith [sq_nonneg (a - b), mul_nonneg (sub_nonneg.2 hc)
      (by linarith : (0 : ℝ) ≤ a + b + c), sq_nonneg (3 * c - 2 * m)]

theorem wedge_lower_bound_attained {m : ℝ} (hm : 0 ≤ m) :
    (m / 3) ^ 2 + (m / 3) ^ 2 + max 0 (m - 2 * m / 3) ^ 2 = m ^ 2 / 3 := by
  rw [max_eq_right (by linarith)]
  ring

variable {V E : Type*} [PseudoMetricSpace E]

theorem dist_wedge_lower_bound (x y z : E) {m : ℝ} (hm : 0 ≤ m) :
    m ^ 2 / 3 ≤ dist x y ^ 2 + dist y z ^ 2 + max 0 (m - dist x z) ^ 2 :=
  wedge_lower_bound hm dist_nonneg dist_nonneg dist_nonneg (dist_triangle x y z)

theorem contrastiveLoss_ge_of_open_wedge {edges nonEdges : Finset (V × V)} {m : ℝ}
    (f : V → E) {u w v : V} (huw : (u, w) ∈ edges) (hwv : (w, v) ∈ edges)
    (huv : (u, v) ∈ nonEdges) (hne : u ≠ w) (hm : 0 ≤ m) :
    m ^ 2 / 3 ≤ contrastiveLoss edges nonEdges m f := by
  classical
  have hpair : (u, w) ≠ (w, v) := fun h => hne (Prod.ext_iff.1 h).1
  have hsub : ({(u, w), (w, v)} : Finset (V × V)) ⊆ edges := by
    intro e he
    rcases Finset.mem_insert.1 he with rfl | he
    · exact huw
    · rw [Finset.mem_singleton.1 he]
      exact hwv
  have hpos : dist (f u) (f w) ^ 2 + dist (f w) (f v) ^ 2
      ≤ ∑ e ∈ edges, dist (f e.1) (f e.2) ^ 2 := by
    have h := Finset.sum_le_sum_of_subset_of_nonneg (f := fun e : V × V =>
      dist (f e.1) (f e.2) ^ 2) hsub (fun _ _ _ => sq_nonneg _)
    rwa [Finset.sum_pair hpair] at h
  have hneg : max 0 (m - dist (f u) (f v)) ^ 2
      ≤ ∑ e ∈ nonEdges, max 0 (m - dist (f e.1) (f e.2)) ^ 2 :=
    Finset.single_le_sum (f := fun e : V × V => max 0 (m - dist (f e.1) (f e.2)) ^ 2)
      (fun _ _ => sq_nonneg _) huv
  have hwedge := dist_wedge_lower_bound (f u) (f w) (f v) hm
  unfold contrastiveLoss
  linarith

end Frustration

section Scale

theorem contrastive_hinge_scale {c : ℝ} (hc : 0 < c) (m d : ℝ) :
    max 0 (m - c * d) = c * max 0 (m / c - d) := by
  rw [mul_max_of_nonneg _ _ hc.le]
  congr 1
  · simp
  · field_simp

theorem triplet_hinge_scale {c : ℝ} (hc : 0 < c) (a b α : ℝ) :
    max (c * a - c * b + α) 0 = c * max (a - b + α / c) 0 := by
  rw [mul_max_of_nonneg _ _ hc.le]
  congr 1
  · field_simp
  · simp

theorem triplet_hinge_vanishes_of_large_scale {a b α c : ℝ} (hab : a < b)
    (hc : α / (b - a) ≤ c) : max (c * a - c * b + α) 0 = 0 := by
  have hpos : 0 < b - a := sub_pos.2 hab
  have h : α ≤ c * (b - a) := (div_le_iff₀ hpos).1 hc
  exact max_eq_right (by linarith)

end Scale

section Symmetry

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem sgns_score_gl_invariant (A : Matrix n n ℝ) (hA : IsUnit A.det) (w c : n → ℝ) :
    (A.mulVec w) ⬝ᵥ (A⁻¹.transpose.mulVec c) = w ⬝ᵥ c := by
  rw [Matrix.mulVec_transpose, dotProduct_comm, Matrix.dotProduct_mulVec,
    Matrix.vecMul_vecMul, Matrix.nonsing_inv_mul _ hA, Matrix.vecMul_one, dotProduct_comm]

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

theorem norm_isometry_sub_sq (Q : H →ₗᵢ[ℝ] H) (z : H) :
    ‖Q z - z‖ ^ 2 = 2 * ‖z‖ ^ 2 - 2 * inner ℝ (Q z) z := by
  rw [@norm_sub_sq_real, Q.norm_map]
  ring

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

theorem integral_inner_eq_zero_of_symmetric (σ : Ω ≃ᵐ Ω) (hσ : MeasurePreserving σ μ μ)
    (Q : Ω → H →ₗᵢ[ℝ] H) (z : H) (hQ : ∀ ω, Q (σ ω) z = -(Q ω z)) :
    ∫ ω, inner ℝ (Q ω z) z ∂μ = 0 := by
  have h := hσ.integral_comp' (fun ω => inner ℝ (Q ω z) z)
  simp only [hQ, inner_neg_left, integral_neg] at h
  linarith

theorem expected_sq_displacement_of_symmetric [IsProbabilityMeasure μ] (σ : Ω ≃ᵐ Ω)
    (hσ : MeasurePreserving σ μ μ) (Q : Ω → H →ₗᵢ[ℝ] H) (z : H)
    (hQ : ∀ ω, Q (σ ω) z = -(Q ω z))
    (hint : Integrable (fun ω => inner ℝ (Q ω z) z) μ) :
    ∫ ω, ‖Q ω z - z‖ ^ 2 ∂μ = 2 * ‖z‖ ^ 2 := by
  simp_rw [norm_isometry_sub_sq]
  rw [integral_sub (integrable_const _) (hint.const_mul 2), integral_const_mul,
    integral_const_mul, integral_inner_eq_zero_of_symmetric σ hσ Q z hQ]
  simp

end Symmetry

section Expressivity

theorem relu_sub_relu_neg (t : ℝ) : max t 0 - max (-t) 0 = t := by
  rcases le_total 0 t with h | h
  · rw [max_eq_left h, max_eq_right (neg_nonpos.2 h), sub_zero]
  · rw [max_eq_right h, max_eq_left (neg_nonneg.2 h), zero_sub, neg_neg]

variable {n : Type*} [Fintype n] [DecidableEq n]

def relu (x : n → ℝ) : n → ℝ := fun i => max (x i) 0

theorem residual_relu_represents_linear (Q : Matrix n n ℝ) (z : n → ℝ) :
    z + (relu ((Q - 1).mulVec z) - relu (-((Q - 1).mulVec z))) = Q.mulVec z := by
  ext i
  simp only [Pi.add_apply, Pi.sub_apply, relu, Pi.neg_apply]
  rw [relu_sub_relu_neg]
  simp [Matrix.sub_mulVec]

end Expressivity

section Anchors

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] {ι : Type*}

theorem linear_alignment_unique_of_spanning_anchors {anchors : Set H}
    (hspan : Submodule.span ℝ anchors = ⊤) {Q₁ Q₂ : H →ₗ[ℝ] H}
    (h : ∀ a ∈ anchors, Q₁ a = Q₂ a) : Q₁ = Q₂ :=
  LinearMap.ext_on hspan h

theorem procrustes_objective (s : Finset ι) (a b : ι → H) (Q : H →ₗᵢ[ℝ] H) :
    ∑ i ∈ s, ‖Q (a i) - b i‖ ^ 2
      = ∑ i ∈ s, ‖a i‖ ^ 2 + ∑ i ∈ s, ‖b i‖ ^ 2 - 2 * ∑ i ∈ s, inner ℝ (Q (a i)) (b i) := by
  simp_rw [@norm_sub_sq_real, Q.norm_map]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

theorem gram_eq_of_exact_alignment (a b : ι → H) (Q : H →ₗᵢ[ℝ] H)
    (h : ∀ i, Q (a i) = b i) (i j : ι) : inner ℝ (b i) (b j) = inner ℝ (a i) (a j) := by
  rw [← h i, ← h j, Q.inner_map_map]

end Anchors

section UnitSphere

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] {ι : Type*}

theorem norm_sub_sq_of_unit {x y : H} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ‖x - y‖ ^ 2 = 2 - 2 * inner ℝ x y := by
  rw [@norm_sub_sq_real, hx, hy]
  ring

omit [InnerProductSpace ℝ H] in
theorem norm_sub_le_two_of_unit {x y : H} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ‖x - y‖ ≤ 2 := by
  have h := norm_sub_le x y
  rw [hx, hy] at h
  linarith

theorem hinge_active_iff_inner {x y : H} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) {m : ℝ}
    (hm : 0 < m) : ‖x - y‖ < m ↔ 1 - m ^ 2 / 2 < inner ℝ x y := by
  have hsq := norm_sub_sq_of_unit hx hy
  have hnonneg : 0 ≤ ‖x - y‖ := norm_nonneg _
  constructor
  · intro h
    nlinarith
  · intro h
    by_contra hcon
    have hle : m ≤ ‖x - y‖ := not_lt.1 hcon
    nlinarith

omit [InnerProductSpace ℝ H] in
theorem hinge_ge_of_unit_of_large_margin {x y : H} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    {m : ℝ} (hm : 2 ≤ m) : (m - 2) ^ 2 ≤ max 0 (m - ‖x - y‖) ^ 2 := by
  have hle := norm_sub_le_two_of_unit hx hy
  rw [max_eq_right (by linarith)]
  nlinarith

omit [InnerProductSpace ℝ H] in
theorem triplet_margin_le_two_of_zero_loss {a p q : H} (ha : ‖a‖ = 1) (hq : ‖q‖ = 1)
    {α : ℝ} (h : max (‖a - p‖ - ‖a - q‖ + α) 0 = 0) : α ≤ 2 := by
  have hle := norm_sub_le_two_of_unit ha hq
  have hpos : 0 ≤ ‖a - p‖ := norm_nonneg _
  have hmax := le_max_left (‖a - p‖ - ‖a - q‖ + α) 0
  rw [h] at hmax
  linarith

theorem linear_preserves_norm_of_preserves_sphere (f : H →ₗ[ℝ] H)
    (h : ∀ x, ‖x‖ = 1 → ‖f x‖ = 1) (x : H) : ‖f x‖ = ‖x‖ := by
  by_cases hx : x = 0
  · simp [hx]
  · have hnorm : ‖x‖ ≠ 0 := norm_ne_zero_iff.2 hx
    have hunit : ‖(‖x‖⁻¹ : ℝ) • x‖ = 1 := by
      rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hnorm]
    have hsplit : x = ‖x‖ • ((‖x‖⁻¹ : ℝ) • x) := by
      rw [smul_smul, mul_inv_cancel₀ hnorm, one_smul]
    conv_lhs => rw [hsplit]
    rw [map_smul, norm_smul, h _ hunit, norm_norm, mul_one]

theorem procrustes_objective_of_unit (s : Finset ι) (a b : ι → H) (Q : H →ₗᵢ[ℝ] H)
    (ha : ∀ i ∈ s, ‖a i‖ = 1) (hb : ∀ i ∈ s, ‖b i‖ = 1) :
    ∑ i ∈ s, ‖Q (a i) - b i‖ ^ 2 = 2 * s.card - 2 * ∑ i ∈ s, inner ℝ (Q (a i)) (b i) := by
  rw [Finset.mul_sum, ← nsmul_eq_mul', ← Finset.sum_const, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [norm_sub_sq_of_unit (by rw [Q.norm_map, ha i hi]) (hb i hi)]

theorem replica_spread_of_unit (s : Finset ι) (w : ι → H) {c : H}
    (hw : ∀ i ∈ s, ‖w i‖ = 1) (hc : ‖c‖ = 1) :
    ∑ i ∈ s, ‖w i - c‖ ^ 2 = 2 * s.card - 2 * inner ℝ (∑ i ∈ s, w i) c := by
  rw [sum_inner, Finset.mul_sum, ← nsmul_eq_mul', ← Finset.sum_const,
    ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [norm_sub_sq_of_unit (hw i hi) hc]

theorem replica_spread_ge (s : Finset ι) (w : ι → H) {c : H}
    (hw : ∀ i ∈ s, ‖w i‖ = 1) (hc : ‖c‖ = 1) :
    2 * s.card - 2 * ‖∑ i ∈ s, w i‖ ≤ ∑ i ∈ s, ‖w i - c‖ ^ 2 := by
  rw [replica_spread_of_unit s w hw hc]
  have h := real_inner_le_norm (∑ i ∈ s, w i) c
  rw [hc, mul_one] at h
  linarith

theorem replica_spread_at_normalised_sum (s : Finset ι) (w : ι → H)
    (hw : ∀ i ∈ s, ‖w i‖ = 1) (hsum : ∑ i ∈ s, w i ≠ 0) :
    ∑ i ∈ s, ‖w i - (‖∑ j ∈ s, w j‖⁻¹ : ℝ) • ∑ j ∈ s, w j‖ ^ 2
      = 2 * s.card - 2 * ‖∑ i ∈ s, w i‖ := by
  have hnorm : ‖∑ j ∈ s, w j‖ ≠ 0 := norm_ne_zero_iff.2 hsum
  have hunit : ‖(‖∑ j ∈ s, w j‖⁻¹ : ℝ) • ∑ j ∈ s, w j‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hnorm]
  rw [replica_spread_of_unit s w hw hunit, inner_smul_right, real_inner_self_eq_norm_sq]
  field_simp

theorem replica_pairwise_spread_of_unit (s : Finset ι) (w : ι → H)
    (hw : ∀ i ∈ s, ‖w i‖ = 1) :
    ∑ i ∈ s, ∑ j ∈ s, ‖w i - w j‖ ^ 2
      = 2 * (s.card : ℝ) ^ 2 - 2 * ‖∑ i ∈ s, w i‖ ^ 2 := by
  have hinner : ∑ i ∈ s, ∑ j ∈ s, inner ℝ (w i) (w j) = ‖∑ i ∈ s, w i‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, sum_inner]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [inner_sum]
  have hterm : ∀ i ∈ s, ∑ j ∈ s, ‖w i - w j‖ ^ 2
      = 2 * s.card - 2 * ∑ j ∈ s, inner ℝ (w i) (w j) := by
    intro i hi
    rw [Finset.mul_sum, ← nsmul_eq_mul', ← Finset.sum_const, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [norm_sub_sq_of_unit (hw i hi) (hw j hj)]
  rw [Finset.sum_congr rfl hterm, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
    ← Finset.mul_sum, hinner]
  ring

end UnitSphere

section Factorisation

variable {n m d : Type*} [Fintype n] [Fintype m] [Fintype d] [DecidableEq d]

theorem mul_left_cancel_of_gram_unit {k : Type*} [Fintype k] (W : Matrix n d ℝ)
    (hW : IsUnit (W.transpose * W).det) {X Y : Matrix d k ℝ} (h : W * X = W * Y) : X = Y := by
  calc X = (W.transpose * W)⁻¹ * (W.transpose * W * X) := by
        rw [← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hW, Matrix.one_mul]
    _ = (W.transpose * W)⁻¹ * (W.transpose * (W * X)) := by rw [Matrix.mul_assoc W.transpose]
    _ = (W.transpose * W)⁻¹ * (W.transpose * (W * Y)) := by rw [h]
    _ = (W.transpose * W)⁻¹ * (W.transpose * W * Y) := by rw [Matrix.mul_assoc W.transpose]
    _ = Y := by rw [← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hW, Matrix.one_mul]

theorem factorisation_unique_up_to_gl (W W' : Matrix n d ℝ) (C C' : Matrix m d ℝ)
    (hscore : W * C.transpose = W' * C'.transpose)
    (hW : IsUnit (W.transpose * W).det) (hC : IsUnit (C.transpose * C).det)
    (hC' : IsUnit (C'.transpose * C').det) :
    ∃ A : Matrix d d ℝ, IsUnit A.det ∧ W' = W * A ∧ C' = C * A⁻¹.transpose := by
  set A : Matrix d d ℝ := C.transpose * C' * (C'.transpose * C')⁻¹ with hAdef
  set B : Matrix d d ℝ := C'.transpose * C * (C.transpose * C)⁻¹ with hBdef
  have hWA : W * A = W' := by
    rw [hAdef, ← Matrix.mul_assoc, ← Matrix.mul_assoc, hscore, Matrix.mul_assoc W',
      Matrix.mul_assoc W', Matrix.mul_nonsing_inv _ hC', Matrix.mul_one]
  have hWB : W' * B = W := by
    rw [hBdef, ← Matrix.mul_assoc, ← Matrix.mul_assoc, ← hscore, Matrix.mul_assoc W,
      Matrix.mul_assoc W, Matrix.mul_nonsing_inv _ hC, Matrix.mul_one]
  have hAB : A * B = 1 := by
    apply mul_left_cancel_of_gram_unit W hW
    rw [← Matrix.mul_assoc, hWA, hWB, Matrix.mul_one]
  have hBA : B * A = 1 := mul_eq_one_comm.1 hAB
  have hAunit : IsUnit A.det := Matrix.isUnit_det_of_right_inverse hAB
  have hAinv : A⁻¹ = B := Matrix.inv_eq_right_inv hAB
  have hCt : C.transpose = A * C'.transpose := by
    apply mul_left_cancel_of_gram_unit W hW
    rw [← Matrix.mul_assoc, hWA, hscore]
  refine ⟨A, hAunit, hWA.symm, ?_⟩
  have hC't : C'.transpose = B * C.transpose := by
    rw [hCt, ← Matrix.mul_assoc, hBA, Matrix.one_mul]
  rw [hAinv, ← Matrix.transpose_inj, Matrix.transpose_mul, Matrix.transpose_transpose, hC't]

end Factorisation

section NormalisedLinear

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

theorem normalised_linear_is_linear_only_if_conformal (f g : H →ₗ[ℝ] H)
    (hf : ∀ x, ‖x‖ = 1 → f x ≠ 0) (hg : ∀ x, ‖x‖ = 1 → g x = ‖f x‖⁻¹ • f x) :
    ∃ c : ℝ, 0 < c ∧ (∀ x, f x = c • g x) ∧ ∀ x, ‖g x‖ = ‖x‖ := by
  have hgnorm : ∀ x, ‖g x‖ = ‖x‖ :=
    linear_preserves_norm_of_preserves_sphere g (fun x hx => by
      rw [hg x hx, norm_smul, norm_inv, norm_norm,
        inv_mul_cancel₀ (norm_ne_zero_iff.2 (hf x hx))])
  have hginj : ∀ x, g x = 0 → x = 0 := fun x h =>
    norm_eq_zero.1 (by rw [← hgnorm, h, norm_zero])
  have hunit : ∀ x : H, x ≠ 0 → ‖(‖x‖⁻¹ : ℝ) • x‖ = 1 := fun x hx => by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.2 hx)]
  have hfne : ∀ x : H, x ≠ 0 → f x ≠ 0 := fun x hx h0 => by
    have h := hf _ (hunit x hx)
    rw [map_smul, h0, smul_zero] at h
    exact h rfl
  have key : ∀ x : H, x ≠ 0 → f x = (‖f x‖ / ‖x‖) • g x := fun x hx => by
    have hnx : ‖x‖ ≠ 0 := norm_ne_zero_iff.2 hx
    have hnf : ‖f x‖ ≠ 0 := norm_ne_zero_iff.2 (hfne x hx)
    have h := hg _ (hunit x hx)
    rw [map_smul, map_smul, norm_smul, norm_inv, norm_norm, smul_smul] at h
    have h2 := congrArg (fun v => (‖f x‖ : ℝ) • v) h
    simp only [smul_smul] at h2
    have e1 : ‖f x‖ * ‖x‖⁻¹ = ‖f x‖ / ‖x‖ := by ring
    have e2 : ‖f x‖ * ((‖x‖⁻¹ * ‖f x‖)⁻¹ * ‖x‖⁻¹) = 1 := by field_simp
    rw [e1, e2, one_smul] at h2
    exact h2.symm
  have hscale : ∀ (x : H) (t : ℝ), x ≠ 0 → t ≠ 0 →
      ‖f (t • x)‖ / ‖t • x‖ = ‖f x‖ / ‖x‖ := fun x t hx ht => by
    have hnx : ‖x‖ ≠ 0 := norm_ne_zero_iff.2 hx
    have hnt : |t| ≠ 0 := abs_ne_zero.2 ht
    rw [map_smul, norm_smul, norm_smul, Real.norm_eq_abs]
    field_simp
  have hconst : ∀ x y : H, x ≠ 0 → y ≠ 0 → ‖f x‖ / ‖x‖ = ‖f y‖ / ‖y‖ := by
    intro x y hx hy
    by_cases hli : LinearIndependent ℝ ![x, y]
    · have hpair := LinearIndependent.pair_iff.1 hli
      have hxy : x + y ≠ 0 := fun h0 => by
        have := hpair 1 1 (by simpa using h0)
        exact one_ne_zero this.1
      have h1 := key (x + y) hxy
      have hkx := key x hx
      have hky := key y hy
      have hzero : g ((‖f x‖ / ‖x‖ - ‖f (x + y)‖ / ‖x + y‖) • x +
          (‖f y‖ / ‖y‖ - ‖f (x + y)‖ / ‖x + y‖) • y) = 0 := by
        have h1' : f x + f y =
            (‖f (x + y)‖ / ‖x + y‖) • g x + (‖f (x + y)‖ / ‖x + y‖) • g y := by
          rw [← smul_add, ← map_add, ← map_add]
          exact h1
        rw [map_add, map_smul, map_smul, sub_smul, sub_smul, ← hkx, ← hky,
          sub_add_sub_comm, h1', sub_self]
      have h2 := hpair _ _ (hginj _ hzero)
      linarith [h2.1, h2.2]
    · rw [LinearIndependent.pair_iff] at hli
      push Not at hli
      obtain ⟨s, t, hst, hne⟩ := hli
      have hs : s ≠ 0 := fun h0 => by
        rw [h0, zero_smul, zero_add] at hst
        rcases smul_eq_zero.1 hst with ht | hy0
        · exact hne h0 ht
        · exact hy hy0
      have ht : t ≠ 0 := fun h0 => by
        rw [h0, zero_smul, add_zero] at hst
        rcases smul_eq_zero.1 hst with hs0 | hx0
        · exact hs hs0
        · exact hx hx0
      have hy_eq : y = (-(s / t)) • x := by
        have h3 : t • y = -(s • x) := eq_neg_of_add_eq_zero_right hst
        have h4 := congrArg (fun v => (t⁻¹ : ℝ) • v) h3
        simp only [smul_smul, inv_mul_cancel₀ ht, one_smul, smul_neg] at h4
        rw [h4, neg_smul]
        congr 2
        ring
      rw [hy_eq, hscale x _ hx (neg_ne_zero.2 (div_ne_zero hs ht))]
  by_cases hex : ∃ x0 : H, x0 ≠ 0
  · obtain ⟨x0, hx0⟩ := hex
    refine ⟨‖f x0‖ / ‖x0‖,
      div_pos (norm_pos_iff.2 (hfne x0 hx0)) (norm_pos_iff.2 hx0), fun x => ?_, hgnorm⟩
    by_cases hx : x = 0
    · simp [hx]
    · rw [← hconst x x0 hx hx0]
      exact key x hx
  · push Not at hex
    exact ⟨1, one_pos, fun x => by simp [hex x], hgnorm⟩

end NormalisedLinear

section SkipGramOptimum

theorem sigmoid_at_log_ratio {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    1 / (1 + Real.exp (-Real.log (a / b))) = a / (a + b) := by
  rw [Real.exp_neg, Real.exp_log (div_pos ha hb)]
  field_simp

theorem one_sub_sigmoid (x : ℝ) :
    1 - 1 / (1 + Real.exp (-x)) = 1 / (1 + Real.exp (-(-x))) := by
  rw [neg_neg, Real.exp_neg]
  have h : 0 < Real.exp x := Real.exp_pos x
  field_simp
  ring

theorem pair_objective_le_at_ratio {a b p : ℝ} (ha : 0 < a) (hb : 0 < b) (hp0 : 0 < p)
    (hp1 : p < 1) :
    a * Real.log p + b * Real.log (1 - p) ≤
      a * Real.log (a / (a + b)) + b * Real.log (b / (a + b)) := by
  have hab : 0 < a + b := add_pos ha hb
  have hq : 0 < 1 - p := sub_pos.2 hp1
  have h1 := Real.log_le_sub_one_of_pos (div_pos hp0 (div_pos ha hab))
  have h2 := Real.log_le_sub_one_of_pos (div_pos hq (div_pos hb hab))
  rw [Real.log_div hp0.ne' (div_pos ha hab).ne'] at h1
  rw [Real.log_div hq.ne' (div_pos hb hab).ne'] at h2
  have e1 : a * (p / (a / (a + b)) - 1) = (a + b) * p - a := by field_simp
  have e2 : b * ((1 - p) / (b / (a + b)) - 1) = (a + b) * (1 - p) - b := by field_simp
  have m1 := mul_le_mul_of_nonneg_left h1 ha.le
  have m2 := mul_le_mul_of_nonneg_left h2 hb.le
  rw [e1] at m1
  rw [e2] at m2
  linarith

end SkipGramOptimum

section Headroom

open PartitioningF1

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {F F' : Ω → ℝ} {Cut : Set Ω}

theorem alignment_gain_le (hCut : MeasurableSet Cut) (hF : Integrable F μ)
    (hF' : Integrable F' μ) (h0 : μ Cut ≠ 0) (h1 : μ Cutᶜ ≠ 0)
    (hwithin : withinMean μ F' Cut = withinMean μ F Cut)
    (hcross : crossMean μ F' Cut ≤ 1) :
    ∫ ω, F' ω ∂μ - ∫ ω, F ω ∂μ ≤ cutFraction μ Cut * (1 - crossMean μ F Cut) := by
  have hδ : 0 ≤ cutFraction μ Cut := ENNReal.toReal_nonneg
  rw [expected_eq_within_cross hCut hF h0 h1, expected_eq_within_cross hCut hF' h0 h1,
    hwithin]
  nlinarith [mul_le_mul_of_nonneg_right hcross hδ]

end Headroom

section GradientCancellation

/-! Two loss terms with gradients `a` (attraction) and `r` (repulsion): the optimiser steps on
`a + r`, so their angle is recovered from the three logged norms, and the norm of the step is
squeezed between `|‖a‖ - ‖r‖|` and `‖a‖ + ‖r‖`. -/

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] {ι : Type*}

theorem norm_add_sq_of_terms (a r : H) :
    ‖a + r‖ ^ 2 = ‖a‖ ^ 2 + ‖r‖ ^ 2 + 2 * inner ℝ a r := by
  rw [@norm_add_sq_real]
  ring

theorem cosine_of_term_norms {a r : H} (ha : a ≠ 0) (hr : r ≠ 0) :
    inner ℝ a r / (‖a‖ * ‖r‖) =
      (‖a + r‖ ^ 2 - ‖a‖ ^ 2 - ‖r‖ ^ 2) / (2 * (‖a‖ * ‖r‖)) := by
  have hpos : 0 < ‖a‖ * ‖r‖ := mul_pos (norm_pos_iff.mpr ha) (norm_pos_iff.mpr hr)
  rw [norm_add_sq_of_terms]
  field_simp
  ring

omit [InnerProductSpace ℝ H] in
theorem norm_add_le_sum_of_term_norms (a r : H) : ‖a + r‖ ≤ ‖a‖ + ‖r‖ :=
  norm_add_le a r

omit [InnerProductSpace ℝ H] in
theorem abs_sub_term_norms_le_norm_add (a r : H) : |‖a‖ - ‖r‖| ≤ ‖a + r‖ := by
  have h := abs_norm_sub_norm_le a (-r)
  rwa [norm_neg, sub_neg_eq_add] at h

omit [InnerProductSpace ℝ H] in
theorem cancellation_ratio_le_one {a r : H} (h : 0 < ‖a‖ + ‖r‖) :
    ‖a + r‖ / (‖a‖ + ‖r‖) ≤ 1 := by
  rw [div_le_one h]
  exact norm_add_le a r

theorem cosine_le_of_cancellation {a r : H} (ha : a ≠ 0) (hr : r ≠ 0) {ρ : ℝ}
    (hρ : ‖a + r‖ ≤ ρ * (‖a‖ + ‖r‖)) :
    inner ℝ a r / (‖a‖ * ‖r‖) ≤
      ((ρ ^ 2 * (‖a‖ + ‖r‖) ^ 2) - ‖a‖ ^ 2 - ‖r‖ ^ 2) / (2 * (‖a‖ * ‖r‖)) := by
  have hpos : 0 < 2 * (‖a‖ * ‖r‖) :=
    by positivity
  rw [cosine_of_term_norms ha hr]
  apply div_le_div_of_nonneg_right _ hpos.le
  have hsq : ‖a + r‖ ^ 2 ≤ (ρ * (‖a‖ + ‖r‖)) ^ 2 :=
    pow_le_pow_left₀ (norm_nonneg _) hρ 2
  nlinarith [hsq]

theorem sum_norm_sub_sq_of_unit (s : Finset ι) (x y : ι → H)
    (hx : ∀ i ∈ s, ‖x i‖ = 1) (hy : ∀ i ∈ s, ‖y i‖ = 1) :
    ∑ i ∈ s, ‖x i - y i‖ ^ 2 = 2 * s.card - 2 * ∑ i ∈ s, inner ℝ (x i) (y i) := by
  rw [Finset.sum_congr rfl (fun i hi => norm_sub_sq_of_unit (hx i hi) (hy i hi)),
    Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp [mul_comm]

end GradientCancellation

section CutRecall

/-! If no reconstructed edge crosses a partition, every true positive is a within edge, and
recall over all edges is bounded by the within share `1 - δ`. -/

theorem recall_le_within_share {tp within cut : ℝ} (hn : 0 < within + cut)
    (htp : tp ≤ within) :
    tp / (within + cut) ≤ 1 - cut / (within + cut) := by
  rw [div_le_iff₀ hn, sub_mul, div_mul_cancel₀ _ hn.ne']
  linarith

theorem recall_split_by_scope {tpw tpc w c : ℝ} (hw : 0 < w) (hc : 0 < c) :
    (tpw + tpc) / (w + c) = w / (w + c) * (tpw / w) + c / (w + c) * (tpc / c) := by
  have hn : 0 < w + c := add_pos hw hc
  field_simp

theorem recall_le_within_share_add_cut_recall {tpw tpc w c : ℝ} (hw : 0 < w) (hc : 0 < c)
    (htpw : tpw ≤ w) :
    (tpw + tpc) / (w + c) ≤ w / (w + c) + c / (w + c) * (tpc / c) := by
  rw [recall_split_by_scope hw hc]
  have hn : 0 < w + c := add_pos hw hc
  have hratio : tpw / w ≤ 1 := (div_le_one hw).mpr htpw
  have hshare : 0 ≤ w / (w + c) := div_nonneg hw.le hn.le
  nlinarith [mul_le_mul_of_nonneg_left hratio hshare]

end CutRecall

section CrossWeighting

/-! `n` positive links of which `c` cross partitions, `δ = c / n`. Weighting the cross links by
`1 / δ` and normalising by the total weight gives the cross scope the share `1 / (2 - δ) ≥ 1/2`,
and the weighted attraction is the share-weighted mix of the two scope means. -/

theorem total_weight_of_inverse_fraction {n c : ℝ} (hc : 0 < c) (hcn : c < n) :
    (n - c) + c / (c / n) = 2 * n - c := by
  have hn : 0 < n := lt_trans hc hcn
  field_simp
  ring

theorem cross_share_of_inverse_fraction {n c : ℝ} (hc : 0 < c) (hcn : c < n) :
    (c / (c / n)) / ((n - c) + c / (c / n)) = 1 / (2 - c / n) := by
  have hn : 0 < n := lt_trans hc hcn
  rw [total_weight_of_inverse_fraction hc hcn]
  have h2 : 0 < 2 * n - c := by linarith
  field_simp

theorem within_share_of_inverse_fraction {n c : ℝ} (hc : 0 < c) (hcn : c < n) :
    (n - c) / ((n - c) + c / (c / n)) = (1 - c / n) / (2 - c / n) := by
  have hn : 0 < n := lt_trans hc hcn
  rw [total_weight_of_inverse_fraction hc hcn]
  have h2 : 0 < 2 * n - c := by linarith
  field_simp

theorem half_le_cross_share {n c : ℝ} (hc : 0 < c) (hcn : c < n) :
    1 / 2 ≤ 1 / (2 - c / n) := by
  have hn : 0 < n := lt_trans hc hcn
  have hδ : 0 < c / n := div_pos hc hn
  have hδ1 : c / n < 1 := (div_lt_one hn).mpr hcn
  apply one_div_le_one_div_of_le (by linarith) (by linarith)

theorem weighted_mean_of_scopes {n c sw sc : ℝ} (hc : 0 < c) (hcn : c < n) :
    (sw + sc / (c / n)) / ((n - c) + c / (c / n)) =
      (1 - c / n) / (2 - c / n) * (sw / (n - c)) + 1 / (2 - c / n) * (sc / c) := by
  have hn : 0 < n := lt_trans hc hcn
  have hw : 0 < n - c := by linarith
  rw [total_weight_of_inverse_fraction hc hcn]
  have h2 : 0 < 2 * n - c := by linarith
  have h2' : (2 : ℝ) - c / n ≠ 0 := by
    have : c / n < 1 := (div_lt_one hn).mpr hcn
    linarith
  field_simp

theorem cancellation_ratio_of_terms_le_one {H ι : Type*} [NormedAddCommGroup H]
    (s : Finset ι) (g : ι → H) (h : 0 < ∑ i ∈ s, ‖g i‖) :
    ‖∑ i ∈ s, g i‖ / ∑ i ∈ s, ‖g i‖ ≤ 1 := by
  rw [div_le_one h]
  exact norm_sum_le s g

end CrossWeighting

section TripletOrthogonalNegative

/-! A triplet `(a, p, q)` of unit vectors whose negative `q` is orthogonal to the anchor `a`
(the typical position of a random direction) has zero hinge exactly when the positive is close
enough in cosine: `⟪a, p⟫ ≥ 1 - (√2 - α)² / 2`, provided `0 ≤ α ≤ √2`. -/

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

theorem norm_sub_of_orthogonal_unit {a q : H} (ha : ‖a‖ = 1) (hq : ‖q‖ = 1)
    (horth : inner ℝ a q = 0) : ‖a - q‖ = Real.sqrt 2 := by
  have hsq := norm_sub_sq_of_unit ha hq
  rw [horth, mul_zero, sub_zero] at hsq
  rw [← hsq, Real.sqrt_sq (norm_nonneg _)]

theorem triplet_hinge_zero_iff_of_orthogonal_negative {a p q : H} (ha : ‖a‖ = 1)
    (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (horth : inner ℝ a q = 0) {α : ℝ} (hα : 0 ≤ α)
    (hαle : α ≤ Real.sqrt 2) :
    max (‖a - p‖ - ‖a - q‖ + α) 0 = 0 ↔
      1 - (Real.sqrt 2 - α) ^ 2 / 2 ≤ inner ℝ a p := by
  rw [norm_sub_of_orthogonal_unit ha hq horth]
  have hsq := norm_sub_sq_of_unit ha hp
  have hnonneg : 0 ≤ ‖a - p‖ := norm_nonneg _
  have hgap : 0 ≤ Real.sqrt 2 - α := by linarith
  constructor
  · intro h
    have hle : ‖a - p‖ ≤ Real.sqrt 2 - α := by
      have := le_of_eq h
      have h0 : ‖a - p‖ - Real.sqrt 2 + α ≤ 0 := le_trans (le_max_left _ _) this
      linarith
    have := pow_le_pow_left₀ hnonneg hle 2
    nlinarith
  · intro h
    have hsq_le : ‖a - p‖ ^ 2 ≤ (Real.sqrt 2 - α) ^ 2 := by nlinarith
    have hle : ‖a - p‖ ≤ Real.sqrt 2 - α :=
      (pow_le_pow_iff_left₀ hnonneg hgap (by norm_num)).mp hsq_le
    exact max_eq_right (by linarith)

end TripletOrthogonalNegative

section TripletCorrelatedNegative

/-! A negative `q` with cosine `κ` to the anchor sits at distance `√(2 - 2κ)`. The hinge is zero
exactly when `⟪a, p⟫ ≥ 1 - (√(2 - 2κ) - α)² / 2`, provided `α ≤ √(2 - 2κ)`. When the margin
exceeds that distance, the hinge is positive for every positive `p`. -/

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

theorem norm_sub_of_unit_eq_sqrt {a q : H} (ha : ‖a‖ = 1) (hq : ‖q‖ = 1) :
    ‖a - q‖ = Real.sqrt (2 - 2 * inner ℝ a q) := by
  rw [← norm_sub_sq_of_unit ha hq, Real.sqrt_sq (norm_nonneg _)]

theorem triplet_hinge_zero_iff_of_unit {a p q : H} (ha : ‖a‖ = 1) (hp : ‖p‖ = 1)
    {α : ℝ} (hα : α ≤ ‖a - q‖) :
    max (‖a - p‖ - ‖a - q‖ + α) 0 = 0 ↔
      1 - (‖a - q‖ - α) ^ 2 / 2 ≤ inner ℝ a p := by
  have hsq := norm_sub_sq_of_unit ha hp
  have hnonneg : 0 ≤ ‖a - p‖ := norm_nonneg _
  have hgap : 0 ≤ ‖a - q‖ - α := by linarith
  constructor
  · intro h
    have h0 : ‖a - p‖ - ‖a - q‖ + α ≤ 0 := le_trans (le_max_left _ _) (le_of_eq h)
    have hle : ‖a - p‖ ≤ ‖a - q‖ - α := by linarith
    have := pow_le_pow_left₀ hnonneg hle 2
    nlinarith
  · intro h
    have hsq_le : ‖a - p‖ ^ 2 ≤ (‖a - q‖ - α) ^ 2 := by nlinarith
    have hle : ‖a - p‖ ≤ ‖a - q‖ - α :=
      (pow_le_pow_iff_left₀ hnonneg hgap (by norm_num)).mp hsq_le
    exact max_eq_right (by linarith)

omit [InnerProductSpace ℝ H] in
theorem triplet_hinge_pos_of_close_negative {a p q : H} {α : ℝ} (hclose : ‖a - q‖ < α) :
    0 < max (‖a - p‖ - ‖a - q‖ + α) 0 := by
  have hnonneg : 0 ≤ ‖a - p‖ := norm_nonneg _
  exact lt_of_lt_of_le (by linarith) (le_max_left _ _)

theorem norm_sub_lt_of_inner_gt {a q : H} (ha : ‖a‖ = 1) (hq : ‖q‖ = 1) {α : ℝ}
    (hα : 0 ≤ α) (hinner : 1 - α ^ 2 / 2 < inner ℝ a q) : ‖a - q‖ < α := by
  have hsq := norm_sub_sq_of_unit ha hq
  have hlt : ‖a - q‖ ^ 2 < α ^ 2 := by nlinarith
  exact (pow_lt_pow_iff_left₀ (norm_nonneg _) hα (by norm_num)).mp hlt

end TripletCorrelatedNegative

end PartitionAlignment
