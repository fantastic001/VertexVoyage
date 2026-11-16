import Mathlib

/-!
# Partitioning impact on the expected F1 score

Formalizes `results/E08-f1-decomposition/README.md`.

Model: a uniformly random edge `e` of a finite edge set, a score `F` on edges, and the event
`Cut` that `e` crosses partitions. The note decomposes `E[F]` over `Cut` and its complement,
assumes `E[F | Cut] = α E[F | ¬Cut]`, and reads off the dependence on `δ = |Δ| / |E|`.

The formalization exposes two corrections to the note: `α` is only constrained by `α ≥ 0`
(the note's `α ∈ [0,1]` contradicts its later `α > 1`), and `F1` is nonlinear in the edge
outcomes, so the decomposition holds for `F` an average of per-edge scores.
-/

open MeasureTheory ProbabilityTheory

noncomputable section

namespace PartitioningF1

section TotalExpectation

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {F : Ω → ℝ} {A : Set Ω}

theorem setIntegral_eq_measure_mul_condIntegral (hA : μ A ≠ 0) :
    ∫ ω in A, F ω ∂μ = (μ A).toReal * ∫ ω, F ω ∂μ[|A] := by
  have hpos : 0 < (μ A).toReal := ENNReal.toReal_pos hA (measure_ne_top μ A)
  rw [ProbabilityTheory.cond, integral_smul_measure, ENNReal.toReal_inv, smul_eq_mul, ← mul_assoc,
    mul_inv_cancel₀ hpos.ne', one_mul]

theorem total_expectation (hA : MeasurableSet A) (hF : Integrable F μ)
    (hA0 : μ A ≠ 0) (hAc0 : μ Aᶜ ≠ 0) :
    ∫ ω, F ω ∂μ
      = (μ A).toReal * ∫ ω, F ω ∂μ[|A] + (μ Aᶜ).toReal * ∫ ω, F ω ∂μ[|Aᶜ] := by
  rw [← integral_add_compl hA hF, setIntegral_eq_measure_mul_condIntegral hA0,
    setIntegral_eq_measure_mul_condIntegral hAc0]

theorem measure_compl_toReal (hA : MeasurableSet A) :
    (μ Aᶜ).toReal = 1 - (μ A).toReal := by
  rw [measure_compl hA (measure_ne_top μ A), measure_univ,
    ENNReal.toReal_sub_of_le prob_le_one ENNReal.one_ne_top, ENNReal.toReal_one]

end TotalExpectation

section ExpectedF1

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {F : Ω → ℝ} {Cut : Set Ω}

def cutFraction (μ : Measure Ω) (Cut : Set Ω) : ℝ := (μ Cut).toReal

def crossMean (μ : Measure Ω) (F : Ω → ℝ) (Cut : Set Ω) : ℝ := ∫ ω, F ω ∂μ[|Cut]

def withinMean (μ : Measure Ω) (F : Ω → ℝ) (Cut : Set Ω) : ℝ := ∫ ω, F ω ∂μ[|Cutᶜ]

theorem expected_eq_within_cross (hCut : MeasurableSet Cut) (hF : Integrable F μ)
    (h0 : μ Cut ≠ 0) (h1 : μ Cutᶜ ≠ 0) :
    ∫ ω, F ω ∂μ
      = withinMean μ F Cut * (1 - cutFraction μ Cut)
        + crossMean μ F Cut * cutFraction μ Cut := by
  have h := total_expectation hCut.compl hF h1 (by simpa using h0)
  simp only [compl_compl] at h
  rw [h, measure_compl_toReal hCut]
  simp only [withinMean, crossMean, cutFraction]
  ring

theorem expected_closed_form (hCut : MeasurableSet Cut) (hF : Integrable F μ)
    (h0 : μ Cut ≠ 0) (h1 : μ Cutᶜ ≠ 0) {α : ℝ}
    (hα : crossMean μ F Cut = α * withinMean μ F Cut) :
    ∫ ω, F ω ∂μ
      = withinMean μ F Cut * (1 - (1 - α) * cutFraction μ Cut) := by
  rw [expected_eq_within_cross hCut hF h0 h1, hα]
  ring

theorem expected_alpha_zero (hCut : MeasurableSet Cut) (hF : Integrable F μ)
    (h0 : μ Cut ≠ 0) (h1 : μ Cutᶜ ≠ 0) (hα : crossMean μ F Cut = 0) :
    ∫ ω, F ω ∂μ = withinMean μ F Cut * (1 - cutFraction μ Cut) := by
  have := expected_closed_form (α := 0) hCut hF h0 h1 (by simpa using hα)
  simpa using this

theorem expected_alpha_one (hCut : MeasurableSet Cut) (hF : Integrable F μ)
    (h0 : μ Cut ≠ 0) (h1 : μ Cutᶜ ≠ 0)
    (hα : crossMean μ F Cut = withinMean μ F Cut) :
    ∫ ω, F ω ∂μ = withinMean μ F Cut := by
  have := expected_closed_form (α := 1) hCut hF h0 h1 (by simpa using hα)
  simpa using this

omit [IsProbabilityMeasure μ] in
theorem alpha_nonneg_of_nonneg (hF : ∀ ω, 0 ≤ F ω)
    (hpos : 0 < withinMean μ F Cut) {α : ℝ}
    (hα : crossMean μ F Cut = α * withinMean μ F Cut) :
    0 ≤ α := by
  have hcross : 0 ≤ crossMean μ F Cut := integral_nonneg hF
  rw [hα] at hcross
  exact nonneg_of_mul_nonneg_left hcross hpos

end ExpectedF1

section Sensitivity

def expectedScore (S α δ : ℝ) : ℝ := S * (1 - (1 - α) * δ)

theorem expectedScore_strictAnti (S α : ℝ) (hS : 0 < S) (hα : α < 1) :
    StrictAnti (expectedScore S α) := by
  intro a b hab
  unfold expectedScore
  have : 0 < S * (1 - α) := mul_pos hS (by linarith)
  nlinarith

theorem expectedScore_const (S δ₁ δ₂ : ℝ) :
    expectedScore S 1 δ₁ = expectedScore S 1 δ₂ := by
  simp [expectedScore]

theorem expectedScore_strictMono (S α : ℝ) (hS : 0 < S) (hα : 1 < α) :
    StrictMono (expectedScore S α) := by
  intro a b hab
  unfold expectedScore
  have : 0 < S * (α - 1) := mul_pos hS (by linarith)
  nlinarith

theorem expectedScore_bounds (S α δ : ℝ) (hS : 0 ≤ S) (hδ₀ : 0 ≤ δ) (hδ₁ : δ ≤ 1) :
    S * min 1 α ≤ expectedScore S α δ ∧ expectedScore S α δ ≤ S * max 1 α := by
  unfold expectedScore
  have hconv : S * (1 - (1 - α) * δ) = S * ((1 - δ) * 1 + δ * α) := by ring
  rw [hconv]
  constructor
  · apply mul_le_mul_of_nonneg_left _ hS
    have h1 : min 1 α ≤ 1 := min_le_left _ _
    have h2 : min 1 α ≤ α := min_le_right _ _
    nlinarith
  · apply mul_le_mul_of_nonneg_left _ hS
    have h1 : 1 ≤ max 1 α := le_max_left _ _
    have h2 : α ≤ max 1 α := le_max_right _ _
    nlinarith

end Sensitivity

section CutFraction

variable {E : Type*} [Fintype E] [MeasurableSpace E] [MeasurableSingletonClass E]

theorem uniform_cutFraction (Cut : Set E) [DecidablePred (· ∈ Cut)] :
    cutFraction (uniformOn (Set.univ : Set E)) Cut
      = (Finset.univ.filter (· ∈ Cut)).card / Fintype.card E := by
  unfold cutFraction
  rw [uniformOn_univ, Measure.count_apply_finite' (Set.toFinite Cut)
    (Set.toFinite Cut).measurableSet]
  rw [ENNReal.toReal_div]
  simp

end CutFraction

end PartitioningF1
