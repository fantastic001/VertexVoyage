import Mathlib

/-!
# Range of the partition balance metric

Formalizes the balance metric of `doc/evaluation.md`, as computed by
`vertex_voyage.partitioning.get_partition_average_balance`:
for partition sizes `s : Fin K → ℝ` with mean `m = (∑ s) / K`,

  `β(s) = (1 / K) * ∑ |s i - m| / m`.

The README tables report `B = 1 + β`.

Main results:
* `balance_nonneg`: `0 ≤ β`.
* `balance_le`: `β ≤ 2 - 2 / K`, hence `B ≤ 3 - 2 / K`.
* `balance_of_single_partition`: the bound is attained when every vertex
  sits in one partition, e.g. `β = 1` for `K = 2` and `β = 3/2` for `K = 4`.
-/

open Finset

namespace VertexVoyageProofs.PartitionBalance

noncomputable def balance {K : ℕ} (s : Fin K → ℝ) : ℝ :=
  let m := (∑ i, s i) / K
  (∑ i, |s i - m| / m) / K

theorem abs_sub_eq_add_sub_two_min (x m : ℝ) :
    |x - m| = x + m - 2 * min x m := by
  rcases le_total x m with h | h
  · rw [min_eq_left h, abs_of_nonpos (by linarith)]; ring
  · rw [min_eq_right h, abs_of_nonneg (by linarith)]; ring

theorem le_sum_min {ι : Type*} (t : Finset ι) (s : ι → ℝ) (m : ℝ)
    (hs : ∀ i ∈ t, 0 ≤ s i) (hm0 : 0 ≤ m) (hm : m ≤ ∑ i ∈ t, s i) :
    m ≤ ∑ i ∈ t, min (s i) m := by
  by_cases h : ∃ i ∈ t, m ≤ s i
  · obtain ⟨i, hi, hle⟩ := h
    calc m = min (s i) m := (min_eq_right hle).symm
      _ ≤ ∑ j ∈ t, min (s j) m :=
        single_le_sum (f := fun j => min (s j) m)
          (fun j hj => le_min (hs j hj) hm0) hi
  · push Not at h
    calc m ≤ ∑ i ∈ t, s i := hm
      _ = ∑ i ∈ t, min (s i) m :=
        sum_congr rfl (fun i hi => (min_eq_left (h i hi).le).symm)

theorem balance_eq_sum_abs_div_total {K : ℕ} (hK : 0 < K) (s : Fin K → ℝ)
    (hS : 0 < ∑ i, s i) :
    balance s = (∑ i, |s i - (∑ j, s j) / K|) / ∑ i, s i := by
  have hKr : (0 : ℝ) < K := by exact_mod_cast hK
  unfold balance
  simp only
  rw [← sum_div]
  field_simp
  exact sum_congr rfl (fun x _ => by ring_nf)

theorem balance_nonneg {K : ℕ} (hK : 0 < K) (s : Fin K → ℝ)
    (hS : 0 < ∑ i, s i) : 0 ≤ balance s := by
  rw [balance_eq_sum_abs_div_total hK s hS]
  exact div_nonneg (sum_nonneg (fun i _ => abs_nonneg _)) hS.le

theorem balance_le {K : ℕ} (hK : 0 < K) (s : Fin K → ℝ)
    (hs : ∀ i, 0 ≤ s i) (hS : 0 < ∑ i, s i) :
    balance s ≤ 2 - 2 / K := by
  set S := ∑ i, s i with hSdef
  set m := S / K with hmdef
  have hKr : (0 : ℝ) < K := by exact_mod_cast hK
  have hK1 : (1 : ℝ) ≤ K := by exact_mod_cast hK
  have hm0 : 0 ≤ m := div_nonneg hS.le hKr.le
  have hmS : m ≤ S := div_le_self hS.le hK1
  have hKm : (K : ℝ) * m = S := by rw [hmdef]; field_simp
  have hsum : ∑ i, |s i - m| = 2 * S - 2 * ∑ i, min (s i) m := by
    simp_rw [abs_sub_eq_add_sub_two_min]
    rw [sum_sub_distrib, sum_add_distrib, ← mul_sum]
    simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [hKm]; ring
  have hmin : m ≤ ∑ i, min (s i) m :=
    le_sum_min univ s m (fun i _ => hs i) hm0 hmS
  rw [balance_eq_sum_abs_div_total hK s hS, ← hSdef, ← hmdef, hsum,
    div_le_iff₀ hS]
  have : 2 / (K : ℝ) * S = 2 * m := by rw [hmdef]; ring
  nlinarith [this]

theorem balance_of_single_partition {K : ℕ} (hK : 0 < K) (c : ℝ) (hc : 0 < c)
    (j : Fin K) :
    balance (fun i => if i = j then c else 0) = 2 - 2 / K := by
  have hKr : (0 : ℝ) < K := by exact_mod_cast hK
  have hS : ∑ i : Fin K, (if i = j then c else 0) = c := by simp
  rw [balance_eq_sum_abs_div_total hK _ (by rw [hS]; exact hc), hS]
  have hm : c / K ≤ c := div_le_self hc.le (by exact_mod_cast hK)
  have hterm : ∀ i : Fin K, |(if i = j then c else 0) - c / K|
      = c / K + (if i = j then c - 2 * (c / K) else 0) := by
    intro i
    split_ifs
    · rw [abs_of_nonneg (by linarith)]; ring
    · rw [zero_sub, abs_neg, abs_of_nonneg (div_nonneg hc.le hKr.le)]; ring
  simp_rw [hterm]
  rw [sum_add_distrib, sum_ite_eq' univ j, if_pos (mem_univ j)]
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

end VertexVoyageProofs.PartitionBalance
