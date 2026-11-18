import Mathlib

/-!
# Prequential link prediction: no leakage and valid negatives

Formalizes the protocol of `vertex_voyage.tasks.prequential_link_prediction`
as described in `doc/evaluation.md` (prequential link prediction).

Let `V` be the vertex type. An undirected edge is an element of `Sym2 V`.
The stream is a sequence of buffers `B : ℕ → Set (Sym2 V)` that only add
edges (assumption A4), indexed from `0` as in the code.

* `edgesBefore B t = ⋃ i < t, B i` is the edge set `E_{t-1}` the embedding
  `Z_{t-1}` was trained on.
* `edgesThrough B t = ⋃ i ≤ t, B i` is the snapshot `E_t`.
* The positives of buffer `t` satisfy `P t ⊆ B t \ edgesBefore B t`; the
  code further keeps only edges between known vertices without self-loops,
  which only shrinks `P t`.
* `trainedPositivesBefore P t = ⋃ i < t, P i` are the positives the
  warm-started decoder was trained on before it is tested on buffer `t`.
* `snapshotComplement K E` is the set `S_t` of non-edges of `E` between
  known vertices `K`, without self-pairs; the negatives satisfy
  `N ⊆ snapshotComplement K (edgesThrough B t)`.

Main results:
* `no_leakage` (Lemma L2): the positives tested at buffer `t` are disjoint
  from every edge the embedding saw and from every positive the decoder was
  trained on.
* `negatives_valid` (Lemma L3): negatives are non-edges of the snapshot,
  hence never positives, never self-pairs, and only join known vertices.
-/

namespace PrequentialLinkPrediction

variable {V : Type*}

def edgesBefore (B : ℕ → Set (Sym2 V)) (t : ℕ) : Set (Sym2 V) :=
  ⋃ i, ⋃ (_ : i < t), B i

def edgesThrough (B : ℕ → Set (Sym2 V)) (t : ℕ) : Set (Sym2 V) :=
  ⋃ i, ⋃ (_ : i ≤ t), B i

def trainedPositivesBefore (P : ℕ → Set (Sym2 V)) (t : ℕ) : Set (Sym2 V) :=
  ⋃ i, ⋃ (_ : i < t), P i

def snapshotComplement (K : Set V) (E : Set (Sym2 V)) : Set (Sym2 V) :=
  {e | (∀ v ∈ e, v ∈ K) ∧ ¬ e.IsDiag ∧ e ∉ E}

/-- The positives of every buffer are new edges of that buffer. -/
def PositivesAreNewEdges (B P : ℕ → Set (Sym2 V)) : Prop :=
  ∀ t, P t ⊆ B t \ edgesBefore B t

theorem mem_edgesBefore {B : ℕ → Set (Sym2 V)} {t : ℕ} {e : Sym2 V} :
    e ∈ edgesBefore B t ↔ ∃ i < t, e ∈ B i := by
  simp [edgesBefore]

theorem mem_trainedPositivesBefore {P : ℕ → Set (Sym2 V)} {t : ℕ}
    {e : Sym2 V} :
    e ∈ trainedPositivesBefore P t ↔ ∃ i < t, e ∈ P i := by
  simp [trainedPositivesBefore]

theorem buffer_subset_edgesBefore {B : ℕ → Set (Sym2 V)} {i t : ℕ}
    (hit : i < t) : B i ⊆ edgesBefore B t :=
  fun _ he => mem_edgesBefore.2 ⟨i, hit, he⟩

theorem buffer_subset_edgesThrough (B : ℕ → Set (Sym2 V)) (t : ℕ) :
    B t ⊆ edgesThrough B t := by
  intro e he
  simp only [edgesThrough, Set.mem_iUnion]
  exact ⟨t, le_rfl, he⟩

theorem positives_subset_buffer {B P : ℕ → Set (Sym2 V)}
    (hP : PositivesAreNewEdges B P) (t : ℕ) : P t ⊆ B t :=
  fun _ he => (hP t he).1

theorem positives_subset_edgesThrough {B P : ℕ → Set (Sym2 V)}
    (hP : PositivesAreNewEdges B P) (t : ℕ) : P t ⊆ edgesThrough B t :=
  (positives_subset_buffer hP t).trans (buffer_subset_edgesThrough B t)

/-- A tested positive was not an edge of any earlier buffer, so the
embedding `Z_{t-1}` never saw it. -/
theorem positives_disjoint_edgesBefore {B P : ℕ → Set (Sym2 V)}
    (hP : PositivesAreNewEdges B P) (t : ℕ) :
    Disjoint (P t) (edgesBefore B t) :=
  Set.disjoint_left.2 fun _ he => (hP t he).2

/-- A tested positive was never a training positive of the decoder: every
earlier training positive lies in an earlier buffer. -/
theorem positives_disjoint_trainedPositivesBefore
    {B P : ℕ → Set (Sym2 V)} (hP : PositivesAreNewEdges B P) (t : ℕ) :
    Disjoint (P t) (trainedPositivesBefore P t) := by
  refine Set.disjoint_left.2 fun e he hTrained => ?_
  obtain ⟨i, hit, hei⟩ := mem_trainedPositivesBefore.1 hTrained
  exact (hP t he).2
    (buffer_subset_edgesBefore hit (positives_subset_buffer hP i hei))

/-- **Lemma L2 (no leakage).** -/
theorem no_leakage {B P : ℕ → Set (Sym2 V)}
    (hP : PositivesAreNewEdges B P) (t : ℕ) :
    Disjoint (P t) (edgesBefore B t ∪ trainedPositivesBefore P t) :=
  Disjoint.union_right (positives_disjoint_edgesBefore hP t)
    (positives_disjoint_trainedPositivesBefore hP t)

/-- Positives of two different buffers never coincide. -/
theorem positives_pairwise_disjoint {B P : ℕ → Set (Sym2 V)}
    (hP : PositivesAreNewEdges B P) {i t : ℕ} (hit : i < t) :
    Disjoint (P t) (P i) :=
  (positives_disjoint_trainedPositivesBefore hP t).mono_right
    fun _ he => mem_trainedPositivesBefore.2 ⟨i, hit, he⟩

theorem negatives_disjoint_edges {K : Set V} {E N : Set (Sym2 V)}
    (hN : N ⊆ snapshotComplement K E) : Disjoint N E :=
  Set.disjoint_left.2 fun _ he => (hN he).2.2

theorem negatives_disjoint_positives {K : Set V} {E P N : Set (Sym2 V)}
    (hP : P ⊆ E) (hN : N ⊆ snapshotComplement K E) : Disjoint N P :=
  (negatives_disjoint_edges hN).mono_right hP

/-- **Lemma L3 (valid negatives).** -/
theorem negatives_valid {K : Set V} {E P N : Set (Sym2 V)}
    (hP : P ⊆ E) (hN : N ⊆ snapshotComplement K E) :
    Disjoint N E ∧ Disjoint N P ∧
      ∀ e ∈ N, ¬ e.IsDiag ∧ ∀ v ∈ e, v ∈ K :=
  ⟨negatives_disjoint_edges hN, negatives_disjoint_positives hP hN,
    fun _ he => ⟨(hN he).2.1, (hN he).1⟩⟩

/-- Lemma L3 on the stream: negatives sampled from the complement of the
snapshot `E_t` never contain a positive of buffer `t`. -/
theorem stream_negatives_disjoint_positives {K : Set V}
    {B P : ℕ → Set (Sym2 V)} {N : Set (Sym2 V)}
    (hP : PositivesAreNewEdges B P) (t : ℕ)
    (hN : N ⊆ snapshotComplement K (edgesThrough B t)) :
    Disjoint N (P t) :=
  negatives_disjoint_positives (positives_subset_edgesThrough hP t) hN

end PrequentialLinkPrediction
