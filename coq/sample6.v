Require Import Coq.Init.Nat.

Compute orb true false.
Compute andb true false.

(*n以下の自然数に0があることの証明*)

Fixpoint contains_eq0 (n : nat) :=
  match n with
  | 0 => eqb n 0
  | S n' => orb (eqb n 0) (contains_eq0 n')
  end.

Compute contains_eq0 0.
Compute contains_eq0 100.


Theorem contains_eq0_true:
  forall n : nat, contains_eq0 n = true.

Proof.
  induction n.
  - simpl.
    reflexivity.
  - simpl.
    apply IHn.
Qed.

(*n以下の自然数に1があることの証明*)

Fixpoint contains_eq1 (n : nat) :=
  match n with
  | 0 => eqb n 1
  | S n' => orb (eqb n 1) (contains_eq1 n')
  end.

Compute contains_eq1 0.
Compute contains_eq1 100.

Theorem contains_eq1_true:
  forall n : nat, ltb 0 n = contains_eq1 n.

Proof.
  induction n.
  - simpl.
    reflexivity.
  - induction n.
    -- simpl.
       reflexivity.
    -- simpl.
       apply IHn.
Qed.


(*n以下の自然数にn+1が無いことの証明*)

Fixpoint contains_eqnp1 (n : nat) :=
  (fix contains_eq (n m : nat) :=
     match n with
  | 0 => eqb n m
  | S n' => orb (eqb n m) (contains_eq n' m)
  end) n (n + 1).

Compute contains_eqnp1 0.
Compute contains_eqnp1 100.

Theorem contains_eq1_false:
  forall n : nat, contains_eqnp1 n = false.

Proof.
  induction n.
  - simpl.
    reflexivity.
  - unfold contains_eqnp1.
    simpl.
    unfold contains_eq.



Fixpoint contains_eq (n m : nat) :=
  match n with
  | 0 => eqb n m
  | S n' => orb (eqb n m) (contains_eq n' m)
  end.


Lemma contains_eq_equiv1:
  forall n m : nat, m <= n -> (contains_eq n m) = true.

Proof.
  intros n m x.
  induction n.
  - simpl.
    
    induction m.
    reflexivity.
  - 

Lemma ltb_concat:
  forall (n m : nat), ((n =? m) || (S m <=? n))%bool = (m <=? n).


Proof.
  induction m.
  - simpl.
    induction n.
    -- simpl.
       reflexivity.
    -- simpl.
       reflexivity.
  -
    induction n.
    -- simpl.
       reflexivity.
    -- simpl.
       



Lemma contains_eq_condition:
  forall (n m : nat), contains_eq n m = leb m n.


Proof.
  induction n.
  - simpl.
    induction m.
    -- simpl.
       reflexivity.
    -- simpl.
       reflexivity.
  - 
    induction m.
    -- simpl.
       apply IHn.
    -- simpl.
       rewrite IHn.
       

Fixpoint contains_eqnp1 (n : nat) :=
  contains_eq n (n + 1).

Theorem contains_eq1_false:
  forall n : nat, contains_eqnp1 n = false.

Compute contains_eqnp1 0.
Compute contains_eqnp1 100.

Proof.
  induction n.
  - simpl.
    reflexivity.
  - unfold contains_eqnp1.
    simpl.
    unfold contains_eq.