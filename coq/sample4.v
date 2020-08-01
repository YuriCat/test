
Fixpoint f (x y : nat) := pair x y.

Theorem f_eq : forall (x y : nat), pair x y = f x y.

Proof.
  intros x y.
  induction x as [| i].
  - simpl.
    reflexivity.
  - simpl.
    reflexivity.
Qed.


Fixpoint add (x y : nat) := x + y.

Theorem add_eq : forall (x y : nat), x + y = add x y.

Proof.
  intros x y.
  induction x as [| i].
  - simpl.
    reflexivity.
  - simpl.
    reflexivity.
Qed.
