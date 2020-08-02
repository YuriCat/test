
Theorem zero: forall a : nat, a = 0 -> a = 0.

Proof.
  intros a.
  SearchPattern(_ -> _).
  apply iff_refl.
Qed.

Theorem MP: forall (P Q : Prop), (P -> Q) -> P -> Q.

Proof.
  intros ab b c.
  apply c.
Qed.

Theorem plus_zero: forall (a b: nat), b = 0 -> a + b = a.

Proof.
  intros x y z.
  rewrite z.
  SearchPattern (_ = _ -> _ = _).
  apply eq_sym.
  SearchPattern (_ = _ + 0).
  apply plus_n_O.
Qed.



