Definition prop0 : forall (A : Prop), A -> A :=
  fun A x => x.

Definition prop1 : forall (A B C : Prop), (B -> C) -> (A -> B) -> (A -> C) :=
  fun A B C f g x => f (g x).