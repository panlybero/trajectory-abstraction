from pyswip import Prolog, Functor, Variable, call

prolog = Prolog()
prolog.consult("ilp/lgg.pl")
# Define the has/2 and next_to/1 functors
has = Functor("has", 2)
next_to = Functor("next_to", 1)

# Define a Variable to hold the result
LGG = Variable()

# Define the states
states = [
    [has("wood", 1), next_to("air")],
    [has("wood", 3), next_to("air")],
    [has("wood", 2), next_to("air")]
]

# Assert the lgg_multiple_states/2 rule to Prolog's database
# prolog.assertz(
#     """lgg_multiple_states([State], State)."""
#     """lgg_multiple_states([H|T], LGG) :-
#        lgg_multiple_states(T, LGG1),
#        lgg(H, LGG1, LGG)."""
# )

# Query Prolog
a = list(prolog.query(f"lgg_multiple_states(states, LGG)", maxresult=1))
print(a)
