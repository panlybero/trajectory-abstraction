:- use_module(library(lists)).

lgg(X, Y, Z) :- 
    var(X), !, 
    X = Y, 
    Z = Y.
lgg(X, Y, Z) :- 
    var(Y), !, 
    X = Y, 
    Z = X.
lgg(X, X, X).

% Handling range for numbers
lgg(X, Y, [X, Y]) :-
    number(X), 
    number(Y), 
    X \= Y.

lgg([X1, Y1], Y, [X, Y]) :-
    number(Y), 
    X is min(X1, Y), 
    Y is max(Y1, Y).
    
lgg(X, [X2, Y2], [X, Y]) :-
    number(X), 
    X is min(X, X2), 
    Y is max(X, Y2).
    
lgg([X1, Y1], [X2, Y2], [X, Y]) :-
    X is min(X1, X2), 
    Y is max(Y1, Y2).

% Handling general case
lgg(X, Y, Z) :-
    compound(X), 
    compound(Y), 
    compound_name_arguments(X, Name, ArgsX),
    compound_name_arguments(Y, Name, ArgsY),
    maplist(lgg, ArgsX, ArgsY, ArgsZ),
    compound_name_arguments(Z, Name, ArgsZ).

% A rule to determine the lgg of multiple states.
lgg_multiple_states([State], State).
lgg_multiple_states([H|T], LGG) :-
    lgg_multiple_states(T, LGG1),
    lgg(H, LGG1, LGG).
