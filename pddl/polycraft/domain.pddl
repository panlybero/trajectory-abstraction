;Header and description

(define (domain polycraft_generated)

;remove requirements that are not needed
(:requirements :typing :strips :fluents :negative-preconditions :equality)

(:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    pickaxe_breakable - breakable
    hand_breakable - pickaxe_breakable
    breakable - placeable
    placeable - physobj
    physobj - physical
    actor - physobj
    trader - actor
    pogoist - actor
    agent - actor
    oak_log - log
    distance - var
    agent - placeable
    bedrock - placeable
    oak_log - hand_breakable
    sapling - placeable
    planks - physobj
    pogo_stick - physobj
    blue_key - physobj
)

(:constants 
    air - physobj
    one - distance
    two - distance
    rubber - physobj
    blue_key - physobj
)

(:predicates ;todo: define predicates here
    (holding ?v0 - physobj)
    (floating ?v0 - physobj)
    (facing_obj ?v0 - physobj ?d - distance)
    (next_to ?v0 - physobj ?v1 - physobj)
)


(:functions ;todo: define numeric functions here
    (world ?v0 - physobj)
    (inventory ?v0 - physobj)
    (container ?v0 - physobj ?v1 - physobj)
)

; define actions here
(:action approach
    :parameters    (?physobj01 - physobj ?physobj02 - physobj )
    :precondition  (and
        (>= ( world ?physobj02) 1)
        (facing_obj ?physobj01 one)
    )
    :effect  (and
        (facing_obj ?physobj02 one)
        (not (facing_obj ?physobj01 one))
    )
)


(:action break
    :parameters    (?physobj - hand_breakable)
    :precondition  (and
        (facing_obj ?physobj one)
        (not (floating ?physobj))
    )
    :effect  (and
        (facing_obj air one)
        (not (facing_obj ?physobj one))
        (increase ( inventory ?physobj) 1)
        (increase ( world air) 1)
        (decrease ( world ?physobj) 1)
    )
)


; additional actions, including craft and trade
(:action craft_pogo_stick
    :parameters ()
    :precondition (and
        (>= ( inventory planks) 2)
    )
    :effect (and
        (decrease ( inventory planks) 2)
        (increase ( inventory pogo_stick) 1)
    )
)


(:action craft_planks
    :parameters ()
    :precondition (and
        (>= ( inventory oak_log) 1)
    )
    :effect (and
        (decrease ( inventory oak_log) 1)
        (increase ( inventory planks) 4)
    )
)

)
