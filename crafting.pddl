;Headernot and description

(define (domain polycraft_generated)

;remove requirements that are not needed
(:requirements :typing :strips)

(:types 
    breakable - placeable
    placeable - physobj
    physobj - physical
    actor - physobj
    agent - actor
    oak_log - log
    agent - placeable
    bedrock - placeable
    oak_log - physobj
    sapling - placeable
    planks - physobj
    pogo_stick - physobj
)

(:constants 
    air - physobj
    planks - physobj
    oak_log - physobj
    sapling - physobj
    pogo_stick - physobj
)

(:predicates ;todo: define predicates here
    (next_to ?v0 - physobj)
    (available ?v0 - physobj)
    (has ?v0 - physobj)
    
)



; define actions here
(:action approach
    :parameters    (?physobj01 - physobj ?physobj02 - physobj )
    :precondition  (and
        (available ?physobj02)
        (next_to ?physobj01)
    )
    :effect  (and
        (next_to ?physobj02)
        (not (next_to ?physobj01))
    )
)


(:action break
    :parameters    (?physobj - physobj)
    :precondition  (and
        (next_to ?physobj)
        (available ?physobj)
    )
    :effect  (and
        (next_to air)
        (not (next_to ?physobj))
        (not (available ?physobj))
        (has ?physobj)
    )
)


; additional actions, including craft and trade
(:action craft_pogo_stick
    :parameters ()
    :precondition (and
        (has planks)
    )
    :effect (and
        (not (has planks))
        (has pogo_stick)
    )
)


(:action craft_planks
    :parameters ()
    :precondition (and (has oak_log))
    
    :effect (and
        (not (has oak_log))
        (has planks)
    )
)

)
