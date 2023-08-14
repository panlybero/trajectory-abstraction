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
    agent - placeable
    bedrock - placeable
    wood - physobj
    chair_parts - physobj
    sapling - placeable
    planks - physobj
    chair - physobj
    decoration - physobj
    stick - physobj
    
)

(:constants 
    air - physobj
    planks - physobj
    wood - physobj
    sapling - physobj
    chair - physobj
    chair_parts - physobj
    decoration - physobj
    stick - physobj
    trader - physobj

)

(:predicates ;todo: define predicates here
    (next_to ?v0 - physobj)
    (available ?v0 - physobj)
    (has ?v0 - physobj)
    
)



; define actions here
(:action approach
    :parameters    (?physobj01 - physobj)
    :precondition  (and
        (available ?physobj01)
        
    )
    :effect  (and
        (next_to ?physobj01)
    )
)


(:action break
    :parameters    (?physobj - physobj)
    :precondition  (and
        (next_to ?physobj)
        (available ?physobj)
    )
    :effect  (and
        (not (next_to ?physobj))    
        (has ?physobj)
    )
)


; additional actions, including craft and trade
(:action craft_chair
    :parameters ()
    :precondition (and
        (has chair_parts)
    )
    :effect (and
        (not (has chair_parts))
        (has chair)
    )
)


(:action craft_planks
    :parameters ()
    :precondition (and (has wood))
    
    :effect (and
        (not (has wood))
        (has planks)
    )
)

(:action craft_decoration
    :parameters ()
    :precondition (and (has planks))
    
    :effect (and
        (has decoration)
        (not (has planks))
    )
)
(:action craft_stick
    :parameters ()
    :precondition (and (has planks))
    
    :effect (and
        (has stick)
        (not (has planks))
    )
)



(:action craft_chair_parts
    :parameters ()
    :precondition (and (has planks) (has wood))
    
    :effect (and
        (has chair_parts)
        (not (has planks))
        (not (has wood))
    )
)

)
