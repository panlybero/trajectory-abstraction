(define (domain gridworld-domain)

    (:requirements :strips :negative-preconditions)


    ; Define the predicates
    (:predicates
        (is_next_to_wood)
        (has_wood)
        (has_planks)
        (has_chair_parts)
        (has_chair)
        (has_decoration)
    )

    ; Define the actions
    (:action approach_wood
        :parameters ()
        :precondition ()
        :effect (and (is_next_to_wood))
    )

    (:action get_wood
        :parameters ()
        :precondition (and (is_next_to_wood))
        :effect (and (has_wood))
    )

    (:action craft_planks
        :parameters ()
        :precondition (and (has_wood))
        :effect (and (not (has_wood)) (has_planks))
    )

    (:action craft_chair_parts
        :parameters ()
        :precondition (and (has_wood) (has_planks))
        :effect (and (not (has_wood)) (not (has_planks)) (has_chair_parts))
    )

    (:action craft_chair
        :parameters ()
        :precondition (and (has_chair_parts))
        :effect (and (not (has_chair_parts)) (has_chair))
    )

    (:action craft_decoration
        :parameters ()
        :precondition (and (has_planks))
        :effect (and (not (has_planks)) (has_decoration))
    )
)
