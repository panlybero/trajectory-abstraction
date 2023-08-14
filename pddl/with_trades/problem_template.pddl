(define
	(problem polycraft_problem)
	(:domain polycraft_generated)
    (:objects 
        agent - agent
        bedrock - bedrock
    )

    (:init
        (available air)
        (available bedrock)
        (available  trader)
        ;(not (available sapling))
        ;(not (has agent))
        ;(not (has bedrock) )
        ;(not (has wood) )
        ;(not (has sapling) )
        ;(not (has planks) )
        ;(not (has chair) )
        ;(not (has chair_parts) )
        ;(not (has decoration) )
        ;(next_to air)
        <INIT>
    )

    (:goal <GOAL>) 
)
