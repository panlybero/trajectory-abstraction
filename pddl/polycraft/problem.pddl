(define
	(problem polycraft_problem)
	(:domain polycraft_generated)
    (:objects 
        agent - agent
        bedrock - bedrock
        oak_log - oak_log
        sapling - sapling
        planks - planks
        pogo_stick - pogo_stick
    )

    (:init
        (= (world air) 7)
        (= (world bedrock) 16)
        (= (world oak_log) 1)
        (= (world agent) 0)
        (= (world sapling) 0)
        (= (inventory agent) 0)
        (= (inventory bedrock) 0)
        (= (inventory oak_log) 0)
        (= (inventory sapling) 0)
        (= (inventory planks) 0)
        (= (inventory pogo_stick) 0)
        (facing_obj air one)
        (holding air)
    )

    (:goal (>= (inventory pogo_stick) 1))
)
