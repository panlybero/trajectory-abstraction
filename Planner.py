import os
import subprocess
import tempfile
import io

# class Pyperplanner:
#     def __init__(self, path, domain_file='crafting.pddl'):
#         self.domain_file = os.path.join(path,domain_file)
#         self.problem_template = open(os.path.join(path,'problem_template.pddl')).read()
    
#     def plan(self, goal_state):




class Planner:
    def __init__(self, path, domain_file='crafting.pddl', exec_path = 'ff'):
        self.domain_file = os.path.join(path,domain_file)
        self.problem_template = open(os.path.join(path,'problem_template.pddl')).read()
        self.exec_path = exec_path
    def plan(self,init_state, goal_state):
        # Fill in the problem template with the goal state
        problem_content = self.problem_template.replace('<GOAL>', goal_state)
        problem_content = problem_content.replace('<INIT>', init_state)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        #with open('problem.pddl', 'w') as f:
            f.write(problem_content)
            problem_file = f.name
            
        cmd = [self.exec_path, '-o', self.domain_file, '-f',problem_file ]
        
        try:
            process_out = subprocess.check_output(cmd, timeout=1).decode('utf-8').splitlines()
        except subprocess.CalledProcessError as e:
            print(problem_content)
            print(e.output)
            return []
        os.remove(problem_file)

        #print(process_out)
        
        plan = self.extract_plan(process_out)

        return plan

    @staticmethod
    def extract_plan(lines):
        
        plan_start = False
        plan = []

        for line in lines:
            if line.startswith('ff: found legal plan'):
                plan_start = True
                continue
            
            if not plan_start:
                continue
            
            if line.startswith('time spent:'):
                break
            plan.append(line.strip())

        if not plan_start:
            raise ValueError("No plan found")
        
        plan = list(filter(lambda x: not x.startswith(';') and not len(x)==0 and not x=='step', plan))
        
        plan = [a.split(":")[1][1:] for a in plan]
        
        return plan


if __name__ == '__main__':

    planner = Planner(exec_path='/home/plymper/trajectory-abstraction/pddlgym_planners/FF-v2.3/ff', path='pddl', domain_file='crafting.pddl')
    plan = planner.plan('(and (has wood))')
    