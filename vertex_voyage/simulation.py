from vertex_voyage.model import BaseModel, get_model_info, construct_model
from typing import List, Dict
from vertex_voyage.data_source import DataSource, load_data_source
class Simulation:
    def __init__(self, name: str = None, steps: List[BaseModel] = None, data_sources: Dict[str, DataSource] = None, params: List[Dict[str, any]] = None, input: DataSource = None):
        if steps is None:
            steps = []
        self.steps = steps
        if data_sources is None:
            data_sources = {}
        self.data_sources = data_sources
        self.name = name
        if params is None:
            params = []
        self.params = params
        self.input = input
    
    def add_step(self, step: BaseModel):
        self.steps.append(step)

    def add_data_source(self, name: str, data_source: DataSource):
        self.data_sources[name] = data_source
    
    def get_non_ready_steps(self):
        return [step for step in self.steps if not step.ready()]
    
    def get_runnable_steps(self):
        return [step for step in self.steps if step.ready()]
    
    def runnable(self):
        return len(self.get_non_ready_steps()) == 0
    
    def run_until(self, step_index: int):
        x = self.input
        for i, step in enumerate(self.steps):
            if i == step_index:
                break
            if step.ready():
                x = step.run(x)
                if x is None:
                    raise ValueError(f"Step {i} did not return anything.")
            else:
                raise ValueError(f"Step {i} is not ready to run.")
        return x
    
    def run(self):
        return self.run_until(len(self.steps))


def get_simulation_info(simulation: Simulation):
    return {
        "steps": [get_model_info(step) for step in simulation.steps],
        "data_sources": [
            {
                "name": name,
                "data_type": data_source.get_data_type(),
                "key": data_source.key(),
                "params": data_source.to_dict(),
                "class": data_source.__class__.__name__
            } for name, data_source in simulation.data_sources.items()
        ], 
        "params": simulation.params,
        "input": {
            "data_type": simulation.input.get_data_type(),
            "key": simulation.input.key(),
            "params": simulation.input.to_dict(),
            "class": simulation.input.__class__.__name__
        }
    }

def load_simulation(params: Dict[str, any]):
    simulation = Simulation(
        name=params.get("name", None),
        input=load_data_source(params["input"]["class"], params["input"]["params"]),
        params=params.get("params", [])
    )
    for step in params.get("steps", []):
        simulation.add_step(construct_model(step["class"], step["actions"]))
    for data_source in params.get("data_sources", []):
        simulation.add_data_source(data_source["name"], load_data_source(data_source["class"], data_source["params"]))
    return simulation

def run_simulation(simulation: Simulation):
    if not simulation.runnable():
        raise ValueError("Simulation is not ready to run.")
    for step in simulation.get_runnable_steps():
        step.run()

def run_step(simularion: Simulation) -> Simulation:
    step = simularion.steps[0]
    if not step.ready():
        raise ValueError("Step is not ready to run.")
    x = step.run(simularion.input)
    return Simulation(
        name=simularion.name,
        steps=simularion.steps[1:],
        data_sources=simularion.data_sources,
        input=x,
        params=simularion.params[1:],
    )

def prepare_step(simulation: Simulation) -> Simulation:
    step: BaseModel
    step, params = next([(step, params) for step, params in zip(simulation.steps, simulation.params) if not step.ready()], (None, None))
    if step is None:
        return simulation
    if step.valid():
        step.fit(**params)
    return Simulation(
        data_sources=simulation.data_sources,
        input=simulation.input,
        name=simulation.name,
        steps=simulation.steps,
        params=simulation.params
    )