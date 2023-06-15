import json
from dataclasses import dataclass
from typing import Union

@dataclass
class Agent:
    """
    A class representing an agent in the chat.
    It has a system prompt, a behavior that defines when it is called
    and how its output is displyed, and sampling parameters.
    """
    name: str
    # file containing the full system prompt to use for the agent
    prompt_path: str
    # the type of the agent: speaker | supervisor | planner | judge
    behavior: str

    # sampling parameters to be sent to the API
    sampling: dict

    @property
    def prompt(self) -> str:
        """
        Returns the text of the agent's system prompt.
        """
        # If we haven"t read the prompt from the file yet, do it now
        if not hasattr(self, "_prompt"):
            with open(self.prompt_path) as f:
                self._prompt = f.read()
        return self._prompt

@dataclass
class AgentList:
    """
    A class representing a list of agents.
    """
    agents: list[Agent]

    def __getitem__(self, i):
        return self.agents[i]

    def __len__(self):
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)

    def get(self, behavior = None, name = None) -> Union[Agent, None]:
        """
        Returns the first agent with the given behavior or name, or None if there is none.
        """
        if behavior is not None:
            return next(self._filter(behavior), None)
        return next(filter(lambda a: a.name == name, self.agents), None)

    def get_all(self, behavior: str) -> list:
        """
        Returns a list of agents with the given behavior.
        """
        return list(self._filter(behavior))

    def _filter(self, behavior: str):
        """
        Returns a generator that yields agents with the given behavior.
        """
        return filter(lambda a: a.behavior == behavior, self.agents)

def from_file(file: str) -> AgentList:
    """
    Reads a JSON file containing a list of agents and returns an AgentList
    """
    with open(file) as f:
        agents_json = json.load(f).get("agents", [])
        # instantiate an Agent dataclass for each one
        return AgentList(agents = [Agent(**conf) for conf in agents_json])
