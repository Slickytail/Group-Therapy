import openai
import json
from itertools import chain
from dataclasses import dataclass
from rich import print

MAX_TOKENS = 1024

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
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

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


def chat_loop(speakers, planner = None, judge = None):
    """
    The main chat loop. It will keep running until KeyboardInterrupt is raised.
    """
    messages = []
    while True:
        # start by getting input from the user
        # because we are using rich, we have to print first, and then call input()
        print(f"[green] >>> [/green][grey85]:[/grey85]", end=" ")
        user_message = input()

        messages.append({"role": "user", "content": user_message})

        # Speaker Pass: get batches of messages from the speakers
        # and flatten the list of lists into a single list
        suggestions = list(chain(*(chat_step(messages, speaker) for speaker in speakers)))
        # TODO: if no judge, only do one speaker.

        # Judge Pass: pick between a batch of messages
        if len(suggestions) > 1 and judge is not None:
            # judge read the entire history, plus a system message containing the proposed messages
            judge_history = messages + [{"role": "system",
                                         "content": "\n".join(f"(Therapist {i}) \n {s}" for i, s in enumerate(suggestions))
                                        }]
            judgement = chat_step(judge_history, judge)[0]
            try:
                # The judge might write some notes before it outputs its json.
                # so we split on { and then grab everything after that.
                judgement_json = json.loads("{" + judgement.split("{", 1)[1])
                
                message = suggestions[int(judgement_json.get("choice", 0))]
            # errors that can happen here:
            #   - indexerror: the judge chose a message that didn't exist
            #   - json error: the judge didn't write valid json
            # if either of these happens, 
            except:
                print("[red] Judge : Error [/red]")
                message = suggestions[0]
        else:
            message = suggestions[0]
        # pretty-print the message
        print(f"\n[green] Helper [/green][grey85]:[/grey85] {message} \n")
        # add the message to the history
        messages.append({"role": "assistant", "content": message})


def chat_step(history, agent):
    """
    Call the OpenAI API with the given history and agent.
    """
    # put the current agent"s system prompt first
    messages = [{"role": "system", "content": agent.prompt}] + history
    # Call the ChatComplete endpoint
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = messages,
        max_tokens = MAX_TOKENS,
        # sampling behavior from the agent
        temperature = agent.temperature,
        top_p = agent.top_p,
        frequency_penalty = agent.frequency_penalty,
        presence_penalty = agent.presence_penalty,
        # todo: use streaming
        n = 2
    )

    return list(m["message"]["content"] for m in response["choices"])


if __name__ == "__main__":

    # read the API key from the file
    with open("config.json") as f:
        config = json.load(f)
        openai.api_key = config["api_key"]
        # some openai keys are organization specific
        # check if the config contains the organization key
        if "organization" in config:
            openai.organization = config["organization"]

        MAX_TOKENS = config.get("max_tokens", MAX_TOKENS)

    # read the agent config
    with open("agents.json") as f:
        agents_json = json.load(f).get("agents", [])
        # instantiate an Agent dataclass for each one
        agents = [Agent(**agent) for agent in agents_json]

    # identify different kinds of agents
    speakers = list(filter(lambda a: a.behavior == "speaker", agents))
    judge = next(filter(lambda a: a.behavior == "judge", agents))
    chat_loop(speakers, judge=judge)
    
