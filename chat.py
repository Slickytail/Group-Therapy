import openai
import json
from dataclasses import dataclass


MAX_TOKENS = 1024

@dataclass
class Agent:
    """
    A class representing an agent in the chat.
    It has a system prompt, a behavior that defines when it is called
    and how its output is displyed, and sampling parameters.
    """
    name: str
    # the full system prompt to use for the agent
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
        Returns the prompt to be sent to the API.
        """
        # If we haven"t read the prompt from the file yet, do it now
        if not hasattr(self, "_prompt"):
            with open(self.prompt_path) as f:
                self._prompt = f.read()
        return self._prompt


def chat_loop(speaker, planner = None, judge = None):
    """
    The main chat loop. It will keep running until KeyboardInterrupt is raised.
    """
    messages = []
    while True:
        # start by getting input from the user
        user_message = input(f">>> : ")
        messages.append({"role": "user", "content": user_message})

        # get the next batch of messages from the speaker
        suggestions = chat_step(messages, speaker)
        # if there's a judge, get their input
        if len(suggestions) > 1 and judge is not None:
            judge_history = messages + [{"role": "system",
                                        "content": "\n".join(f"(Therapist {i} \n {s}" for i, s in enumerate(suggestions))}]
            judgement = chat_step(judge_history, judge)[0]
            try:
                judgement_json = json.loads("{" + judgement.split("{", 1)[1])
                
                message = suggestions[int(judgement_json.get("choice", 0))]
            except Exception as e:
                message = suggestions[0]
        else:
            message = suggestions[0]
        # print the message
        print(f"\n{speaker.name} : {message} \n")
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
        n = 3
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
        agents = [Agent(**agent) for agent in agents_json]

    # just for testing, we"ll pick the first speaker and no advisors
    agent = next(filter(lambda a: a.behavior == "speaker", agents))
    judge = next(filter(lambda a: a.behavior == "judge", agents))
    chat_loop(speaker=agent, judge=judge)
    
