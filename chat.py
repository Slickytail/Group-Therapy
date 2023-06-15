import openai
import json
from itertools import chain
from rich import print

from therapy.agent import from_file as agents_from_file
from therapy.util import *

MAX_TOKENS = 1024
VERBOSE = False

def chat_loop(speakers, planner = None, judge = None):
    """
    The main chat loop. It will keep running until KeyboardInterrupt is raised.
    """
    messages = []
    while True:
        # start by getting input from the user
        # flush stdin, otherwise any accidental newlines etc could end up sending the messagebefore anything is typed.
        flush_input()
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
                                         "content": "\n".join(f"(Therapist {i}) \n{s}" for i, s in enumerate(suggestions))
                                        }]
            judgement = chat_step(judge_history, judge)[0]
            try:
                # Decode the JSON part of the judge's message
                judgement_json = safe_json_parse(judgement)
                msg_index = int(judgement_json.get("choice", 0))
                message = suggestions[msg_index]
                # for debugging and analysis, print the name of the agent that generated the chosen message
                if VERBOSE:
                    print(f"[red] Judge : Chose {speakers[msg_index // 2].name} [/red]")
            # errors that can happen here:
            #   - indexerror: the judge chose a message that didn't exist
            #   - json error: the judge didn't write valid json
            # if either of these happens, give some debug output and just pick the first message
            except Exception as e:
                print(f"[red] Judge : {e} [/red]")
                if VERBOSE:
                    print(f"[red]       : judgement [/red]")
                message = suggestions[0]
        else:
            message = suggestions[0]

        # pretty-print the message
        print(f"\n[on bright_black][green] Helper [/green][grey85]:[/grey85] [white on bright_black]{message}[/on bright_black] \n")
        # add the message to the history
        messages.append({"role": "assistant", "content": message})


def chat_step(history, agent, n = 1):
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
        n = n
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

        # other misc config
        MAX_TOKENS = config.get("max_tokens", MAX_TOKENS)
        VERBOSE = config.get("verbose", VERBOSE)

    # read the agent config
    agents = agents_from_file("agents.json")

    # identify different kinds of agents
    speakers = agents.get_all("speaker")
    judge = agents.get("judge")
    chat_loop(speakers, judge=judge)
    
