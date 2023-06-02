# Group Therapy

A Python script to orchestrate a group of multiple ChatGPT instances.

#### Requirements:

The `openai` python library.

#### Config:

Create a file called `config.json` in the root directory. Example config:

```json
{
    "organization": "org-...",
    "api_key": "..."
}
```

#### Why?

ChatGPT is good at conversation and reasoning, but it's very bad at planning.
We can get around this by spawning multiple instances with different system prompts. 
For example, you might have one instance that converses with the user,
and a second "supervisor" instance which reads the chat and writes notes, directions, and plans for the conversation instance.
