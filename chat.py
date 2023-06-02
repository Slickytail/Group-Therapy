import openai
import json


if __name__ == '__main__':
    # read the API key from the file
    with open('config.json') as f:
        config = json.load(f)
        openai.api_key = config['api_key']
        # some openai keys are organization specific
        # check if the config contains the organization key
        if 'organization' in config:
            config.organization = config['organization']

