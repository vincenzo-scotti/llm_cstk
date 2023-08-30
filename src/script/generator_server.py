import sys
from argparse import ArgumentParser, Namespace
import logging

from typing import Dict, Optional

from submodules.llm_cstk.src.llm_cstk.generator import AIAssistant
from llm_cstk.utils.scripting import init_rest_server_environment

import json
from flask import Flask, request, jsonify


# Global varaibles
flask_app: Flask = Flask(__name__)
ai_assistant: Optional[AIAssistant] = None


@flask_app.post("/generate")
def generate():
    params = json.loads(request.data)['params']
    output = ai_assistant.generate(**params).to_json()
    response = jsonify(output)

    return response


@flask_app.post("/generate/candidate_responses/custom_lm")
def generate_candidate_responses():
    params = json.loads(request.data)['params']
    candidates = ai_assistant.generate_candidate_responses_custom_lm(**params).to_json()
    response = jsonify(candidates)

    return response


@flask_app.post("/generate/candidate_responses/llm")
def generate_candidate_responses():
    params = json.loads(request.data)['params']
    candidate = ai_assistant.generate_candidate_responses_llm(**params).to_json()
    response = jsonify(candidate)

    return response


@flask_app.post("/generate/info_extraction")
def generate_doc_analysis_response():
    params = json.loads(request.data)['params']
    output = ai_assistant.info_extraction(**params).to_json()
    response = jsonify(output)

    return response


@flask_app.post("/generate/kb_qa")
def generate_kb_response():
    params = json.loads(request.data)['params']
    output = ai_assistant.kb_qa(**params).to_json()
    response = jsonify(output)

    return response


def main(args: Namespace):
    # Init environment
    configs: Dict = init_rest_server_environment(args.config_file_path)
    # Start Logging info
    logging.info("Generator API script started and configuration file loaded")
    global ai_assistant, flask_app
    # Initialise document retriever
    ai_assistant = AIAssistant.load(configs['generator'])
    # Run FLASK application
    flask_app.run(**configs['flask'])
    # Close script
    logging.info("Stop signal received, closing script")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='text_generator_server',
        description='REST API service for language model-based text generation)'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the generator API."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
