import sys
from argparse import ArgumentParser, Namespace
import logging

from typing import Dict, Optional

import pandas as pd

from llm_cstk.retrieval import DocRetriever
from llm_cstk.utils.scripting import init_rest_server_environment

import json
from flask import Flask, request, jsonify


# Global varaibles
flask_app: Flask = Flask(__name__)
doc_retriever: Optional[DocRetriever] = None


@flask_app.post("/search")
def search():
    params = json.loads(request.data)['params']
    results = doc_retriever.search(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/snippet")
def snippet():
    params = json.loads(request.data)['params']
    params['search_results'] = pd.DataFrame(params['search_results'])
    results = doc_retriever.snippet(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/corpus")
def corpus():
    params = json.loads(request.data)['params']
    params['docs'] = pd.DataFrame(params['docs'])
    if ['chunked_docs'] in params:
        params['chunked_docs'] = pd.DataFrame(params['chunked_docs'])
    doc_retriever.corpus(**params).to_dict(orient='list')


@flask_app.post("/score/query_doc_pair")
def score_query_doc_pair():
    params = json.loads(request.data)['params']
    results = doc_retriever.score_query_doc_pair(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/search/doc")
def search_doc():
    params = json.loads(request.data)['params']
    results = doc_retriever.search_doc(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/search/doc_chunk")
def search_doc_chunk():
    params = json.loads(request.data)['params']
    results = doc_retriever.search_doc_chunk(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/search/doc_long_query")
def search_doc_long_query():
    params = json.loads(request.data)['params']
    results = doc_retriever.search_doc_long_query(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/search/doc_chunk_long_query")
def search_doc_chunk_long_query():
    params = json.loads(request.data)['params']
    results = doc_retriever.search_doc_chunk_long_query(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/snippet/generate")
def generate_snippet():
    params = json.loads(request.data)['params']
    params['search_results'] = pd.DataFrame(params['search_results'])
    results = doc_retriever.generate_snippet(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/snippet/generate_long_query")
def generate_snippet_long_query():
    params = json.loads(request.data)['params']
    params['search_results'] = pd.DataFrame(params['search_results'])
    results = doc_retriever.generate_snippet_long_query(**params).to_dict(orient='list')
    response = jsonify(results)

    return response


@flask_app.post("/corpus/add")
def add_corpus():
    params = json.loads(request.data)['params']
    params['docs'] = pd.DataFrame(params['docs'])
    if ['chunked_docs'] in params:
        params['chunked_docs'] = pd.DataFrame(params['chunked_docs'])
    doc_retriever.add_corpus(**params).to_dict(orient='list')


@flask_app.post("/corpus/add_large")
def add_large_corpus():
    # TODO update to work with dask data frames
    params = json.loads(request.data)['params']
    params['docs'] = pd.DataFrame(params['docs'])
    if ['chunked_docs'] in params:
        params['chunked_docs'] = pd.DataFrame(params['chunked_docs'])
    doc_retriever.add_large_corpus(**params).to_dict(orient='list')


@flask_app.post("/corpus/index")
def index_corpus():
    params = json.loads(request.data)['params']
    doc_retriever.index_corpus(**params).to_dict(orient='list')


@flask_app.post("/corpus/index_large")
def index_large_corpus():
    params = json.loads(request.data)['params']
    doc_retriever.index_large_corpus(**params).to_dict(orient='list')


def main(args: Namespace):
    # Init environment
    configs: Dict = init_rest_server_environment(args.config_file_path)
    # Start Logging info
    logging.info("Retrieval API script started and configuration file loaded")
    global doc_retriever, flask_app
    # Initialise document retriever
    doc_retriever = DocRetriever.load(**configs['retriever'])
    # Run FLASK application
    flask_app.run(**configs['flask'])
    # Close script
    logging.info("Stop signal received, closing script")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='info_retrieval_server',
        description='REST API service for semantic and lexical information retrieval'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the retrieval API."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
