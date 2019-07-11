#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template
import json, requests, os
from .pytorch import ProjectParams
from .utils_load_model import LoadModel
from .search import GreedySearchDecoder
from .build_model import Model


class TalkControl:
    def __init__(self):
        self.app = Flask(__name__, static_url_path='/static')
        self.pp = ProjectParams()
        self.encoder, self.decoder, self.voc = LoadModel(self.pp).load_model()
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder, self.pp)


app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') # use methods = GET


@app.route('/talk', methods=['POST'])
def talk():
    data = json.loads(request.get_data().decode())
    input = data['nlp']['source']
    # print(input)
    answer = Model(tc.pp).evaluateInputSapCai(input, tc.searcher, tc.voc)
    answer = ' '.join(answer)
    return respond(answer)


def respond(answer):
    return jsonify(
    status=200,
    replies=[{
      'type': 'text',
      'content': '%s' % (answer)
    }],
    # conversation={
    #   'memory': { 'name': value}
    # }
)

@app.route('/errors', methods=['POST'])
def errors():
  print(json.loads(request.get_data().decode()))
  return jsonify(status=200)


if __name__ == "__main__":
    tc = TalkControl()

    # local testing
    #app.run(debug=True, host = '0.0.0.0', port = 5000)

    # for Heroku deployment
    port = int(os.environ['PORT'])
    app.run(port=port, host="0.0.0.0")
