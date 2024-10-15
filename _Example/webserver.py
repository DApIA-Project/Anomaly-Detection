# flask app
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import numpy as np

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import threading
sem = threading.Semaphore()

from AdsbAnomalyDetector import predict

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def api():
    message:"list[dict[str, str]]" = request.get_json()

    sem.acquire()
    out = predict(message)
    sem.release()

    # convert out bool_ to boolean
    for i in range(len(out)):
        for key in out[i].keys():
            if (isinstance(out[i][key], np.bool_)):
                out[i][key] = bool(out[i][key])
            # check if the value is nan
            if (isinstance(out[i][key], np.float64)):
                if (np.isnan(out[i][key])):
                    out[i][key] = None

    response = app.response_class(
        response=json.dumps(out),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    print("Server ready")
    app.run(debug=True, port=3033, use_reloader=False)