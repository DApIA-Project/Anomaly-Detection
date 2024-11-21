# flask app
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import numpy as np

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import threading

class SemaphoreQueue:
    thread_update = None
    queue = []
    sem_storage = []
    available = []
   

    def acquire(self):
        sem_id = len(self.sem_storage)

        if (len(self.available) == 0):
            self.sem_storage.append(threading.Semaphore())
        else:
            sem_id = self.available.pop(0)

        self.sem_storage[sem_id].acquire()
        self.queue.append(sem_id)
        
        if (len(self.queue) >= 2):
            sem_id_prev = self.queue[-2]
            self.sem_storage[sem_id_prev].acquire()            
            self.sem_storage[sem_id_prev].release()
            self.available.append(sem_id_prev)

            
    def release(self):
        sem_id = self.queue.pop(0)
        self.sem_storage[sem_id].release()

    def __len__(self):
        return len(self.queue)

sem = SemaphoreQueue()

from AdsbAnomalyDetector import predict, clear_cache

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def api():
    message:"list[dict[str, str]]" = request.get_json()
    t = int(message[0]["timestamp"])%3600
    print("Recieved msg time ", t//60, "m", t%60, "s")
    # print("Call queue lenght :", len(sem), "msg time : ", t//60, "m", t%60, "s")

    sem.acquire()
    out = predict(message, debug = True)
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

@app.route("/reset", methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def reset():
    icaos:"list[str]" = request.get_json()
    sem.acquire()
    for icao in icaos:
        clear_cache(icao)
    sem.release()

    response = app.response_class(
        response=json.dumps({"status": "ok"}),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    print("Server ready")
    app.run(debug=True, port=3033, use_reloader=False)