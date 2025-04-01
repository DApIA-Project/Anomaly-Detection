# flask app
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import numpy as np
import time

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

DEBUG = False

import threading

class SemaphoreQueue:
    thread_update = None
    queue = []
    queue_order = []
    sem_storage = []
    sem_locked = []
    available = []
    last_release = -1
    first_call = True
   
    def __get_queue_index__(self, order):
        a = 0
        b = len(self.queue_order)
        while (a < b):
            m = (a+b)//2
            if (self.queue_order[m] < order):
                a = m + 1
            else:
                b = m
                
        return a

    def acquire(self, order):
        sem_id = len(self.sem_storage)
        
        if (DEBUG): print("begin acquire ", order)

        if (len(self.available) == 0):
            self.sem_storage.append(threading.Semaphore())
            self.sem_locked.append(True)
        else:
            sem_id = self.available.pop(0)
            self.sem_locked[sem_id] = True

        self.sem_storage[sem_id].acquire()
        i = self.__get_queue_index__(order)
        self.queue.insert(i, sem_id)
        self.queue_order.insert(i, order)
        
        if (DEBUG): print("queue ", self.queue_order)

        
        # wait order
        while (self.last_release + 1 != order):
            time.sleep(0.1) 
            
        if (DEBUG): print("acquire open for ", order)
        
        i = self.__get_queue_index__(order)
        if (i > 0):
            sem_id_prev = self.queue[i-1]
            self.sem_storage[sem_id_prev].acquire()            
            self.sem_storage[sem_id_prev].release()
            self.available.append(sem_id_prev)
            
        if (DEBUG): print("thread acquired ", order)

            
    def release(self, order):
        if (DEBUG): print("begin release ", order)
        i = self.__get_queue_index__(order)
        sem_id = self.queue[i]
        self.sem_storage[sem_id].release()
        self.sem_locked[sem_id] = False
        
        while len(self.queue) > 0 and not(self.sem_locked[self.queue[0]]):
            self.queue.pop(0)
            self.queue_order.pop(0)
            
        self.last_release = order
        if (DEBUG): print("release ", order)
        if (DEBUG): print("queue ", self.queue_order)
        
    def reset(self, order=0):
        for sem in self.sem_storage:
            sem.release()
        self.queue = []
        self.queue_order = []
        self.sem_storage = []
        self.sem_locked = []
        self.available = []
        self.last_release = order-1
        
    def is_first_call(self):
        if self.first_call:
            self.first_call = False
            return True
        return False
            
    def __len__(self):
        return len(self.queue)



sem = SemaphoreQueue()

def check_sem_reset(order):
    if (sem.is_first_call() or order == 0):
        sem.reset(order)

from AdsbAnomalyDetector import predict, clear_cache

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def api():
    data = request.get_json()
    order = data["order"]
    message:"list[dict[str, str]]" = data["messages"]
    t = int(message[0]["timestamp"])%3600
    if (DEBUG): print("Recieved msg time ", t//60, "m", t%60, "s")
    # if (DEBUG): print("Call queue lenght :", len(sem), "msg time : ", t//60, "m", t%60, "s")

    check_sem_reset(order)
    sem.acquire(order)
    out = predict(message, debug = True)
    sem.release(order)

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
    data = request.get_json()
    order = data["order"]
    icaos:"list[str]" = data["icaos"]
    
    check_sem_reset(order)
    sem.acquire(order)
    for icao in icaos:
        clear_cache(icao)
    sem.release(order)

    response = app.response_class(
        response=json.dumps({"status": "ok"}),
        status=200,
        mimetype='application/json'
    )
    return response

    
class webserver:
    def run(port=3033):
        print("Startup AdsbAnomalyDetector webserver at http://localhost:"+str(port)+"/")
        app.run(debug=True, port=port, use_reloader=False, load_dotenv=False)
        
        
if (__name__ == "__main__"):
    webserver.run()