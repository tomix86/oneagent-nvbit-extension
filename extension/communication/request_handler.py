from flask import Flask, request

app = Flask(__name__)
UDSPath="/tmp/comm_endpoint"

def start(configCallback):
    global cb_config
    cb_config = configCallback
    app.run(debug=True, port=5000)
#    app.run(debug=True, host=f"unix://{UDSPath}")


def stop():
    func = request.environ.get('werkzeug.server.shutdown')
    if not func:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/config')
def config():
    devicesList = cb_config()
    return str(devicesList)