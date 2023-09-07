import random
import string

from ga import Individual
TARGET_STRING = "Hello, World!"
GENES = string.ascii_letters + ' ,!'
POPULATION_SIZE = 1000

from flask import Flask, jsonify

# ... (GA code here, including the Individual class definition) ...

app = Flask(__name__)

@app.route('/get_results', methods=['GET'])
def get_results():
    return jsonify('HEFUWEOUIEFHUOI')

if __name__ == "__main__":
    app.run(port=5000)
