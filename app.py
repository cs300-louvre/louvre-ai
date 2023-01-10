from flask import Flask, request, g
from werkzeug.local import LocalProxy
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure
from bson.objectid import ObjectId
from bson.errors import InvalidId

from core import Core
from utils import try_catch
from dotenv import dotenv_values

app = Flask(__name__)
core = None
db = None


@try_catch
@app.route('/api/search/image', methods=['GET', 'POST'])
def search_image():
    return "Not implemented", 404


@try_catch
@app.route('/api/search/query', methods=['GET', 'POST'])
def search_query():
    res = core.search_query(request.json["query"], 10)
    results = []
    for r in res:
        results.append({
            "museumId": r[1:] if r[0] == "M" else None,
            "eventId": r[1:] if r[0] == "E" else None,
        })
    return results


@try_catch
@app.route('/api/search/similar', methods=['GET', 'POST'])
def search_similar():
    return "Not implemented", 404


@try_catch
@app.route('/api/text/analysis', methods=['GET', 'POST'])
def text_analysis():
    ...


@try_catch
@app.route('/api/image/analysis', methods=['GET', 'POST'])
def image_analysis():
    ...


if __name__ == '__main__':
    config = dotenv_values(".env")
    db = MongoClient(config['MONGO_URI']).cs300louvre
    core = Core(db)
    app.run(host=config['HOST'], port=config['PORT'], threaded=False, load_dotenv=True)
