from flask import Flask, jsonify, request
from services import RabbitMQService, PostgresService

app = Flask(__name__)
db = PostgresService(username="pw1tt", password="securepostgrespassword")


@app.route("/")
def home():
    return jsonify(message="Welcome to the Table Tennis Analysis Server!")


@app.route("/checkandreturn", methods=["POST"])
def check_and_return():
    try:
        data = request.json
        if not data:
            return jsonify(error="Missing JSON body"), 400

        frameid = data.get("frameid")
        columnlist = data.get("columnlist", [])
        if frameid is None or not isinstance(columnlist, list):
            return jsonify(error="Missing or invalid 'frameid' or 'columnlist'"), 400

        dbresult = db.get_columns_and_values_by_frameid(frameid)
        if dbresult is None:
            return jsonify(error=f"No data found for frameid {frameid}"), 404

        result = {column: dbresult.get(column, None) for column in columnlist}
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/updatecolumn", methods=["POST"])
def update_column():
    try:
        data = request.json
        if not data:
            return jsonify(error="Missing JSON body"), 400

        frameid = data.get("frameid")
        column = data.get("column")
        value = data.get("value")

        if frameid is None or column is None or value is None:
            return jsonify(error="Missing 'frameid', 'column', or 'value'"), 400

        updated = db.set_column_value_by_frameid(column, value, frameid)
        if not updated:
            return (
                jsonify(error=f"Failed to update {column} for frameid {frameid}"),
                404,
            )

        return jsonify(message=f"Updated {column} for frameid {frameid} to {value}")
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True, port=6060, host="0.0.0.0")
