from flask import Flask, jsonify, request
from services import RabbitMQService, PostgresService
from typing import Set
import pprint

app = Flask(__name__)
db = PostgresService(username="pw1tt", password="securepostgrespassword")
mqtt = RabbitMQService(username="pw1tt", password="securerabbitmqpassword")
mqtt.connect()

consumer_column_queue_map = {
    "tablex1" : "table-vertex-detection",
    "tabley1" : "table-vertex-detection", 
    "tablex2" : "table-vertex-detection", 
    "tabley2" : "table-vertex-detection", 
    "tablex3" : "table-vertex-detection", 
    "tabley3" : "table-vertex-detection", 
    "tablex4" : "table-vertex-detection", 
    "tabley4" : "table-vertex-detection",
    "ballx":"ball-position-detection",
    "bally":"ball-position-detection",
    "ballz":"ball-position-detection",
    "player1x":"player-heatmap-generation",
    "player1y":"player-heatmap-generation",
    "player1z":"player-heatmap-generation",
    "player2x":"player-heatmap-generation",
    "player2y":"player-heatmap-generation",
    "player2z":"player-heatmap-generation"
}


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
        columnlist = data.get("columns", [])
        pprint.pprint(f"Received check_and_return request for frameid {frameid} and columns {columnlist}")
        if frameid is None or not isinstance(columnlist, list):
            return jsonify(error="Missing or invalid 'frameid' or 'columnlist'"), 400

        dbresult = db.get_columns_and_values_by_frameid(frameid)
        pprint.pprint(f"Database result for frameid {frameid}: {dbresult}")
        if dbresult is None:
            return jsonify(error=f"No data found for frameid {frameid}"), 404

        # result = {column: dbresult.get(column, None) for column in columnlist}
        result = {column: dbresult.get(column) for column in columnlist if column in dbresult and dbresult.get(column) is not None}
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

@app.route("/update-player-coordinates", methods=["POST"])
def update_player_coordinates():
    data = request.json
    videoid = data.get("videoId")
    both_player_coords_map = data.get("both_player_coords_map", {})
    if not both_player_coords_map:
        return jsonify(error="Missing 'both_player_coords_map'"), 400
    if videoid is None:
        return jsonify(error="Missing 'videoId'"), 400
    
    # Update player coordinates in the database
    db.update_player_coordinates(videoid, both_player_coords_map)

    return jsonify(message="Player coordinates updated successfully")

@app.route("/placerequest", methods=["POST"])
def placerequest():
    message = request.json
    targetqueue:Set = set([consumer_column_queue_map.get(c, None) for c in message["columnslist"]])
    print(f"Placed request from {message['requesterid']} to {targetqueue} for {message['columnslist']}")
    if len(targetqueue) != 1:
        return (
            jsonify(
                message="Unable to determine the right target queue / multiple queues detected for requested columns"
            ),
            500,
        )
    mqtt.publish(str(message), targetqueue.pop())
    return {"message":"placed message", "status":200}

if __name__ == "__main__":
    app.run(debug=True, port=6060, host="0.0.0.0")
