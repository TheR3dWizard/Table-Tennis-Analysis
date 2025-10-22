from flask import Flask, jsonify, request
from services import RabbitMQService, PostgresService
from typing import Set
import pprint
from constants import Constants
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)
db = PostgresService(username=Constants.POSTGRES_USERNAME, password=Constants.POSTGRES_PASSWORD)
mqtt = RabbitMQService(username=Constants.RABBITMQ_USERNAME, password=Constants.RABBITMQ_PASSWORD)
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
    "ballx":"ball-2d-position-detection",
    "bally":"ball-2d-position-detection",
    "ballz":"depth-map-estimator",
    "player1x":"player-heatmap-generation",
    "player1y":"player-heatmap-generation",
    "player1z":"player-heatmap-generation",
    "player2x":"player-heatmap-generation",
    "player2y":"player-heatmap-generation",
    "player2z":"player-heatmap-generation"
}

dynamic_consumer_column_queue_map = dict()

@app.route("/")
def home():
    return jsonify(message="Welcome to the Table Tennis Analysis Server!")


@app.route("/consumer/join", methods=["POST"])
def consumer_join():
    data = request.json
    if not data:
        return jsonify(error="Missing JSON body"), 400

    consumer_id = data.get("consumer_id")
    consumer_queuename = data.get("consumer_queuename")
    processable_columns = data.get("processable_columns", [])
    if not consumer_id or not consumer_queuename or not processable_columns:
        return jsonify(error="Missing 'consumer_id', 'consumer_queuename' or 'processable_columns'"), 400

    for column in processable_columns:
        dynamic_consumer_column_queue_map[column] = consumer_queuename
    
    # Process the consumer join request
    # For example, add the consumer to a database or a list
    return jsonify(message=f"Consumer {consumer_id} joined successfully"), 200

@app.route("/checkandreturn", methods=["POST"])
def check_and_return():
    try:
        data = request.json
        if not data:
            return jsonify(error="Missing JSON body"), 400

        frameid = data.get("frameid")
        columnlist = data.get("columns", [])
        videoid = data.get("videoid")
        pprint.pprint(f"Received check_and_return request for frameid {frameid} and columns {columnlist}")
        if frameid is None or not isinstance(columnlist, list):
            return jsonify(error="Missing or invalid 'frameid' or 'columnlist'"), 400

        dbresult = db.get_columns_and_values_by_frameid(frameid, videoid)
        # pprint.pprint(f"Database result for frameid {frameid}: {dbresult}")
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

        videoid = data.get("videoid")
        frameid = data.get("frameid")
        column = data.get("column")
        value = data.get("value")

        if frameid is None or column is None or value is None:
            return jsonify(error="Missing 'frameid', 'column', or 'value'"), 400

        updated = db.set_column_value_by_frameid(column, value, frameid, videoid)
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

@app.route("/clearqueues", methods=["POST"])
def clear_queues():
    try:
        # Get all unique queue names from the consumer_column_queue_map
        queues = set(consumer_column_queue_map.values())
        
        cleared_queues = []
        failed_queues = []
        
        for queue in queues:
            try:
                mqtt.clear_queue(queue)
                cleared_queues.append(queue)
            except Exception as queue_error:
                # Queue might not exist or other error
                failed_queues.append({
                    "queue": queue,
                    "error": str(queue_error)
                })
        
        if failed_queues and not cleared_queues:
            return jsonify(
                message="Failed to clear any queues",
                failed_queues=failed_queues
            ), 404
        
        response = {
            "message": "Queue clearing completed",
            "cleared_queues": cleared_queues
        }
        
        if failed_queues:
            response["failed_queues"] = failed_queues
            response["message"] = "Partially cleared queues"
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify(error=f"Failed to clear queues: {str(e)}"), 500

@app.route("/get-video-path-against-id", methods=["GET"])
def get_video_path_against_id():
    video_id = request.args.get("videoid")
    if not video_id:
        return jsonify(error="Missing 'videoid'"), 400

    video_path = db.get_video_path_by_videoid(video_id)
    if not video_path:
        return jsonify(error=f"Video path not found for videoid {video_id}"), 404

    return jsonify(videoPath=video_path)

@app.route("/upload-video", methods=["POST"])
def upload_video():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify(error="Missing 'file' in form-data"), 400

    filename_from_body = request.form.get("filename") or request.args.get("filename") or uploaded_file.filename
    if not filename_from_body:
        return jsonify(error="Missing 'filename' in form-data or query string"), 400
    
    base_name = secure_filename(os.path.splitext(filename_from_body)[0])
    ext = os.path.splitext(uploaded_file.filename)[1] or ""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    final_filename = f"{base_name}_{timestamp}{ext}"

    save_dir = os.path.join(Constants.DEFAULT_FILE_SAVE_PATH)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, final_filename)

    uploaded_file.save(save_path)
    return jsonify(message="Video uploaded successfully", filename=final_filename, path=save_path), 201

if __name__ == "__main__":
    app.run(debug=True, port=6060, host="0.0.0.0")
