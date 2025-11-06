from flask import Flask, jsonify, request
from matplotlib.style import context
from services import RabbitMQService, PostgresService
from typing import Set
import pprint
from constants import Constants
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from AnsweringMachine import extract_frame_range,classify_question_with_llm,answer_question_1,answer_question2
import time
import requests

app = Flask(__name__)
db = PostgresService(
    username=Constants.POSTGRES_USERNAME, password=Constants.POSTGRES_PASSWORD
)

if os.getenv("NO_RABBITMQ", "false").lower() == "true":
    print("Skipping RabbitMQ connection as NO_RABBITMQ is set to true")
else:
    mqtt = RabbitMQService(
        username=Constants.RABBITMQ_USERNAME, password=Constants.RABBITMQ_PASSWORD
    )
    mqtt.connect()

consumer_column_queue_map = {
    "tablex1": "table-vertex-detection",
    "tabley1": "table-vertex-detection",
    "tablex2": "table-vertex-detection",
    "tabley2": "table-vertex-detection",
    "tablex3": "table-vertex-detection",
    "tabley3": "table-vertex-detection",
    "tablex4": "table-vertex-detection",
    "tabley4": "table-vertex-detection",
    "ballx": "ball-2d-position-detection",
    "bally": "ball-2d-position-detection",
    "ballz": "ball-2d-position-detection",
    "ballxvector": "trajectory-analysis",
    "ballyvector": "trajectory-analysis",
    "ballzvector": "trajectory-analysis",
    "ballbounce": "trajectory-analysis",
    "player1x": "player-heatmap-generation",
    "player1y": "player-heatmap-generation",
    "player1z": "player-heatmap-generation",
    "player2x": "player-heatmap-generation",
    "player2y": "player-heatmap-generation",
    "player2z": "player-heatmap-generation",
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
        return (
            jsonify(
                error="Missing 'consumer_id', 'consumer_queuename' or 'processable_columns'"
            ),
            400,
        )

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
        pprint.pprint(
            f"Received check_and_return request for frameid {frameid} and columns {columnlist}"
        )
        if frameid is None or not isinstance(columnlist, list):
            return jsonify(error="Missing or invalid 'frameid' or 'columnlist'"), 400

        dbresult = db.get_columns_and_values_by_frameid(frameid, videoid)
        # pprint.pprint(f"Database result for frameid {frameid}: {dbresult}")
        if dbresult is None:
            return jsonify(error=f"No data found for frameid {frameid}"), 404

        # result = {column: dbresult.get(column, None) for column in columnlist}
        result = {
            column: dbresult.get(column)
            for column in columnlist
            if column in dbresult and dbresult.get(column) is not None
        }
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500

def determineifnone(map, questionclass=0):
    for key,value in map.items():
        basecheckset = {"videoid", "frameid", "frameaction", "ballz", "ballzvector", "player1z", "player2z", "remarks", "combinedheatmappath", "depthmappath"}
        if questionclass == 1:
            checkset = basecheckset.union({'ballvisibility'})
        elif questionclass == 2:
            checkset = basecheckset.union({
                "player2x",
                "ballvisibility",
                "player1y",
                "ballyvector",
                "player2y",
                "ballxvector",
                "player1x",
                
            })
        else:
            checkset = basecheckset
            
        if key not in checkset and value is None and value != 'ballbounce':
            print(f"\n\nMissing key '{key}' in {checkset}\n\n")
            
            return True
    return False

def check_and_return_in_range_fun(startframeid,endframeid,columnlist,videoid,placerequest,questionclass=0):
        pprint.pprint(
            f"Received check_and_return request for {startframeid} to {endframeid} and columns {columnlist}"
        )
        if startframeid is None or endframeid is None or not isinstance(columnlist, list):
            return jsonify(error="Missing or invalid 'startframeid', 'endframeid' or 'columnlist'"), 400
        
        returnresult = dict()
        returnresult['missing_frames'] = []
        
        for frameid in range(startframeid, endframeid + 1):
            dbresult = db.get_columns_and_values_by_frameid(frameid, videoid)
            print("frameid", frameid)
            pprint.pprint(dbresult)
            if dbresult is None:
                returnresult['missing_frames'].append(frameid)
                continue

            # result = {column: dbresult.get(column, None) for column in columnlist}
            result = {
                column: dbresult.get(column)
                for column in columnlist
                if column in dbresult and dbresult.get(column) is not None
            }
            returnresult[frameid] = result    
        return returnresult


@app.route("/checkandreturninrange", methods=["POST"])
def check_and_return_in_range():
    try:
        data = request.json
        if not data:
            return jsonify(error="Missing JSON body"), 400

        startframeid = data.get("startframeid")
        endframeid = data.get("endframeid")
        columnlist = data.get("columns", [])
        videoid = data.get("videoid")
        placerequest = data.get("placerequest", False)

        returnresult = check_and_return_in_range_fun(startframeid,endframeid,columnlist,videoid,placerequest)
        return jsonify(returnresult)
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
                jsonify(error=f"Failed to update {column} for frameid {frameid} for videoid {videoid}"),
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


def placerequest_fun(startframe, endframe, columnslist, videoid, questionclass=0):
    
    
    targetqueue: Set = set(
        [consumer_column_queue_map.get(c, None) for c in columnslist]
    )
    msg = {
        "type": "request",
        "requestid":"bruno-request-100",
        "requesterid": "bruno-api-client",
        "returnqueue": "default",
        "targetid": "table-vertex-detection",
        "columnslist": [
        columnslist
        ],
        "returnmessageid": "bruno-test-message-100",
        "startframeid": startframe,
        "endframeid": endframe,
        "frameid": 197,
        "videoid": videoid
    }
    t = targetqueue.pop()
    print(f"placing request to queue {t} for frames {startframe} to {endframe} for columns {columnslist}")
    time.sleep(5)
    mqtt.publish(str(msg),t)
    print(2)

@app.route("/placerequest", methods=["POST"])
def placerequest():
    message = request.json
    targetqueue: Set = set(
        [consumer_column_queue_map.get(c, None) for c in message["columnslist"]]
    )
    print(
        f"Placed request from {message['requesterid']} to {targetqueue} for {message['columnslist']}"
    )
    if len(targetqueue) != 1:
        return (
            jsonify(
                message="Unable to determine the right target queue / multiple queues detected for requested columns"
            ),
            500,
        )
    mqtt.publish(str(message), targetqueue.pop())
    return {"message": "placed message", "status": 200}


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
                failed_queues.append({"queue": queue, "error": str(queue_error)})

        if failed_queues and not cleared_queues:
            return (
                jsonify(
                    message="Failed to clear any queues", failed_queues=failed_queues
                ),
                404,
            )

        response = {
            "message": "Queue clearing completed",
            "cleared_queues": cleared_queues,
        }

        if failed_queues:
            response["failed_queues"] = failed_queues
            response["message"] = "Partially cleared queues"

        return jsonify(response), 200
    except Exception as e:
        return jsonify(error=f"Failed to clear queues: {str(e)}"), 500


@app.route("/get-video-path-against-id", methods=["GET"])
def get_video_path_against_id():
    video_id = request.args.get("videoId")
    if not video_id:
        print("Missing 'videoid' in request")
        return jsonify(error="Missing 'videoid'"), 400

    video_path = db.get_video_path_by_videoid(video_id)
    if not video_path:
        print(f"Video path not found for videoid {video_id}")
        return jsonify(error=f"Video path not found for videoid {video_id}"), 404

    return jsonify(videoPath=video_path)


@app.route("/insert-bulk-rows", methods=["POST"])
def insert_bulk_rows():
    data = request.json
    if not data:
        return jsonify(error="Missing JSON body"), 400

    videoid = data.get("videoid")
    framestart = data.get("framestart")
    numberofrows = data.get("numberofrows")

    if videoid is None or framestart is None or numberofrows is None:
        return jsonify(error="Missing 'videoid', 'framestart', or 'numberofrows'"), 400
    inserted = db.insertbulkrows(videoid, framestart, numberofrows)
    # if not inserted:
    #     return jsonify(error="Failed to insert bulk rows"), 500
    return (
        jsonify(
            message=f"Inserted {numberofrows} rows starting from frame {framestart} for videoid {videoid}"
        ),
        201,
    )


@app.route("/upload-video", methods=["POST"])
def upload_video():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        print("Missing 'file' in form-data")
        return jsonify(error="Missing 'file' in form-data"), 400

    filename_from_body = (
        request.form.get("filename")
        or request.args.get("filename")
        or uploaded_file.filename
    )
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
    return (
        jsonify(
            message="Video uploaded successfully",
            filename=final_filename,
            path=save_path,
        ),
        201,
    )

required_keys = {
    1: [
        "ballx",
        "bally",
        "ballz",
        "ballxvector",
        "ballyvector",
        "ballzvector",
        "ballbounce",
        "player1x",
        "player1y",
        "player1z",
        "player2x",
        "player2y",
        "player2z",
        "tablex1",
        "tabley1",
        "tablex2",
        "tabley2",
        "tablex3",
        "tabley3",
        "tablex4",
        "tabley4",
    ],
    2: [
        "ballx",
        "bally",
        "ballbounce",
        "ballz",
        "tablex1",
        "tabley1",
        "tablex2",
        "tabley2",
        "tablex3",
        "tabley3",
        "tablex4",
        "tabley4",
    ],
}

def get_ball_bounces(data):
        bounces = []
        for key, value in data.items():
            if key == "missing_frames":
                continue
            try:
                frameid = int(key)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            ballbounce = value.get("ballbounce")
            if ballbounce is None:
                continue
            if isinstance(ballbounce, bool) and ballbounce:
                bounces.append(frameid)
                continue
            if str(ballbounce).strip().lower() in ("true", "1", "yes"):
                bounces.append(frameid)
        return bounces

def get_data_at_frame(data,frameid,columnslist):
    res = []
    for key,value in data.items():
        if key == "missing_frames":
            continue
        try:
            fid = int(key)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        for column in columnslist:
            res.append(value.get(column,None))
    return res


@app.route("/ask-question", methods=["POST"])
def ask_question():
    data = request.json
    if not data:
        return jsonify(error="Missing JSON body"), 400

    question = data.get("question")
    if not question:
        return jsonify(error="Missing 'question'"), 400
    videoid = data.get("videoid")
    if not videoid:
        return jsonify(error="Missing videoid"), 400
    print(f"video id inside ask-question is {videoid}")
    question_class = classify_question_with_llm(question)
    pprint.pprint(f"Classified question '{question}' as class {question_class}")
    start_frame,end_frame = extract_frame_range(question)
    pprint.pprint(f"Extracted frame range {start_frame} to {end_frame} from question")

    keys = required_keys[question_class]
    # data = check_and_return_in_range_fun(start_frame,end_frame,keys,videoid,False,question_class)
    # pprint.pprint(f"Data retrieved for frames {start_frame} to {end_frame}")
    # pprint.pprint(data) 
    missingframes = (start_frame, end_frame)
    pprint.pprint(f"Missing frames for requested data: {missingframes}")

    # Convert missingframes (list of ints) into contiguous (start,end) ranges like [(x,y), (a,b), ...]
    # try:
    #     frames_sorted = sorted(set(missingframes))
    #     missing_frame_ranges = []
    #     if frames_sorted:
    #         start = end = frames_sorted[0]
    #         for f in frames_sorted[1:]:
    #             if f == end + 1:
    #                 end = f
    #             else:
    #                 missing_frame_ranges.append((start, end))
    #                 start = end = f
    #         missing_frame_ranges.append((start, end))
    # except Exception:
    #     missing_frame_ranges = []
    missing_frame_ranges = missingframes
    pprint.pprint(f"Converted missing frames into ranges: {missing_frame_ranges}")
    # keep the original list intact for the existing loop below; if you prefer to iterate ranges instead,
    # you can replace missingframes = missing_frame_ranges

    placerequest_fun(start_frame, end_frame, keys, videoid)
    
    time.sleep(5)
    # retry with exponential backoff up to 60 seconds (account for the 5s already slept)
    max_wait = 600.0
    delay = 1.0
    elapsed = 5.0  # already waited above

    # Re-check missing frames until none remain or timeout
    while elapsed < max_wait:
        data = check_and_return_in_range_fun(start_frame, end_frame, keys, videoid, False, question_class)
        missingframes = data.get("missing_frames", [])
        if not missingframes:
            break
        pprint.pprint(f"Still missing frames after {elapsed:.1f}s: {missingframes}. Retrying in {delay}s")
        time.sleep(delay)
        elapsed += delay
        delay = min(delay * 2, max_wait - elapsed) if (max_wait - elapsed) > 0 else delay


    # Final evaluation after retries
    if len(data) == 1:
        return (
            jsonify(
                error="Timeout waiting for requested frames",
                missing_frames=missingframes,
                partial_data=data,
            ),
            504,
        )

    '''
    data = {
        frameid(int):{
            columnname(str):value(int)
        }
        missingframes:[]
    }
    '''
# def answer_question_1(ball_position, ball_velocity, player_positions, table_coords):
#     """
#     From frame x to frame y, why did this player lose the point?
    
#     Data needed:
#         Ball position at last bounce
#         Ball velocity
#         Player positions at last bounce
#         Table coordinates
#     """

    if question_class == 1:   
        ball_bounces = get_ball_bounces(data)
        if not ball_bounces:
            return jsonify(error="No ball bounces found in range", data=data), 404

        last_bounce_frame = max(ball_bounces)
        frame_data = data.get(last_bounce_frame) or data.get(str(last_bounce_frame))
        if not frame_data or not isinstance(frame_data, dict):
            return jsonify(
                error=f"No data found for last bounce frame {last_bounce_frame}", frame=last_bounce_frame
            ), 404

        # Ball position at last bounce
        ball_position = {
            "x": frame_data.get("ballx"),
            "y": frame_data.get("bally"),
            "z": frame_data.get("ballz"),
        }

        # Ball velocity (vectors) at last bounce
        ball_velocity = {
            "vx": frame_data.get("ballxvector"),
            "vy": frame_data.get("ballyvector"),
            "vz": frame_data.get("ballzvector"),
        }

        # Player positions at last bounce
        player_positions = {
            "player1": {
                "x": frame_data.get("player1x"),
                "y": frame_data.get("player1y"),
                "z": frame_data.get("player1z"),
            },
            "player2": {
                "x": frame_data.get("player2x"),
                "y": frame_data.get("player2y"),
                "z": frame_data.get("player2z"),
            },
        }

        # Table coordinates (four corners)
        table_coordinates = [
            (frame_data.get("tablex1"), frame_data.get("tabley1")),
            (frame_data.get("tablex2"), frame_data.get("tabley2")),
            (frame_data.get("tablex3"), frame_data.get("tabley3")),
            (frame_data.get("tablex4"), frame_data.get("tabley4")),
        ]

        response = {
            "question_class": question_class,
            "last_bounce_frame": last_bounce_frame,
            "ball_position": ball_position,
            "ball_velocity": ball_velocity,
            "player_positions": player_positions,
            "table_coordinates": table_coordinates,
        }

        reason = answer_question_1(ball_position, ball_velocity, player_positions, table_coordinates)
        response["reason"] = reason
        return jsonify(response), 200
    elif question_class == 2:
        ball_bounces = get_ball_bounces(data)
        if not ball_bounces:
            print("\n\nNo ball bounces found in range\n\n")
            pprint.pprint(data)
            return jsonify(error="No ball bounces found in range", data={str(k): v for k, v in data.items()}), 404


        # Collect ball positions for each frame in the requested range
        ball_positions = {}
        for fid in range(start_frame, end_frame + 1):
            print(fid)
            frame_data = data.get(fid) or data.get(str(fid))
            if not frame_data or not isinstance(frame_data, dict):
                continue
            ball_positions[fid] = {
                "x": frame_data.get("ballx"),
                "y": frame_data.get("bally"),
                "z": frame_data.get("ballz"),
            }
            

        if not ball_positions:
            print("\n\nNo ball position data found in the requested range\n\n")
            pprint.pprint(data)
            return jsonify(error="No ball position data found in the requested range"), 404

        # Table coordinates — pick from the latest frame in range that has table data
        table_coordinates = []
        for fid in range(end_frame, start_frame - 1, -1):
            frame_data = data.get(fid) or data.get(str(fid))
            if not frame_data or not isinstance(frame_data, dict):
                continue
            tx1, ty1 = frame_data.get("tablex1"), frame_data.get("tabley1")
            tx2, ty2 = frame_data.get("tablex2"), frame_data.get("tabley2")
            tx3, ty3 = frame_data.get("tablex3"), frame_data.get("tabley3")
            tx4, ty4 = frame_data.get("tablex4"), frame_data.get("tabley4")
            if None not in (tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4):
                table_coordinates = [
                    (tx1, ty1),
                    (tx2, ty2),
                    (tx3, ty3),
                    (tx4, ty4),
                ]
            break

        response = {
            "question_class": question_class,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "ball_positions": ball_positions,
            "ball_bounces": ball_bounces,
            "table_coordinates": table_coordinates,
        }

        # Call the domain-specific analyzer for question type 2
        analysis = answer_question2(start_frame, end_frame, ball_positions, ball_bounces, table_coordinates)
        response["analysis"] = analysis

        return jsonify(response), 200





if __name__ == "__main__":
    app.run(debug=True, port=6060, host="0.0.0.0")