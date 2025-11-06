from services import RabbitMQService, PostgresService
import uuid
import requests
import time
import pprint
from constants import Constants


class Consumer:
    def __init__(
        self,
        name,
        id=None,
        queuename=None,
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        serverurl=Constants.DEFAULT_SERVER_URL,
    ):
        self.name = name
        self.id = id or str(uuid.uuid4())
        self.queuename = queuename or self.name.lower().replace(" ", "-")
        self.newprint(
            f"Created a {self.name} consumer with ID: {self.id} and Queue: {self.queuename}"
        )
        self.rabbitmqservice = RabbitMQService(
            username=rabbitmqusername, password=rabbitmqpassword, queue=self.queuename
        )
        self.rabbitmqservice.connect()
        self.newprint(
            f"{self.name} consumer sucessfully connected to RabbitMQ with {rabbitmqusername} credential"
        )
        self.server = serverurl
        self.hashmap = {}
        self.pgs = PostgresService(
            username=Constants.POSTGRES_USERNAME, password=Constants.POSTGRES_PASSWORD
        )
        self.pgs.connect()
        # self.rabbitmqservice.consume(self.messagecallback, self.queuename)

    def newprint(self, *msgs, end="\n", event="log", skipconsole=False, level="info"):
        # timestamp in [] and id in [] then only message
        # allow multiple message parts (e.g. self.newprint("text", videoId))
        msg = " ".join(str(m) for m in msgs) if msgs else ""
        # ensure end is a string or None
        if end is not None and not isinstance(end, str):
            end = str(end)

        lokiline = f"[{self.id}] {msg}"
        logtime = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        line = logtime + lokiline
        print(line, end=end) if not skipconsole else None

        # Stream the same log line to a local Loki instance at port 3100
        try:
            loki_url = "http://localhost:3100/loki/api/v1/push"
            # Loki expects nanosecond epoch as a string
            timestamp_ns = str(int(time.time() * 1e9))
            # include the printed end (if any) in the log line sent to Loki
            send_line = lokiline + (end if end is not None else "")
            payload = {
                "streams": [
                    {
                        "stream": {
                            "job": "consumer",
                            "consumer_id": self.id,
                            "queue": getattr(self, "queuename", ""),
                            "name": getattr(self, "name", ""),
                            "event": event,
                            "level": level,
                            "session": Constants.SESSION,
                        },
                        "values": [[timestamp_ns, send_line]],
                    }
                ]
            }
            # short timeout and ignore failures so logging doesn't crash the app
            requests.post(loki_url, json=payload, timeout=1)
        except Exception:
            pass

    def joinserver(self):
        message = {
            "consumer_id": self.id,
            "consumer_queuename": self.queuename,
            "processable_columns": self.processable_columns,
        }
        self.newprint(f"{self.name} joining server with message: {message}", event="joinserver")
        response = requests.post(f"{self.server}/consumer/join", json=message)
        self.newprint(f"Server response: {response.json()}", event="joinserver")
        return response.json()

    def threadstart(self):
        self.rabbitmqservice.consume(self.messagecallback, self.queuename)

    def placerequest(
        self,
        columnslist,
        returnmessageid,
        startframeid=None,
        endframeid=None,
        videoid=1,
    ):
        message = {
            "type": "request",
            "requestid": str(uuid.uuid4()),
            "requesterid": self.id,
            "returnqueue": self.queuename,
            "columnslist": columnslist,
            "returnmessageid": returnmessageid,
            "startframeid": startframeid,
            "endframeid": endframeid,
            "videoid": videoid,
        }
        self.newprint(f"Placing request... for {columnslist}", event="placerequest")
        # [ABSTRACTED] self.rabbitmqservice.publish(str(message), queue=targetqueue)
        self.hashmap[returnmessageid] = message
        response = requests.post(f"{self.server}/placerequest", json=message)

        return response.json()

    def placesuccess(
        self,
        requestid,
        requesterid,
        requestorqueue,
        returnmessageid,
        startframeid,
        endframeid,
        videoid=1,
    ):
        self.newprint(
            f"Placing success message... from {self.queuename} to {requestorqueue}", event="placesuccess"
        )
        message = {
            "type": "success",
            "requestid": requestid,
            "requesterid": requesterid,
            "returnqueue": requestorqueue,
            "returnmessageid": returnmessageid,
            "mudithavar": self.id,
            "startframeid": startframeid,
            "endframeid": endframeid,
            "videoid": videoid,
        }
        self.rabbitmqservice.publish(str(message), queue=requestorqueue)

    def messagecallback(self, body):
        body = eval(body)  # convert string to dict

        # Perform callback logic with context specific model
        # self.newprint(body)
        if body["type"] == "request":
            self.newprint(f"\n\n[REQUEST] Message Received:", event="incomingrequest")
            pprint.pprint(body)
            self.newprint(body, skipconsole=True, event="incomingrequest")
            self.newprint("\n\n")

            logicreturn = self.logic(body)
            # time.sleep(5)  # simulate processing time
            if logicreturn:
                self.newprint("Logic executed successfully, sending success message...", event="logicsuccess")
                self.placesuccess(
                    body["requestid"],
                    body["requesterid"],
                    body["returnqueue"],
                    body["returnmessageid"],
                    body.get("startframeid", None),
                    body.get("endframeid", None),
                    videoid=body.get("videoid", 1),
                )
            else:
                self.newprint(
                    "Logic execution failed or pending, not sending success message.",
                    event="logicfailure",
                    level="error",
                )

        elif body["type"] == "success":
            self.newprint(f"\n\n[SUCCESS] Message Received:")
            pprint.pprint(body)
            self.newprint("\n\n")
            if body["returnmessageid"] in self.hashmap:
                messagebody = self.hashmap.pop(body["returnmessageid"])
                messagebody["successmessage"] = True
                self.logic(messagebody)
            else:
                self.newprint(self.hashmap)
                self.newprint(
                    "Request ID not found in hashmap. Ignoring success message."
                )

    def check(self, frameid, columnlist):
        message = {"frameid": frameid, "columns": columnlist}

        response = requests.post(f"{self.server}/checkandreturn", json=message)
        return response.json()

    def update(self, frameid, columnname, value):
        message = {"frameid": frameid, "column": columnname, "value": value}

        response = requests.post(f"{self.server}/updatecolumn", json=message)
        return response.json()
