from services import RabbitMQService
import uuid
import requests


class Consumer:
    def __init__(
        self,
        name,
        id=None,
        queuename=None,
        rabbitmqusername="default",
        rabbitmqpassword="default",
        serverurl="localhost:6060",
    ):
        self.name = name
        self.id = id or str(uuid.uuid4())
        self.queuename = queuename or self.name.lower().replace(" ", "-")
        print(
            f"Created a {self.name} consumer with ID: {self.id} and Queue: {self.queuename}"
        )
        self.rabbitmqservice = RabbitMQService(
            username=rabbitmqusername, password=rabbitmqpassword, queue=self.queuename
        )
        self.rabbitmqservice.connect()
        print(
            f"{self.name} consumer sucessfully connected to RabbitMQ with {rabbitmqusername} credential"
        )
        self.server = serverurl
        self.idqueuemap = dict()

        self.rabbitmqservice.consume(self.messagecallback, self.queuename)
        self.rabbitmqservice.consume(self.broadcastcallback, "bcast")

    def placebroadcast(self):
        message = {
            "type": "join",
            "consumerid": self.id,
            "consumerqueue": self.queuename,
        }

        self.rabbitmqservice.publish(str(message), queue="bcast")

    def placerequest(self, columnslist, targetid, targetqueue):
        message = {
            "type": "request",
            "requestid": str(uuid.uuid4()),
            "requesterid": self.id,
            "targetid": targetid,
            "columnslist": columnslist,
        }
        self.rabbitmqservice.publish(str(message), queue=targetqueue)

    def placesuccess(self, requestid, requesterid, targetid, requestorqueue):
        message = {
            "type": "success",
            "requestid": requestid,
            "requesterid": requesterid,
            "targetid": targetid,
        }
        self.rabbitmqservice.publish(str(message), queue=requestorqueue)

    def broadcastcallback(self, body):
        self.idqueuemap[body["consumerid"]] = body["consumerqueue"]

    def messagecallback(self, body):
        print(f"Message Received: {body}")
        # Perform callback logic with context specific model
        self.placesuccess(
            body["requestid"],
            body["requesterid"],
            body["targetid"],
            self.idqueuemap.get(body["requestorqueue"], "default"),
        )

    def check(self, frameid, columnlist):
        message = {"frameid": frameid, "columns": columnlist}

        response = requests.post(f"{self.server}/checkandreturn", json=message)
        return response.json()

    def update(self, frameid, columnname, value):
        message = {"frameid": frameid, "column": columnname, "value": value}

        response = requests.post(f"{self.server}/updatecolumn", json=message)
        return response.json()
