from TableVertexConsumer import TableVertexConsumer
from BallPositionConsumer import BallPositionConsumer
from PlayerHeatmapConsumer import PlayerHeatmapConsumer

import threading
import random

from constants import Constants

def computerandomplayercoordinates():
    print("Executing computerandomplayercoordinates....")
    x, y, z = random.uniform(0,1000), random.uniform(0,1000), random.uniform(0,1000)

    return (x,y,z)

def run_consumer(consumer):
    consumer.threadstart()  # Assuming Consumer has a start() method

c1 = TableVertexConsumer(rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD)
c2 = BallPositionConsumer(rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD)
c3 = PlayerHeatmapConsumer(rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD)
# c3 = Consumer("Player Heatmap", rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD, logicfunction=computerandomplayercoordinates)
# c4 = Consumer("Trajectory Analysis", rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD, logicfunction=computerandomplayercoordinates)

threads = []
for consumer in [c1, c2, c3]:
    t = threading.Thread(target=run_consumer, args=(consumer,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
