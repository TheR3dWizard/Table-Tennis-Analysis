from TableVertexConsumer import TableVertexConsumer
from BallPositionConsumer import BallPositionConsumer
from PlayerHeatmapConsumer import PlayerHeatmapConsumer

import threading
import random


def computerandomplayercoordinates():
    print("Executing computerandomplayercoordinates....")
    x, y, z = random.uniform(0,1000), random.uniform(0,1000), random.uniform(0,1000)

    return (x,y,z)

def run_consumer(consumer):
    consumer.threadstart()  # Assuming Consumer has a start() method

c1 = TableVertexConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword")
c2 = BallPositionConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword")
c3 = PlayerHeatmapConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword")
# c3 = Consumer("Player Heatmap", rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword", logicfunction=computerandomplayercoordinates)
# c4 = Consumer("Trajectory Analysis", rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword")

threads = []
for consumer in [c1, c2]:
    t = threading.Thread(target=run_consumer, args=(consumer,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
