from TableVertexConsumer import TableVertexConsumer
from BallPositionConsumer import Ball2DPositionConsumer
from PlayerHeatmapConsumer import PlayerHeatmapConsumer
from TrajectoryAnalysisConsumer import TrajectoryAnalysisConsumer
# from SpacePointSearch import SpacePointSearch

import threading
import random

from constants import Constants


def computerandomplayercoordinates():
    self.newprint("Executing computerandomplayercoordinates....")
    x, y, z = random.uniform(0, 1000), random.uniform(0, 1000), random.uniform(0, 1000)

    return (x, y, z)


def run_consumer(consumer):
    consumer.threadstart()  # Assuming Consumer has a start() method


c1 = TableVertexConsumer(
    rabbitmqusername=Constants.RABBITMQ_USERNAME,
    rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
    id="table-vertex-detection-consumer",
)
c2 = Ball2DPositionConsumer(
    rabbitmqusername=Constants.RABBITMQ_USERNAME,
    rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
    id="ball-2d-position-detection",
)
c3 = PlayerHeatmapConsumer(
    rabbitmqusername=Constants.RABBITMQ_USERNAME,
    rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
    id="player-heatmap-consumer",
)
c4 = TrajectoryAnalysisConsumer(
    rabbitmqusername=Constants.RABBITMQ_USERNAME,
    rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
    id="trajectory-analysis",
)
# c5 = SpacePointSearch(
#     rabbitmqusername=Constants.RABBITMQ_USERNAME,
#     rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
# )

threads = []
for consumer in [c1, c2, c3, c4]:  # , c5
    t = threading.Thread(target=run_consumer, args=(consumer,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
