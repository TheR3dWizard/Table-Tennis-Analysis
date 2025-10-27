CREATE TABLE table_tennis_analysis (
    videoId INTEGER,
    frameId INTEGER,
    frameAction VARCHAR,
    tableid INTEGER,
    ballvisibility BOOLEAN,
    ballid INTEGER,
    depthMapPath VARCHAR,
    player1id INTEGER,
    player2id INTEGER,
    remarks VARCHAR,
    PRIMARY KEY (videoId, frameId)
);

CREATE TABLE table_coords(
    tableid INTEGER,
    tablex1 FLOAT,
    tabley1 FLOAT,
    tablex2 FLOAT,
    tabley2 FLOAT,
    tablex3 FLOAT,
    tabley3 FLOAT,
    tablex4 FLOAT,
    tabley4 FLOAT
)

CREATE TABLE ball_data(
    ballId INTEGER,
    ballx FLOAT,
    bally FLOAT,
    ballz FLOAT,
    ballxvector FLOAT,
    ballyvector FLOAT,
    ballzvector FLOAT
)

CREATE TABLE player_positions(
    playerid INTEGER,
    playerx FLOAT,
    playery FLOAT,
    playerz FLOAT
)

CREATE TABLE bounces(
    frameId INTEGER,
    ballId INTEGER
)

CREATE TABLE video_table (
    videoId INTEGER,
    videoPath VARCHAR,
    videoName VARCHAR,
    videoTag VARCHAR,
    fullvideoHeatmapPath VARCHAR,
    videoDotMatrixSource VARCHAR
);
