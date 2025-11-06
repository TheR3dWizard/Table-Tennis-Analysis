CREATE TABLE table_tennis_analysis (
    videoId INTEGER,
    frameId INTEGER,
    frameAction VARCHAR,
    tablex1 FLOAT,
    tabley1 FLOAT,
    tablex2 FLOAT,
    tabley2 FLOAT,
    tablex3 FLOAT,
    tabley3 FLOAT,
    tablex4 FLOAT,
    tabley4 FLOAT,
    ballvisibility BOOLEAN,
    ballx FLOAT,
    bally FLOAT,
    ballz FLOAT,
    depthMapPath VARCHAR,
    player1x FLOAT,
    player1y FLOAT,
    player1z FLOAT,
    player2x FLOAT,
    player2y FLOAT,
    player2z FLOAT,
    ballspeed FLOAT,
    ballxvector FLOAT,
    ballyvector FLOAT,
    ballzvector FLOAT,
    remarks VARCHAR
);

CREATE TABLE video_table (
    videoId INTEGER,
    videoPath VARCHAR,
    videoName VARCHAR,
    videoTag VARCHAR,
    fullvideoHeatmapPath VARCHAR,
    videoDotMatrixSource VARCHAR
);

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        tablex1,
        tabley1,
        tablex2,
        tabley2,
        tablex3,
        tabley3,
        tablex4,
        tabley4,
        ballx,
        bally,
        ballz,
        depthMapPath,
        player1x,
        player1y,
        player1z,
        player2x,
        player2y,
        player2z
    )
VALUES
    (
        1,
        101,
        'serve',
        12.2,
        8.4,
        13.1,
        8.5,
        13.2,
        9.2,
        12.1,
        9.1,
        6.5,
        7.2,
        0.9,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F',
        7.9,
        2.1,
        0.8,
        7.9,
        2.1,
        0.8
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        tablex1,
        tabley1,
        tablex2,
        tabley2,
        tablex3,
        tabley3,
        tablex4,
        tabley4,
        ballx,
        bally,
        ballz,
        depthMapPath,
        player1x,
        player1y,
        player1z,
        player2x,
        player2y,
        player2z
    )
VALUES
    (
        1,
        102,
        'rally',
        12.3,
        8.5,
        13.0,
        8.6,
        13.0,
        9.0,
        12.0,
        9.0,
        6.6,
        7.1,
        0.95,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F',
        7.95,
        2.15,
        0.85,
        7.95,
        2.15,
        0.85
    );

-- start test commands for tablevertexdetection
INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        101,
        'serve',
        6.5,
        7.2,
        0.9,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        102,
        'serve',
        6.5,
        7.2,
        0.9,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        103,
        'serve',
        6.5,
        7.2,
        0.9,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        104,
        'serve',
        6.5,
        7.2,
        0.9,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

-- end test comments for tablevertexdetection
-- start test commands for heatmap
INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        tablex1,
        tabley1,
        tablex2,
        tabley2,
        tablex3,
        tabley3,
        tablex4,
        tabley4,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        101,
        'serve',
        12.2,
        8.4,
        13.1,
        8.5,
        13.2,
        9.2,
        12.1,
        9.1,
        6.5,
        7.2,
        0.9,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        tablex1,
        tabley1,
        tablex2,
        tabley2,
        tablex3,
        tabley3,
        tablex4,
        tabley4,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        102,
        'rally',
        12.3,
        8.5,
        13.0,
        8.6,
        13.0,
        9.0,
        12.0,
        9.0,
        6.6,
        7.1,
        0.95,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        tablex1,
        tabley1,
        tablex2,
        tabley2,
        tablex3,
        tabley3,
        tablex4,
        tabley4,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        103,
        'rally',
        12.3,
        8.5,
        13.0,
        8.6,
        13.0,
        9.0,
        12.0,
        9.0,
        6.6,
        7.1,
        0.95,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        tablex1,
        tabley1,
        tablex2,
        tabley2,
        tablex3,
        tabley3,
        tablex4,
        tabley4,
        ballx,
        bally,
        ballz,
        depthMapPath
    )
VALUES
    (
        1,
        104,
        'rally',
        12.3,
        8.5,
        13.0,
        8.6,
        13.0,
        9.0,
        12.0,
        9.0,
        6.6,
        7.1,
        0.95,
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F'
    );

-- end test comments for heatmap
INSERT INTO
    table_tennis_analysis (
        videoId,
        frameId,
        frameAction,
        depthMapPath,
        player1x,
        player1y,
        player1z,
        player2x,
        player2y,
        player2z
    )
VALUES
    (
        1,
        16,
        'forehand',
        'https://www.paralympic.org/sites/default/files/styles/large_original/public/images/150413103127066_LON_0109_4685.jpg?itok=rgvvqm0F',
        8.0,
        2.2,
        0.9,
        7.95,
        2.15,
        0.85
    );

DELETE FROM
    table_tennis_analysis;

SELECT
    *
FROM
    table_tennis_analysis;

DROP TABLE IF EXISTS video_table;

DROP TABLE IF EXISTS table_tennis_analysis;