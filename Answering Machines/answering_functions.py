segments = {
    (0.0,0.1):'long',
    (0.1,0.4):'mid',
    (0.4,0.5):'short',
    (0.5,0.6):'short',
    (0.6,0.9):'mid',
    (0.9,1.0):'long',
}

def get_longer_edge(corners):
    edge1 = corners[3][0]-corners[0][0]
    edge2 = corners[2][0]-corners[1][0]
    if abs(edge1) > abs(edge2):
        return (abs(edge1),corners[0],corners[3])
    return (abs(edge2),corners[1],corners[2])

def get_landing_segment(ballx, table_corners):
    table_length,left_edge,_ = get_longer_edge(table_corners)
    relative_x = (ballx - left_edge[0]) / table_length
    print(f"Relative X Position: {relative_x}")
    for (start,end),name in segments.items():
        if start <= relative_x < end:
            return name
            return segment_names[i]
    return "out_of_bounds"

def get_player_distance_to_table(player_pos, table_corners):
    table_length,left_edge,_ = get_longer_edge(table_corners)
    table_center_x = left_edge[0] + table_length / 2
    player_x = player_pos[0]
    distance = abs(player_x - table_center_x)
    return distance

def get_player_distance_to_ball(player_pos, ball_pos):
    distance = ((player_pos[0] - ball_pos[0]) ** 2 + (player_pos[1] - ball_pos[1]) ** 2) ** 0.5
    return distance

def get_player_segment(player_distance):
    if player_distance < 0.2:
        return 'close'
    elif player_distance < 0.5:
        return 'mid'
    else:
        return 'far'

"""
    From frame x to frame y, why did this player lose the point?
    
    Data needed:
        Ball position at last bounce
        Ball velocity
        Player positions at last bounce
        Table coordinates
"""

def answer_question_1(ball_position,ball_velocity,player_positions,table_coords):
    ball_xvector = ball_velocity[0]
    if ball_xvector > 0:
        loser = player_positions['player1']
    else:
        loser  = player_positions['player2']
    landing_segment = get_landing_segment(ball_position[0],table_coords)
    player_distance = get_player_distance_to_table(loser,table_coords)
    player_segment = get_player_segment(player_distance)
    # cases = {
    #     ('long','far'): "The player was too far back to reach the long shot.",
    #     ('long','mid'): "The player was not positioned well enough to reach the long shot.",
    #     ('long','close'): "The player was too close to the table and couldn't react in time to the long shot.",
    #     ('mid','far'): "The player was too far back to effectively return the mid shot.",
    #     ('mid','mid'): "The player was not optimally positioned for the mid shot.",
    #     ('mid','close'): "The player was too close to the table to handle the mid shot properly.",
    #     ('short','far'): "The player was too far back to reach the short shot.",
    #     ('short','mid'): "The player was not in a good position to reach the short shot.",
    #     ('short','close'): "The player was too close to the table and couldn't adjust for the short shot.",
    # }
    # reason = cases.get((landing_segment,player_segment),"The player made an unforced error.")
    
    
    prompt = f"""You are an expert table tennis analyst.
                You are analyzing a point using both sensor and video data.

                Context:

                Ball bounce position: {landing_segment}

                Ball velocity: {ball_velocity}

                Player position: {player_segment} distance from table

                Task:
                Based on this data, explain the most likely reason the player lost the point.
                Consider typical causes in table tennis such as:

                Poor positioning

                Late reaction

                Misjudged spin

                Inability to return a long or fast ball effectively

                Your answer should:

                Provide a short cause analysis (2–3 sentences)

                Optionally include a tactical improvement suggestion for the player"""
    reason = llm_function(prompt)


    return reason
     

"""
    From frame x to y, show me how the ball bounced on the table
    For this, we need to send the ball bounces,landing segment for each bounce and like 0.3s worth of ball position data on both sides of the bounce
    Data Needed:
        Ball positions from x to y
        Ball bounces from x to y
        Table coordinates
"""

def answer_question2(x,y,ball_positions,ball_bounces,table_coords):
    """
    Result:
        {   
            bounces:{
                "1":{
                    "bounceFrame":x,
                    "trajectory":{
                        "frameID":{
                            "ballx":x,
                            "bally":y
                        }
                    }
                    "segment":x
                },
            }
            llmans:x
        }
    """
    segments = []
    result = {}
    count = 0
    for bounce in ball_bounces:
        bounce_frame = bounce['frameID']
        start = bounce_frame-20
        end = bounce_frame-20
        frames = list(range(start,end))
        bounce_ball_pos = ball_positions["bounce_frame"]
        segment = get_landing_segment(bounce_ball_pos[0],table_coords)
        segments.append(segment)
        trajectory_dict = {}
        for frame in frames:
            frame_dict = {}
            frame_dict["ballx"] = ball_positions["frame"][0]
            frame_dict["bally"] = ball_positions["frame"][1]
            trajectory_dict["frame"] = frame_dict
        bounce_dict = {
            "bounceFrame":bounce_frame,
            "segment":segment,
            "trajectory":trajectory_dict
        }
        result[count] = bounce_dict
    
    return result 


import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment; set it in your environment or in a .env file")

genai.configure(api_key=GEMINI_API_KEY)

    
def llm_function(question: str) -> str:
    """
    Uses Gemini API to get an answer to the provided question.
    
    Args:
        question (str): The question to be answered by the LLM.
    
    Returns:
        str: The answer generated by the LLM.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")  

        response = model.generate_content(question)
        
        return response.text.strip()
    
    except Exception as e:
        print("❌ Gemini API call failed.")
        print("Error:", str(e))
        return "Error: Unable to get response from LLM."
    
def main():
    ball_position = (0.35, 0.50)          # (x, y) on table
    ball_velocity = (-2.4, 1.1)           # (x_vel, y_vel) negative x means ball going toward player2
    player_positions = {
        "player1": (0.20, 1.5),          # coordinates in meters from table baseline
        "player2": (0.75, 1.2)
    }
    table_coords = [[353.24, 494.26], [353.24, 679.19], [1431.2, 679.19], [1431.2, 494.26]]


    print("\n🔍 Running table tennis point analysis...\n")
    result = answer_question_1(
        ball_position,
        ball_velocity,
        player_positions,
        table_coords
    )

    print("✅ Analysis Result:\n", result)


# Run main only if file is executed directly
if __name__ == "__main__":
    main()
