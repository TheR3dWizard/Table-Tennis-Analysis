import re
from ollama import Client

# Initialize Ollama client (ensure Ollama is running on localhost:11434)
client = Client(host='http://localhost:11434')

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


def answer_question_1(ball_position, ball_velocity, player_positions, table_coords):
    """
    From frame x to frame y, why did this player lose the point?
    
    Data needed:
        Ball position at last bounce
        Ball velocity
        Player positions at last bounce
        Table coordinates
    """
    ball_xvector = ball_velocity[0]
    if ball_xvector > 0:
        loser = player_positions['player1']
    else:
        loser = player_positions['player2']
    
    landing_segment = get_landing_segment(ball_position[0], table_coords)
    player_distance = get_player_distance_to_table(loser, table_coords)
    player_segment = get_player_segment(player_distance)
    
    cases = {
        ('long','far'): "The player was too far back to reach the long shot.",
        ('long','mid'): "The player was not positioned well enough to reach the long shot.",
        ('long','close'): "The player was too close to the table and couldn't react in time to the long shot.",
        ('mid','far'): "The player was too far back to effectively return the mid shot.",
        ('mid','mid'): "The player was not optimally positioned for the mid shot.",
        ('mid','close'): "The player was too close to the table to handle the mid shot properly.",
        ('short','far'): "The player was too far back to reach the short shot.",
        ('short','mid'): "The player was not in a good position to reach the short shot.",
        ('short','close'): "The player was too close to the table and couldn't adjust for the short shot.",
    }
    reason = cases.get((landing_segment, player_segment), "The player made an unforced error.")
    return reason


def answer_question2(x, y, ball_positions, ball_bounces, table_coords):
    """
    From frame x to y, show me how the ball bounced on the table
    
    Data Needed:
        Ball positions from x to y
        Ball bounces from x to y
        Table coordinates
    """
    result = {}
    count = 0
    
    for bounce in ball_bounces:
        bounce_frame = bounce['frameID']
        start = bounce_frame - 20
        end = bounce_frame + 20
        frames = list(range(start, end))
        
        bounce_ball_pos = ball_positions[str(bounce_frame)]
        segment = get_landing_segment(bounce_ball_pos[0], table_coords)
        
        trajectory_dict = {}
        for frame in frames:
            if str(frame) in ball_positions:
                frame_dict = {
                    "ballx": ball_positions[str(frame)][0],
                    "bally": ball_positions[str(frame)][1]
                }
                trajectory_dict[str(frame)] = frame_dict
        
        bounce_dict = {
            "bounceFrame": bounce_frame,
            "segment": segment,
            "trajectory": trajectory_dict
        }
        result[str(count)] = bounce_dict
        count += 1
    
    return result


# ============================================================================
# LLM CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_question_with_llm(question):
    """
    Use Phi-3 Mini to classify the question into:
    - 1: "Why did player lose?" (answer_question_1)
    - 2: "How did ball bounce?" (answer_question2)
    - 0: "Not enough data" / Unknown
    
    Handles indirect and non-obvious questions.
    """
    
    classification_prompt = f"""You are a table tennis analysis assistant. Analyze the following question and classify it into ONE of these categories:

CATEGORY 1 - "why_player_lost": Questions asking for analysis or explanation of why a player failed to win a point or made a mistake. These questions seek root cause analysis of player performance issues.
Examples of indirect forms:
- "What went wrong with player on the right between frames 10 and 50?"
- "During the rally between frame 10 and 50, the player on the right couldn't capitalize. Why?"
- "Analyze the performance drop for player 2 in the segment from frame 10 to 50"
- "What was the deciding factor that cost the right-side player the point between frames 10 and 50?"
- "Looking at frames 10 to 50, explain the player's downfall on the right side"

CATEGORY 2 - "ball_bounce_trajectory": Questions asking for information about ball movement, bounces, trajectory patterns, or how the ball moved during a rally. These questions focus on ball physics and bounce analysis.
Examples of indirect forms:
- "I need to understand the ball's journey during frames 10 to 50"
- "Can you trace the ball's path on the table between frame 10 and 50?"
- "Show me what happened with the ball during the rally from frame 10 to 50"
- "Analyze the bounce pattern and movement of the ball from frames 10 to 50"
- "Let's review the ball dynamics during the exchange from frame 10 to 50"

CATEGORY 3 - "unknown": Questions that don't clearly fit the above categories or lack sufficient context about the specific frames/rally being analyzed.

Question: "{question}"

IMPORTANT: 
- Focus on the intent of the question, not just the exact wording
- Respond with ONLY the classification category (why_player_lost, ball_bounce_trajectory, or unknown)
- Do not include any other text or explanation"""

    try:
        response = client.generate(
            model='phi3:3.8b',
            prompt=classification_prompt,
            stream=False
        )
        
        classification_text = response['response'].strip().lower()
        
        # Extract classification from response
        if 'why_player_lost' in classification_text:
            return 1
        elif 'ball_bounce_trajectory' in classification_text:
            return 2
        else:
            return 0
    
    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return 0


def extract_frame_range(question):
    """
    Extract frame range (start_frame, end_frame) from question using regex.
    Handles various formats.
    
    Examples:
    - "From frame 10 to 50" -> (10, 50)
    - "frame 5 to frame 100" -> (5, 100)
    - "between frames 10 and 50" -> (10, 50)
    - "during the exchange from frame 10 to 50" -> (10, 50)
    """
    
    # Pattern 1: "frame X to frame Y" or "frame X to Y"
    pattern1 = r'frame\s+(\d+)\s+to\s+(?:frame\s+)?(\d+)'
    match = re.search(pattern1, question, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    # Pattern 2: "between frames X and Y"
    pattern2 = r'(?:between|during|from)\s+frames?\s+(\d+)\s+(?:and|to)\s+(\d+)'
    match = re.search(pattern2, question, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    # Pattern 3: "X to Y" (generic numbers)
    pattern3 = r'(\d+)\s+(?:to|and)\s+(\d+)'
    match = re.search(pattern3, question, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    return (None, None)


def handle_question(question, data):
    """
    Main function to handle user questions:
    1. Classify the question using LLM
    2. Extract frame range if needed
    3. Call appropriate answer function
    
    Args:
        question (str): User's question (can be indirect/non-obvious)
        data (dict): Dictionary containing all necessary data:
            - For Q1: ball_position, ball_velocity, player_positions, table_coords
            - For Q2: start_frame, end_frame, ball_positions, ball_bounces, table_coords
    
    Returns:
        str or dict: The answer/result
    """
    
    print(f"\n{'='*70}")
    print(f"Processing Question: {question}")
    print(f"{'='*70}\n")
    
    # Step 1: Classify the question using LLM
    print("Classifying question with Phi-3 Mini LLM...")
    classification = classify_question_with_llm(question)
    
    if classification == 0:
        print("❌ Classification Result: UNKNOWN / NOT ENOUGH DATA")
        return "I couldn't classify this question. Please ask about either:\n" \
               "1. Why a player lost or performed poorly during a specific time period\n" \
               "2. How the ball moved/bounced during a specific time period"
    
    elif classification == 1:
        print("✅ Classification Result: QUESTION 1 (Why did player lose?)")
        print("\nCalling answer_question_1()...")
        
        # Validate required data for Q1
        required_keys = ['ball_position', 'ball_velocity', 'player_positions', 'table_coords']
        if not all(key in data for key in required_keys):
            return f"❌ Missing data for Question 1. Required: {required_keys}"
        
        try:
            result = answer_question_1(
                data['ball_position'],
                data['ball_velocity'],
                data['player_positions'],
                data['table_coords']
            )
            print(f"\n✅ Answer: {result}")
            return result
        except Exception as e:
            return f"❌ Error processing Question 1: {str(e)}"
    
    elif classification == 2:
        print("✅ Classification Result: QUESTION 2 (Ball bounce trajectory)")
        print("\nExtracting frame range from question...")
        
        start_frame, end_frame = extract_frame_range(question)
        
        if start_frame is None or end_frame is None:
            print("⚠️ Could not extract frame range. Using data['start_frame'] and data['end_frame']")
            start_frame = data.get('start_frame')
            end_frame = data.get('end_frame')
        
        print(f"Frame range: {start_frame} to {end_frame}")
        print("Calling answer_question2()...")
        
        # Validate required data for Q2
        required_keys = ['ball_positions', 'ball_bounces', 'table_coords']
        if not all(key in data for key in required_keys):
            return f"❌ Missing data for Question 2. Required: {required_keys}"
        
        if start_frame is None or end_frame is None:
            return "❌ Could not extract frame range from question"
        
        try:
            result = answer_question2(
                start_frame,
                end_frame,
                data['ball_positions'],
                data['ball_bounces'],
                data['table_coords']
            )
            print(f"\n✅ Answer retrieved {len(result)} bounces")
            return result
        except Exception as e:
            return f"❌ Error processing Question 2: {str(e)}"


# ============================================================================
# EXAMPLE USAGE WITH INDIRECT QUESTIONS
# ============================================================================

if __name__ == "__main__":
    
    # Sample data (replace with your actual data)
    sample_data = {
        # For Question 1
        'ball_position': (0.5, 0.3),
        'ball_velocity': (0.1, -0.2),
        'player_positions': {
            'player1': (0.2, 0.3),
            'player2': (0.7, 0.3)
        },
        'table_coords': [(0, 0), (2, 0), (2, 1), (0, 1)],
        
        # For Question 2
        'start_frame': 10,
        'end_frame': 50,
        'ball_positions': {
            '10': (0.4, 0.2), '11': (0.41, 0.19), '12': (0.42, 0.18),
            '20': (0.5, 0.3), '21': (0.51, 0.29),
            '30': (0.6, 0.4), '31': (0.61, 0.39),
            '50': (0.8, 0.5)
        },
        'ball_bounces': [
            {'frameID': 20},
            {'frameID': 35}
        ]
    }
    
    # Test Question 1 - INDIRECT: Why player lost
    print("\n" + "="*70)
    print("TEST 1: Indirect Question - Why did player lose?")
    print("="*70)
    question1 = "What went wrong with the player on the right between frames 10 and 50? The opponent capitalized on something..."
    answer1 = handle_question(question1, sample_data)
    print(f"\nFinal Answer:\n{answer1}")
    
    # Test Question 2 - INDIRECT: Ball bounce trajectory
    print("\n" + "="*70)
    print("TEST 2: Indirect Question - Ball bounce trajectory")
    print("="*70)
    question2 = "I need to understand what happened with the ball during the exchange between frames 10 and 50. Can you trace its path?"
    answer2 = handle_question(question2, sample_data)
    print(f"\nFinal Answer:\n{answer2}")
    
    # Test Question 3 - INDIRECT: Another way to ask about player loss
    print("\n" + "="*70)
    print("TEST 3: Indirect Question - Another way to ask why player lost")
    print("="*70)
    question3 = "Looking at frames 10 to 50, explain why the player on the right couldn't capitalize. There seems to be a positioning issue..."
    answer3 = handle_question(question3, sample_data)
    print(f"\nFinal Answer:\n{answer3}")
    
    # Test Question 4 - INDIRECT: Asking about ball dynamics
    print("\n" + "="*70)
    print("TEST 4: Indirect Question - Ball dynamics during rally")
    print("="*70)
    question4 = "Show me the dynamics of the ball movement during the rally from frame 10 to 50. How did it bounce?"
    answer4 = handle_question(question4, sample_data)
    print(f"\nFinal Answer:\n{answer4}")
    
    # Test Question 5 - UNKNOWN
    print("\n" + "="*70)
    print("TEST 5: Unknown Question")
    print("="*70)
    question5 = "What is the weather like today in table tennis?"
    answer5 = handle_question(question5, sample_data)
    print(f"\nFinal Answer:\n{answer5}")
    
    # Test Question 6 - INDIRECT: More ambiguous
    print("\n" + "="*70)
    print("TEST 6: Indirect Question - Ambiguous phrasing")
    print("="*70)
    question6 = "During the exchange between 10 and 50, the right player seemed to struggle. What factors contributed?"
    answer6 = handle_question(question6, sample_data)
    print(f"\nFinal Answer:\n{answer6}")
