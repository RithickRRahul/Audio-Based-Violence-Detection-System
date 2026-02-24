import os
import pandas as pd
import random

def generate_distress_dataset(output_dir="datasets/custom_distress"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Physical Assault / Danger (Toxic = 1)
    violent_phrases = [
        # Direct physical threats
        "I'm going to kill you", "I have a gun", "Put your hands up", "Drop the weapon",
        "Give me all your money", "Don't move or I'll shoot", "Get on the ground",
        "I'll stab you", "Shut up or you're dead", "I'm going to break your neck",
        "I'll beat the shit out of you", "Stop resisting", "Get in the car now",
        "Don't make me use this", "I'll blow your head off", "You're going to die",
        
        # Pleas for help / Distress
        "Help me!", "Please stop", "Don't hurt me", "Somebody help",
        "He has a knife", "Call the police", "Get off me", "Leave me alone",
        "Ahhhhh!", "No please no", "I'm bleeding", "Stop hitting me",
        "Please don't do this", "Let me go", "Help! He's attacking me",
        "I didn't do anything, stop!", "Ouch! Stop!", "Somebody call 911"
    ]
    
    # Expand violent phrases with variations
    violent_expanded = []
    for phrase in violent_phrases:
        violent_expanded.append(phrase)
        violent_expanded.append(phrase.upper())
        violent_expanded.append(phrase + "!!!")
        violent_expanded.append("Hey! " + phrase)
        violent_expanded.append(phrase + " man!")
        
    num_violent = len(violent_expanded)
    
    # 2. Everyday Safe Speech (Toxic = 0)
    safe_phrases = [
        # Normal conversations
        "How are you doing today?", "Can I get a coffee please", "What time is it?",
        "Have a good weekend", "I'll see you tomorrow", "Did you watch the game?",
        "I'm going to the store", "That's a great idea", "Let's grab lunch",
        "Where is the bathroom?", "I agree with you", "Sorry I'm late",
        "It's sunny outside", "Can you send me that file?", "I love this song",
        "Happy birthday!", "Thank you so much", "No problem", "Excuse me",
        
        # Loud but non-violent (Sports, excitement)
        "Pass the ball!", "He scores!", "Wow that was amazing!", "Let's go team!",
        "Hurry up we're going to be late", "I can't believe we won!", "Shoot the ball!",
        
        # Arguments, but non-physical (Internet trolls, basic disagreements)
        "You are an idiot", "That's the stupidest thing I've ever heard", 
        "I disagree completely", "You have no idea what you're talking about",
        "You're a terrible driver", "I'm so angry at you", "This is awful service",
        "I hate this movie", "You suck!", "Get out of my way"
    ]
    
    # Expand safe phrases
    safe_expanded = []
    for phrase in safe_phrases:
        safe_expanded.append(phrase)
        safe_expanded.append(phrase.upper())
        safe_expanded.append(phrase + "!")
        safe_expanded.append("Hey, " + phrase)
        safe_expanded.append(phrase + " right?")
    
    # Ensure classes are balanced
    if len(safe_expanded) > num_violent:
        safe_expanded = random.sample(safe_expanded, num_violent)
    elif len(safe_expanded) < num_violent:
        extra_needed = num_violent - len(safe_expanded)
        safe_expanded.extend(random.choices(safe_expanded, k=extra_needed))
        
    # Build dataframe
    data = []
    for text in violent_expanded:
        data.append({"comment_text": text, "is_toxic": 1})
    for text in safe_expanded:
        data.append({"comment_text": text, "is_toxic": 0})
        
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    csv_path = os.path.join(output_dir, "train.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Generated Custom Physical Distress Dataset at {csv_path}")
    print(f"Total phrases: {len(df)} (Violent: {len(df[df['is_toxic'] == 1])}, Safe: {len(df[df['is_toxic'] == 0])})")
    
if __name__ == "__main__":
    generate_distress_dataset()
