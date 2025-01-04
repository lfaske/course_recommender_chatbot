import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from chatbot_variables import intent_replies, module_dict
import string
import streamlit as st

### --- Handles data, recommendation generation, feedback interpretation, and user profiles


@st.cache_resource 
def load_model():
    # Load the pretrained model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    return model

@st.cache_data
def load_json_data():
    """
    Loads and caches the course data from the JSON file.

    Returns:
        lists with current courses, past courses, and all attributes
    """
    # Load the course and embedding data (created and saved in data_prep.py)
    with open("courses.json", "r") as file:
        courses = json.load(file)
        current_courses = courses["current"]
        past_courses = courses["past"]
        all_attributes = courses["all_attr"]
    return current_courses, past_courses, all_attributes
    
@st.cache_data
def load_npz_data():
    """
    Loads and caches the embeddings from the NPZ file
    
    Returns:
        dictionaries with embeddings of current courses, past courses, and intents
    """
    loaded_embeddings = np.load('embeddings.npz', allow_pickle=True)
    current_embeddings = dict(loaded_embeddings['current_courses'].item())
    past_embeddings = dict(loaded_embeddings['prev_courses'].item())
    intent_embeddings = dict(loaded_embeddings['intent'].item())
    return current_embeddings, past_embeddings, intent_embeddings



def get_past_idx(title):
    """
    Returns the index of a past course given it's title
    
    Parameter:
        title (str): title of the course
    Returns:
        index (int)
    """
    _, past_courses, _ = load_json_data()
    idx = [c['title'] for c in past_courses].index(title)
    if isinstance(idx, int):
        return idx
    else:
        return None

def get_past_title(idx):
    """
    Returns the title of the course with the given index in the list of past courses

    Parameter:
        idx (int): index of the course
    Returns:
        title (str) of the course
    """
    _, past_courses, _ = load_json_data()
    if isinstance(idx, int) and idx < len(past_courses):
        return past_courses[idx]['title']
    else:
        return None
    
def get_details(idx):
    """
    Returns dictionary with all attributes of a current course

    Parameter:
        idx: index of the course to return
    Returns:
        all attributes of the course
    """
    current_courses, _, _ = load_json_data()
    return current_courses[idx]

def get_all_filter():
    """
    Returns all attributes used for filters
    """
    _, _, all_attributes = load_json_data()
    filter_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'home_institute', 'language', 'filter_time']
    all_filter = {key: all_attributes[key] for key in filter_attributes}
    return all_filter

def weighted_cosine_similarity(embeddings1, embeddings2):
    """
    Compute the weighted cosine similarity between two embeddings.
    
    Parameters:
        embeddings1: Dict of attribute embeddings (of a course or input)
        embeddings2: Dict of attribute embeddings (of a course or input)
    Returns:
        Weighted cosine similarity (float)
    """
    """weights = {
        'title_descr': 0.7,  
        'module': 0.1, 
        'home_institute': 0.1,
        'status': 0.025, 
        'mode': 0.025,
        'lecturer_short': 0.025,
        'area': 0.025,
    }"""
    weights = {
        'title_descr': 0.7,  
        'module': 0.1, 
        'home_institute': 0.1,
        'lecturer_short': 0.04,
        'area': 0.04,
        'status': 0.01, 
        'mode': 0.01,
    }
    similarities = []
    # Compute the cosine similarity for each of the attributes 
    for attr, weight in weights.items():
        u = embeddings1[attr]
        v = embeddings2[attr]
        # Normalize the embeddings
        u_norm = u / np.linalg.norm(u) if np.linalg.norm(u) != 0 else u
        v_norm = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
        # Calculate the similarity of the normalized embeddings
        similarity = np.dot(u_norm, v_norm)
        # Append the similarity multiplicated with the specified weight
        similarities.append(weight * similarity)
    # Return the sum of all computed similarities
    return sum(similarities)
    

###--- Handle Intent Detection ---###

def detect_intent(user_input, last_recommendations):
    """
    Detects the intent of a user's input

    Parameters:
        user_input (str): the user's input
        last_recommendations (list): list of the last recommended courses
    Returns:
        detected_intent: the intent it detected
        intent_replies[detected_intent] or chatbot_reply: the chatbots reply based on the intent it detected
        detected_courses: tuple (course index, x) with x being either the similarity (float) between the title and user input (if detected_intent == reference) or the feedback for the course ('liked' or 'disliked') (if detected_intent == feedback)
    """
    model = load_model()
    _, _, intent_embeddings = load_npz_data()
    intent_similarities = {'free_description': 0.0, 'liked_course_reference': 0.0, 'feedback': 0.0}
    user_embedding = model.encode([user_input])[0]
    detected_intent = "other"  # Default intent

    # First check if the user has explicitly stated the intent at the beginning of the message
    if user_input.lower().startswith("free:"):
        return "free_description", intent_replies["free_description"], []
    elif user_input.lower().startswith("ref:"):
        chatbot_reply, intent_result = check_intent("liked_course_reference", user_input, user_embedding, last_recommendations)
        if chatbot_reply != "":
            return "liked_course_reference", chatbot_reply, intent_result
        else:
            return "other", "I'm sorry, but I don't understand which course you are referring to. Please make sure that the name is correct."
    elif user_input.lower().startswith("feedback:"):
        chatbot_reply, intent_result = check_intent("feedback", user_input, user_embedding, last_recommendations)
        return "feedback", chatbot_reply, intent_result
    
    # If no intent was given, select the one which is most similar to the input
    else:
        # Compare user input with each intent category
        for intent, examples in intent_embeddings.items():
            # Only allow feedback if there have been recommendations before
            if intent == 'feedback' and len(last_recommendations) == 0:
                intent_similarities[intent] = 0.0
                continue
            intent_similarities[intent] = float(cosine_similarity(np.array([user_embedding]), examples).max())

        # Get the intent with the highest similarity
        sorted_intents = dict(sorted(intent_similarities.items(), key=lambda item: item[1])[::-1])
        print(f"*** Intent similarities: {sorted_intents}")
        it = iter(sorted_intents.items())

        for detected_intent in it:        
            # If the similarity score of the intent with the highest similarity to the input is below the threshold, return "other"
            if detected_intent[1] < 0.4 and detected_intent[1] != "nonsense":
                return "other", intent_replies["other"], []
            
            # If it is one of the intents that just return the corresponding reply, directly return it
            #elif detected_intent[0] in ["greeting", "free_description", "no_info", "nonsense", "other"]:
            elif not (detected_intent[0] in ["feedback", "liked_course_reference"]):
                return detected_intent[0], intent_replies[detected_intent[0]], []
            
            # For the other intents, call the function check_intent
            chatbot_reply, intent_result = check_intent(detected_intent, user_input, user_embedding, last_recommendations)

            # If the chatbot reply is not empty, return the current detected intent; otherwise, the loop continues to the next most similar intent
            if chatbot_reply != "":
                return detected_intent[0], chatbot_reply, intent_result
            
    # The function should have returned something before reaching this line; if, for some reason, it didn't, return 'other' as intent
    return "other", intent_replies["other"], []


def check_intent(detected_intent, user_input, user_embedding, last_recommendations):
    """
    Handles references to previously liked courses and feedback.

    Parameter:
        detected_intent (str): The intent that was detected
        user_input (str): The user's message
        user_embedding (ndarray): The embedding of the user input
        last_recommendations (list): The last courses that were recommended
    Returns:
        The reply based on the intent
        The list of found courses (with either a tuple containing the referenced course and the similarity, or tuples with the rated courses and the corresponding ratings)
    """
    current_courses, past_courses, _ = load_json_data()
    _, past_embeddings, _ = load_npz_data()
    if detected_intent[0] == "liked_course_reference":
        # First check if a title is spelled out exactly
        best_fit = ("", 0.0)
        all_titles = [c['title'] for c in past_courses]
        for title in all_titles:
            if title.lower() in user_input.lower():
                idx = get_past_idx(title)
                # As only one reference per message is allowed, directly return the course
                return intent_replies[detected_intent[0]], [(idx, 1.0)]
            
        # Set the threshold lower if the similarity for the intent is very high (usually only the case if it is manually set by the user)
        if detected_intent[1] >= 0.99:
            title_threshold = 0.2
        else:
            title_threshold = 0.5

        # Check if the input is similar enough to any of the titles
        for title, attr in past_embeddings.items():
            title_sim = cosine_similarity([user_embedding], [attr['title']])
                
            # If the similarity score of the title is above the threshold and higher than the currently highest one, save the courses index and the score as best fit
            if title_sim > title_threshold and title_sim > best_fit[1]:
                idx = get_past_idx(title)
                best_fit = (idx, title_sim)
                
        # Return the best fitting title, if any was found with a similarity above the threshold
        if best_fit[1] > title_threshold:
            return intent_replies[detected_intent[0]], [best_fit]
        else: 
            return "", []
        
    elif detected_intent[0] == "feedback":
        c_feedback = give_feedback(user_input, last_recommendations)
        # If a list with at least one course (by it's position in the list of recommendations) with a rating (positive/negative) was found in the input, return it with the corresponding reply
        if len(c_feedback) > 0:
            chatbot_reply = "You gave the following feedback:  \n"
            for (c, f) in c_feedback:
                chatbot_reply += f"- {current_courses[c]['title']}: {f}  \n"
                chatbot_reply += "  \n"
            return chatbot_reply, c_feedback
        
        # Otherwise, ask the user to clarify
        else:
            chatbot_reply = "I think you wanted to give feedback for one or more of the recommendated courses, but I could not clearly understand you. Could you please clarify? To learn more about how to properly give feedback, click on the button 'Feedback Hint' below the chat."
            return chatbot_reply, []
    else:
        return "", []


###--- Handle Feedback ---###

def find_feedback_courses(sentence):
    """
    Find the index/indices of the course(s) the user gave feedback for

    Parameters:
        sentence (str): a sentence that might include a reference to a course
    Return:
        either list of indices (int) or list containing a string ('all'/'none')
    """
    # Sorting words to the corresponding positions
    c_position = [("last", "final"), ("1", "one", "first"), ("2", "two", "second"), ("3", "three", "third"), ("4", "four", "fourth"), ("5", "five", "fifth")]
    position_keys = {word: idx for idx, tpl in enumerate(c_position) for word in tpl}
    merge_numbers = {"second one": "2", "third one": "3", "fourth one": "4", "fifth one": "5", "last one": "0"}
    all_courses = {"all": "all", "any": "all", "every": "all", "everyone": "all", "none": "none", "no": "none"}
    
    # Make the sentence cases-insensitive and split it into separate words
    split_input = sentence.replace("-", " - ")  # Add a space before and after - to count it as a word, separating numbers in a range
    split_input = split_input.replace(",", " , ").lower().split()  # Add a space before each comma to count it as a word, separating numbers in an enumeration
    
    # First check if the user gave feedback for all courses simultaneously
    mentioned_all = [all_courses[word] for word in split_input if word in all_courses]
    mentioned_all = list(set(mentioned_all))  # Remove duplicates
    if len(mentioned_all) == 2:  # If the sentence includes both 'none' and 'all', interpret it as 'none' ###?!? ODER DANN BESSER IGNORIEREN???
        return ["none"]
    elif len(mentioned_all) == 1:
        return mentioned_all

    # Check if the user mentioned specific positions or ranges
    idx_matches = []
    skip_amount = 0
    range_start = -1
    for idx, word in enumerate(split_input):
        position = -1 
        # Skip the word if it was merged with the previous one or was detected to be a range indicator
        if skip_amount > 0:
            skip_amount = skip_amount-1
            continue
        # Replace word pairs like "fourth one" with a single number (in this case: "4"; to avoid interpreting it as "course 4 and course 1")
        elif len(split_input) > idx + 1:
            next_merge = " ".join([word, split_input[idx+1]])
            if next_merge in list(merge_numbers):
                position = int(merge_numbers[next_merge])-1
                skip_amount += 1
            elif word in position_keys:
                position = int(position_keys[word])-1
        elif word in position_keys:
            position = int(position_keys[word])-1
        # If the last word was detected to be the start of a range, set the range
        if range_start >= 0:
            if range_start < position:
                idx_matches += list(range(range_start, position+1))
                range_start = -1  # Reset the start of the range
                continue
            # If the current word is smaller than the previous one, set the previous as a single value
            else:
                idx_matches.append(range_start)
                range_start = -1  # Reset the start of the range
        # If a position was found, check if it is the first number in a range (followed by a range indicator)
        if position >= 0 and len(split_input) > (idx + skip_amount + 2) and split_input[idx + skip_amount + 1] in ["-", "to"]:
            range_start = position  # Don't append the range here, as the second number might be a phrase like "second one" that has to be merged first
            skip_amount += 1  # As the next word is a range indicator, it can directly be skipped
        # If the position was not detected to be the start of a range, append it as a single value
        if range_start == -1 and position >= 0:
            idx_matches.append(position)
    # If a start of a range was detected but no end, append the start as single value
    if range_start >= 0:
        idx_matches.append(range_start)
    # Remove duplicates
    idx_matches = list(set(idx_matches))
    return idx_matches

def sentence_sentiment(sentence):
    """
    Interpret if the user likes or dislikes the courses mentioned in the given sentence

    Parameters:
        sentence (str): sentence to check the sentiment of
    Returns:
        'liked', 'disliked' or 'negation', or None if no sentiment was found
    """
    # Include variations for each sentiment
    sentiment_dict = {
        "liked": ["liked", "like", "love", "interesting", "good", "awesome", "nice", "great"],
        "disliked": ["dislike", "hate", "boring", "bad"],
        "negation": ["not", "don't", "didn't", "doesn't", "aren't", "isn't"]
    }
    # Reverse the dictionary for a more efficient lookup
    sentiment_key = {word: key for key, values in sentiment_dict.items() for word in values}

    # Split the sentence into words
    split_feedback = re.sub(r"[^\w\s']", '', sentence).lower().split()
    matches = list(set([sentiment_key[word] for word in split_feedback if word in sentiment_key]))

    # If exactly one sentiment is found, return it
    if len(matches) == 1 and matches[0]:
        return matches[0]
    # If only 'negation' and 'liked' are found (e.g., "I don't like course 1"), return 'disliked'
    # Not the other way around, because phrases like "I don't hate it" don't necessarily mean that it's liked
    elif len(matches) == 2 and 'negation' in matches and 'liked' in matches:
        return 'disliked'
    return None

def give_feedback(user_input, last_recommendations):
    """
    Processes the user's feedback

    Parameters:
        user_input (str): the user's feedback
        last_recommendations (list): list of the indices of the last recommended courses
    Return:
        List of tuples with course indices (from current_courses) and corresponding feedback 
    """
    # If only one course was recommended, directly check the sentiment
    if len(last_recommendations) == 1:
        sentiment = sentence_sentiment(user_input)
        if sentiment is None:
            return[]
        else:
            return [(last_recommendations[0], sentiment)]

    # Split the input into parts separated by 'but'
    parts = []
    if 'but' in user_input.lower():
        start = 0
        # Find all occurrences of 'but'
        while True:
            index = user_input.lower().find('but', start)
            if index == -1:
                break
            
            # Add the part before the occurence to the list and move the start position to the index after the occurence
            parts.append(user_input[start:index].strip())
            start = index + 3

        # Add the remaining part (after last 'but')
        parts.append(user_input[start:].strip())
    else:
        parts.append(user_input)

    # Split each part of the input by punctuation (.!?) to separate the sentences (if there are multiple)
    parts = [re.split(r'[.!?]', sent.strip()) for sent in parts]

    # Combine the parts (currently list of lists) to a one dimensional list and remove empty elements
    sentences = []
    for s in parts:
        sentences.extend(s)
    sentences = list(filter(None, sentences))
    
    # Find course references and sentiments for each sentence
    given_feedback = []
    for s in sentences:
        courses = find_feedback_courses(s)
        # If no course was found, skip the sentence
        if len(courses) == 0:  
            continue
        
        c_sentiment = sentence_sentiment(s)

        # If no sentiment was found, skip the sentence
        if c_sentiment is None:
            continue

        # If the sentiment is 'negation', use the opposite of the previous sentence's sentiment (if any)
        if c_sentiment == 'negation' and len(given_feedback) > 0:
            c_sentiment = given_feedback[-1][1]
            c_sentiment = 'liked' if given_feedback[-1][1] == 'disliked' else 'disliked'
        # If it is negation but there have not yet been sentences with a detected sentiment, skip the sentence
        elif c_sentiment == 'negation':  
            continue

        # If the user gave feedback for all courses simultaneously, set all courses to the sentiment
        # If the user stated that they like none of the recommended courses, set all to 'dislike'
        if courses[0] == 'all' or (courses[0] == 'none' and c_sentiment == 'liked'):
            given_feedback = [(c, c_sentiment) if courses[0] == 'all' else (c, 'dislike') for c in last_recommendations]

        # If the user specified the positions of the courses they (dis)liked, set each mentioned course separately to the sentiment
        elif isinstance(courses[0], int):
            for i in courses:
                if i >= 0 and i < len(last_recommendations):
                    given_feedback.append((last_recommendations[i], c_sentiment))
    return given_feedback


###--- User Preference Management ---### 

def update_user_preferences(user_profile, input_embedding = None, rated_course = None, liked=True, learning_rate=0.5):
    """
    Updates the user's preferences according to a free description, a reference to a liked course, or given feedback

    Parameters: 
        user_profile (dict): current embedding of the user's preferences (embeddings for each attribute)
        input_embedding (dict)*: embedding of the user's free description (embeddings for each attribute)
        rated_course ((int, str))*: either (index, 'past') of a previously liked course or (index, 'current') of a course the user gave feedback for
        liked (bool): if the user liked the course / the recommendation of the course; for descriptions always True
        learning_rate (float): how strong the feedback should influence the user's preferences
        * either input_embedding or rated_course is necessary
    Returns: 
        updated user_profile
    """
    current_embeddings, past_embeddings, _ = load_npz_data()
    # If no input_embedding is given, set it to the embedding of the rated course
    if input_embedding is None:
        if rated_course[1] == 'past':
            input_embedding = list(past_embeddings.values())[rated_course[0]] 
        else:
            input_embedding = list(current_embeddings.values())[rated_course[0]]
    if user_profile is not None:
        if liked:  # Positive feedback, liked course or free description
            for attr in user_profile:
                # Make the embedding of the user's preferences more similar to the input or liked course
                user_profile[attr] += learning_rate * input_embedding[attr]  
        else:      # Negative feedback
            for attr in user_profile:
                # Make the embedding of the user's preferences less similar to the disliked course
                user_profile[attr] = user_profile[attr] - (learning_rate * input_embedding[attr])
    else:
        user_profile = {}
        # If no profile exists, set user's embedding to the input embedding
        for attr in ['title_descr', 'module', 'home_institute', 'status', 'mode', 'lecturer_short', 'area']:
            user_profile[attr] = input_embedding[attr]
    return user_profile


###--- Filter Management ---###

def find_sws_ects(user_input, old_filter):
    """
    Get SWS and ECTS from user input (free descriptions)

    Parameter:
        user_input (str): Free description from the user
        old_filter (): Previously set filters
    Returns:
        The found SWS, ECTS, and cleaned input
    """
    sws_ects_pattern = r"""
        (?:\b|\s\.)
        (
            (\d+)                           # First sws/ects of range
            (?:\sto\s|\-|\s\-\s|\sand\s)    # Range indicator
            (\d+\s?sws|\d+\s?ects)          # Last sws/ects of range (with indicator)
        |
            (\d+\s?sws|\d+\s?ects)          # Single mention
        )
        """
    # Get all mentioned SWS and ECTS from the input
    matches = re.findall(sws_ects_pattern, user_input, re.VERBOSE | re.IGNORECASE)
    found_matches = [m[0] for m in matches]

    # If there are no matches, return the old filter
    if not found_matches:
        return old_filter['sws'], old_filter['ects'], user_input

    cleaned_input = user_input
    found_sws = []
    found_ects = []
    
    for match in found_matches:
        # Remove SWS and ECTS from input to avoid misinterpreting them as times
        cleaned_input = cleaned_input.replace(match, "")
        # Extract the attribute and number(s) from the match
        attr = re.search(r'(ects|sws)', match, flags = re.IGNORECASE).group().strip().lower()
        numbers = [int(nr) for nr in re.findall(r'\d+', match)]
        numbers = [min(numbers), max(numbers)]
        
        # Add the smallest and biggest number found yet to the found SWS/ECTS
        if attr == 'sws':
            if found_sws == []:
                found_sws = numbers
            else:
                found_sws = [min(found_sws + numbers), max(found_sws + numbers)]
        elif attr == 'ects':
            if found_ects == []:
                found_ects = numbers
            else:
                found_ects = [min(found_ects + numbers), max(found_ects + numbers)]

    # Convert SWS and ECTS to strings
    if found_sws:
        found_sws = [str(found_sws[0]), str(found_sws[1])]
    if found_ects:
        found_ects = [str(found_ects[0]), str(found_ects[1])]
    return found_sws, found_ects, cleaned_input

def input_times(user_input, old_filter):
    # Abbreviations for the days
    weekdays = {'monday': 'Mon.', 'tuesday': 'Tue.', 'wednesday': 'Wed.', 'thursday': 'Thu.', 'friday': 'Fri.', 'saturday': 'Sat.',
                'every day': 'alldays', 'each day': 'alldays', 'any day': 'alldays', 'all days': 'alldays', 'everyday': 'alldays', 'every single day': 'alldays', 
                'other day': 'otherdays', 'other days': 'otherdays', 'remaining days': 'otherdays'}
    all_weekdays = ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.']
    
    # Regex pattern to match weekdays (multiple days / ranges as well as individual days)
    abbrev_weekdays_pattern = r"""
        (?:\b|\s)
        (
            (from\s|between\s)? 
            (Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.)          # First day of range
            (\sto\s|\-|\s\-\s|\sor\s|\sand\s)                              # Range indicator
            (Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.)        # End time
        |
            (Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.|alldays|otherdays)         # Single day
        )
        """
    found_days = []  # List to store all found days
    found_day_time = {}  # days as keys, corresponding times as values

    # Replace full weekdays with abbreviations
    for full_day, abbrev in weekdays.items():
        user_input = re.sub(rf'\b{full_day}s?\b', abbrev, user_input, flags=re.IGNORECASE)

    # Find all days (or ranges of days) mentioned in the input
    matches = re.findall(abbrev_weekdays_pattern, user_input, re.VERBOSE | re.IGNORECASE)
    found_days = [m[0] for m in matches]

    # If there are no days mentioned in the input, return the dictionary from the old filter
    if not found_days:
        return old_filter

        
    ################################################################
    ##### T I M E S
    ################################################################

    # Replace textual times with digits
    textual_time_map = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", 
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", 
        "eleven": "11", "twelve": "12"
    }
    for text_time, numeric_time in textual_time_map.items():
        user_input = re.sub(rf'\b{text_time}\b', numeric_time, user_input)

    # Regex pattern for times
    time_pattern = r"""
        \b
        (
            (?:at\s|from\s|between\s)?                                                              # Optional range indicator at start
            (half\spast\s|quarter\sto\s|quarter\spast\s)?                                           # Optional time modifiers
            ([1-9]|1[0-9]|2[0-3])(:[0-5][0-9])?                                                     # Starting time
            ((\s|\.|\b)?(?:o'clock\s)?(?:in\s)?(?:the\s)?(morning|afternoon|evening|AM|PM|am|pm)?)  # Optional modifiers
            (\sto\s|\sand\s|\sand\send\sat\s|\suntil\s|\-|\s\-\s)                                   # Range indicator
            (half\spast\s|quarter\sto\s|quarter\spast\s)?                                           # Optional time modifiers
            ([1-9]|1[0-9]|2[0-3])(:[0-5][0-9])?                                                     # End time
            ((\s|\.|\b)?(?:o'clock\s)?(?:in\s)?(?:the\s)?(morning|afternoon|evening|AM|PM|am|pm)?)  # Optional modifiers
        |
            (at\s|half\spast\s|quarter\sto\s|quarter\spast\s)?  # Optional time indicators
            ([1-9]|1[0-9]|2[0-3])(:[0-5][0-9])?                 # Hours and optional minutes
            (\s|\.|\b)?                                         # Optional space, period, or word boundary
            (?:o'clock\s)?                                      # Optional "o'clock"
            (?:in\s)?                                           # Optional "in"
            (?:the\s)?                                          # Optional "the"
            (morning|afternoon|evening|AM|PM|am|pm)?            # Optional time of day
        )
        \b
        """

    def extract_times(input_part):
        """
        Get the times from the given part of the input

        Parameter:
            input_part (str): part of the input that is checked for times
        """
        # Find all parts of the input part that match the pattern for times
        found_phrases = []
        found_times = []
        matches = re.findall(time_pattern, input_part, re.VERBOSE)
        found_phrases = [m[0] for m in matches]

        # If no times were found: Return whole day (7:00 - 21:00, as there are only courses between 8:00 and 20:00 (added 1h to be safe))
        if len(found_phrases) == 0:
            return [[7, 21]]

        # For each found time, check for modifiers
        for time in found_phrases:
            numbers = re.findall(r'(\d+\:\d+|\d+)', time)
            if len(numbers) == 1:
                number = re.search(r'\d+', numbers[0])
                hour = int(number.group()) # Convert matched number to integer
                if bool(re.search(r'(\b|\d+)(pm|afternoon|evening)\b', time.lower())) and hour <= 12:
                    hour += 12
                found_times.append([hour, hour+2])
                    
            # Check ranges
            elif len(numbers) == 2:
                start_number = re.search(r'\d+', numbers[0])
                end_number = re.search(r'\d+', numbers[1])
                start_hour = int(start_number.group()) # Convert matched start number to integer
                end_hour = int(end_number.group()) # Convert matched end number to integer
                has_am = bool(re.search(r'(\b|\d+)(am|morning)\b', time.lower()))
                has_pm = bool(re.search(r'(\b|\d+)(pm|afternoon|evening)\b', time.lower()))

                # If both am and pm are found or end is smaller than start, end is pm
                if ((has_am and has_pm) or (end_hour < start_hour)) and end_hour <= 12:
                    end_hour += 12

                # If only pm is found and end_hour is bigger than the start, assume pm for both
                elif has_pm and end_hour > start_hour and end_hour <= 12:
                    start_hour += 12
                    end_hour += 12
                found_times.append([start_hour, end_hour])
        return found_times
    

    def add_time(days, times):
        """
        Add the given times to the given days

        Parameters:
            day (list): list of all days the times should be added to
            times (list): list of times to add to all given days
        """
        # If the given days or times are not lists, convert them to lists
        if not isinstance(times, list):
            times = [times]
        if not isinstance(days, list):
            days = [days]
        # Add each given time to each given day
        for day in days:
            for time in times:
                if day in found_day_time:
                    found_day_time[day].append(time)
                else:
                    found_day_time[day] = [time]
    
    # Split the input text at day or day range boundaries
    split_pattern = "|".join(found_days)
    parts = re.split(rf"(?={split_pattern})", user_input, flags=re.IGNORECASE)
    cleaned_parts = [part for part in parts if any(day in part for day in found_days)]
    parts_dict = dict(zip(cleaned_parts, found_days))

    for part, days in parts_dict.items():
        # Get the list of times for the current part of the input
        times = extract_times(part)

        # Check if 'alldays' is given -> set times for all days of the week
        if days == 'alldays':
            add_time(all_weekdays, times)
        
        # Check if 'otherdays' is given -> set times for all days that have no time yet
        elif days == 'otherdays':
            other_days = [d for d in all_weekdays if not (d in found_day_time.keys())]
            add_time(other_days, times)

        # Check if a range of days is given
        elif any(range_indicator in days for range_indicator in ['between', '-', 'to']):
            found_day_range = []
            for range_day in all_weekdays:
                if range_day in days:
                    found_day_range.append(range_day)
                    # If other days have already been appended, this was the end of the range
                    if len(found_day_range) > 1:
                        break
                # If this day is not in this part of the input, only append it if the first day of the range was already appended
                elif len(found_day_range) > 0:
                    found_day_range.append(range_day)
            add_time(found_day_range, times)

        # Otherwise, set times for the individual days in the current part of the input
        else:
            found_individual_days = []
            for ind_day in all_weekdays:
                if ind_day in days:
                    found_individual_days.append(ind_day)
            add_time(found_individual_days, times)

    # Add the new times to the previously set times
    if old_filter == []:
        old_filter = {}
    for found_day, found_times in found_day_time.items():
        if found_day in old_filter:
            old_filter[found_day] += found_times
        else:
            old_filter[found_day] = found_times

    # Merge overlapping timeframes
    merged_times = {}
    for found_day, found_times in old_filter.items():
        if not found_times:
            merged_times[found_day] = []
            continue
        found_times.sort(key=lambda x: x[0])

        # Initialize the merged list with the first interval
        merged = [found_times[0].copy()]
        
        for current in found_times[1:]:
            previous = merged[-1]
            
            # Check for overlap
            if current[0] <= previous[1]:
                previous[1] = max(previous[1], current[1])
            else:
                # If there is no overlap, add the interval to the result
                merged.append(current)
        merged_times[found_day] = merged

    # Sort merged times by days
    merged_times = {day: merged_times[day] for day in all_weekdays if day in merged_times}
    return merged_times


def find_modules(user_input, old_filter):
    """
    Find modules in given input.
    Module names are constructed from 
    -- "CS-" for Cognitive Science (dataset includes only Cognitive Science courses, therefore all modules start with "CS-")
    -- "[B/M][WP/W/P]-": "B" if Bachelor, "M" if Master; "WP" if elective, "P" if compulsory, "W" if "Distinguishing Elective Courses"
    -- Short form of the area (e.g. "AI")

    Parameter:
        user_input (str): the user's input in which modules should be found
        old_filter (list): list of modules currently filtered for
    Returns:
        list of all found modules
    """
    _, _, all_attributes = load_json_data()
    all_modules = list(set([m.split(" > ")[0].split(",")[0] for m in all_attributes['module']]))
    all_modules.sort()
    modules = []
    found_area = None
    found_program = []
    found_module = []

    # First look for an area - if that is not given, return an empty list
    # Split the input into words and remove punctuation
    split_input = user_input.lower().split()
    split_input = [word.translate(str.maketrans('', '', string.punctuation)) for word in split_input]
    for key, area in module_dict['area'].items():
        split_key = key.split()
        if len (split_key) == 1:  # if key is a single word, search it in split input
            if key in split_input:
                found_area = area
                break # Allow only 1 module per message
        else:  # if key consist of multiple words, search for it in the input sentence(s)
            if key in user_input.lower():
                found_area = area
                break
    if found_area is None:
        return []
    
    # Look for a study program
    for p in module_dict['study_program']:
        if p in user_input.lower():
            found_program = [module_dict['study_program'][p]]
            break

    # Look for the type of module
    # If found area is empty, it is Distinguishing Elective Courses, which is module "W"
    if found_area == "":
        found_module = "W"
    else:
        for m in module_dict['module']:
            if m in user_input.lower():
                # If 'compulsory' is found, check if it is actually 'non compulsory' (i.e., elective)
                if m == 'compulsory' and bool(re.search(rf'\b(non compulsory|non-compulsory)\b', user_input.lower())):
                    found_module = ['WP']
                else:
                    found_module = [module_dict['module'][m]]
                break

    # If only the area was found and it is in the old filter, return only the old filter (in case the user just mentioned, e.g., AI without wanting to set the modules to each module that has '-AI')
    prog_found = True if len(found_program) > 0 else False
    mod_found = True if len(found_module) > 0 else False
    for old_f in old_filter:
        if area in old_f:
            if not prog_found and not mod_found:
                return old_filter
            # If only the study program or module type is not found, set it to the one(s) from the old filter
            old_program = re.findall(r'CS\-(B|M)', old_f)[0]
            if not old_program in found_program:
                found_program += old_program
            old_mod = re.findall(r'CS\-.([A-Z]+)\-', old_f)[0]
            if not old_mod in found_module:
                found_module += old_mod

    # If the program or module is still empty, append each possible value
    if len(found_program) == 0:
        found_program = list(module_dict['study_program'].values())
    if len(found_module) == 0:
        found_module = list(module_dict['module'].values())

    # Combine all found parts and append and return all that match existing modules
    for p in found_program:
        for m in found_module:
            module = f"CS-{p}{m}{found_area} - "
            modules += [mod for mod in all_modules if module in mod]
    return modules


def find_attributes(user_input, old_filter_dict):
    """
    Extracts all attributes from the input

    Parameters:
        user_input (str): the user's input that should be checked for attributes
        old_filter_dict (dict): the previously selected filters
    Returns:
        dictionary with all found attributes (str) and their values (lists)
    """
    _, _, all_attributes = load_json_data()
    relevant_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'home_institute', 'language', 'filter_time']  # filter_time has to be behind ects & sws as otherwise values for those could be misinterpreted as times 

    # Get a dictionary containing all possible values for each attribute that is relevant for the similarity calculation and filter
    check_attr = {a: v for a, v in all_attributes.items() if a in relevant_attributes}
    found_attr = {key: [] for key in check_attr}

    # Check for each relevant attribute if it is contained in the input
    for attr, val in check_attr.items():
        old_filter = old_filter_dict[attr] if attr in old_filter_dict else []

        # First check for individually processed attributes (module, sws, ects, filter_time, mode, status, home_institute)
        if attr == 'module':
            # Only set module from input if the word 'module' is in the input (to avoid setting, e.g., module AI for input like "I'm interested in AI", as also, e.g., Philosophy courses about the Ethics of AI could be interesting)
            if 'module' in user_input:
                found_attr[attr] = find_modules(user_input, old_filter)
            else:
                found_attr[attr] = old_filter
            continue

        # int for SWS or ECTS points have to be directly followed by 'SWS' or 'ECTS' (except for ranges -> only second int has to be followed by it)
        elif attr in ['sws', 'ects']:
            # Only check for sws/ects if at least one of the words is in the input
            if bool(re.search(r'(sws|ects)', user_input.lower())):
                old_sws_ects_filter = {sws_ects: old_filter_dict[sws_ects] if sws_ects in old_filter_dict else [] for sws_ects in ['ects', 'sws']}
                found_attr['sws'], found_attr['ects'], user_input = find_sws_ects(user_input, old_sws_ects_filter)
            else:
                if not found_attr['sws']:
                    found_attr['sws'] = old_filter_dict['sws']
                if not found_attr['ects']:
                    found_attr['ects'] = old_filter_dict['ects']
            continue

        # Check if a time is mentioned
        elif attr == 'filter_time':
            found_attr[attr] = input_times(user_input, old_filter)
            continue

        # Check if one or multiple modes are given in the input (assuming that each mentioned mode is matching the format defined in the courses)
        elif attr == 'mode':
            found_modes = []
            matches = re.findall(r'\b((?:in person|hybrid|online)(\s(\+|with)\srecording)?)', user_input.lower(), re.IGNORECASE)  # No '\b' at the end of the regex pattern as there might also be a 's' after 'recording' and it is highly unlikely that 'recording' is just the start of a bigger word (in that case it is most likely that the user forgot a space anyway)
            found_matches = [m[0].replace("with", "+") for m in matches]
            found_modes = [m for m in found_matches if m in all_attributes['mode']]

            # If modes were found, append ' + recording' to each (when specifying a mode without mentioning 'recording', courses with the mode with recordings should also be fine)
            if found_matches:
                for m in found_matches:
                    extended_mode = m + ' + recording'
                    # Only add the modes with recording to the found modes that exist in the courses' modes
                    if extended_mode in all_attributes['mode']:
                        found_modes.append(extended_mode)
            # If no modes were found but 'recording', add each possible mode that has recordings
            elif bool(re.search(r'\b(recording)', user_input.lower(), re.IGNORECASE)): 
                for rec_mode in all_attributes['mode']:
                    if bool(re.search(r'\b(recording)', rec_mode)):
                        found_modes.append(rec_mode)
            found_modes += old_filter
            found_attr['mode'] = list(set(found_modes))
            continue

        # Check if a status (or multiple) is given in the input (assuming that each mentioned status is matching the format defined in the courses)
        elif attr == 'status':
            found_status = []
            for status in all_attributes['status']:
                if status.lower() in user_input.lower():
                    found_status.append(status)
                    # For lecture or seminar, also append 'Lecture and Practice' or 'Seminar and Practice' as there is not a huge difference; if a user does not want one with practice, they can delete the filter later
                    if status.lower() in ['lecture', 'seminar'] and not (status in old_filter):  # Don't append it if only 'Lecture' or 'Seminar' is in the previous filters, as that means that the user deleted the filter for 'Lecture and Practice' or 'Seminar and Parctice'
                        extended_status = status + ' and Practice'
                        if extended_status in all_attributes['status']:
                            found_status.append(extended_status)
            found_status += old_filter
            found_attr['status'] = list(set(found_status))
            continue

        # Detection of the home institute is not implemented, therefore set it to the previously selected filter (if any was selected from the sidebar)
        elif attr == 'home_institute':  
            found_attr['home_institute'] = old_filter
            continue

        # Check for each other attribute if a new value is found
        found_attr[attr] += old_filter
        for v in val:
            # Add the value if it is found in the input and not yet in the list of found attributes or in the previous filter
            if str(v).lower() in user_input.lower() and str(v) != '' and not (v in found_attr[attr]) and not (v in old_filter):
                if isinstance(v, str):
                    found_attr[attr].append(v.title())
                else:
                    found_attr[attr].append(v)
    return found_attr


def check_filter(filter_dict, course):
    """
    Check if a course matches all given filters

    Parameter:
        filter_dict (dict): All selected filters
        course (dict): The course to check
    """
    missing_filters = 0  # Counts how many filtered attributes are missing (no values) for a course
    active_filters = 0  # Counts how many filters (attributes) are selected
    for filter_key, filter in filter_dict.items():
        # If the filter has no value, continue with next
        if not filter:
            continue
        active_filters += 1

        # Check if the course has a value for the checked filter
        if (not course[filter_key]) or (re.search(r'(not specified)', str(course[filter_key]))):
            missing_filters += 1
            continue
                
        # Check if every time of the course is in the filtered times
        if filter_key == 'filter_time':
            if all(val == [] for val in filter.values()):
                continue
            for c_day, c_times in course['filter_time'].items():
                # Check if the day of the course is in the filtered times
                if (not (c_day in filter)) or (len(filter[c_day]) == 0):
                    return False, missing_filters
                    
                # Check for each time of the day if it matches the filter
                for c_time in c_times:
                    # Times in filters are ordered -> checking smallest time first
                    found_time = False
                    for f_time in filter[c_day]:
                        # If the time of the course is within a timeframe of the filtered times for that day, there is no need to check for more filtered times in that day
                        if (c_time[0] >= f_time[0]) and (c_time[1] <= f_time[1]):
                            found_time = True
                            break
                    # If all filtered times for the day were checked but no fitting time was found, return False
                    if not found_time:
                        return False, missing_filters
                    
        # Filters for lecturer and module are interpreted as wanting at least one of those in the list (if there are more than one) -> one match is enough
        elif filter_key in ['lecturer_short', 'module']:
            found_filter = False
            for c_val in course[filter_key]:
                if (c_val in filter):
                    found_filter = True
                    break
            if not found_filter:
                return False, missing_filters
            
        # For some courses, the language is "German/English" -> both German and English must be in the filter
        elif filter_key == 'language':
            split_lang = course[filter_key].split('/')
            for lang in split_lang:
                if not (lang in filter):
                    return False, missing_filters

        # All other filter attributes are stored in lists -> Check if each value from the course matches the filter
        else:
            if isinstance(course[filter_key], list):
                for c_val in course[filter_key]:
                    if not (c_val in filter):
                        return False, missing_filters

            else:
                # Try to convert both to int -> if that's not possible, compare them as strings
                try:
                    course[filter_key] = int(course[filter_key])
                    filter = [int(f) for f in filter]
                    # If both were converted to ints (SWS or ECTS), check if the courses value is in the filter range
                    if len(filter) >= 2:
                        filter = range(min(filter), max(filter) + 1)
                    if not (course[filter_key] in filter):
                        return False, missing_filters
                except:
                    if not (course[filter_key].lower() in [f.lower() for f in filter]):
                        return False, missing_filters         

    # If more than 5 filters are selected and 50% of filtered attributes are missing, return False (only if more than 5 are selected, as otherwise courses with 1-2 missing filters could not be recommended)
    if (active_filters > 5) and ((missing_filters / active_filters) > 0.5):
        return False, missing_filters
    
    # Otherwise, return True as each filter matched the course (otherwise, it would have returned at some point in the for-loop)
    else:
        return True, missing_filters


def filter_courses(filter_dict, courses):
    """
    Checks for each course in the given list if it matches the given filters
    
    Parameter:
        filter_dict (dict): Dictionary containing all selected filters
        courses (list): All courses to check
    Returns:
        List of all courses that match all filters
    """
    # If no filters are set, return all current courses
    if len(filter_dict) == 0:
        return courses
    matching_courses = []  # Keys: courses; values: percentage (0-1) of how many filter attributes are missing in the course

    # Loop through every course and every filter
    for course in courses:
        is_matching, missing_filters = check_filter(filter_dict, course)
        if is_matching:
            matching_courses.append([course, missing_filters])
    return matching_courses


###--- Recommendation Generation ---###

def input_embedding(user_input, filter_dict):
    """
    Encodes a given string

        Parameters:
            user_input (str): the user's input
            filter_dict (dict): currently set filters
        Returns:
            input_emb: the embedding of the input (dictionary with the embeddings of each attribute that is used for similarity calculations)
            updated filter_dict
    """
    model = load_model()
    # Update the selected filters based on the input
    updated_filter_dict = find_attributes(user_input, filter_dict)

    # Get the embedding of each attribute + the whole input as description
    input_emb = {}
    input_emb['title_descr'] = model.encode(str(user_input))
    for attr in ['module', 'home_institute', 'status', 'mode', 'ects', 'sws', 'lecturer_short', 'area', 'filter_time']:
        input_emb[attr] = model.encode(str(attr))
        
    return input_emb, updated_filter_dict


def recommend_courses(user_profile, filter_dict, amount=5):
    """
    Generates recommendations based on a user's preferences.

        Parameters: 
            user_profile (dict): dictionary containing the embeddings of the user preferences, the already rated courses, the previously liked courses, and the last recommendations
            filter_dict (dict): all selected filters
            amount (int): how many courses should be recommended - default: top 5        
        Returns: 
            response: chatbot message before showing the recommendations
            response_end: chatbot message after showing the recommendations
            to_recommend: indices of the recommended courses
            more_info_counter: number of times the chatbot asked for more information
    """
    current_embeddings, _, _ = load_npz_data()
    current_courses, past_courses, _ = load_json_data()
    user_pref = user_profile['preferences']
    rated_courses = user_profile['rated_courses']
    previously_liked_courses = user_profile['previously_liked_courses']
    last_recommendations = user_profile['last_recommendations']
    more_info_counter = user_profile['more_info_counter']

    # Compute cosine similarity between the user profile and course embeddings
    similarities = []
    for course_emb in current_embeddings.values():
        similarities.append(weighted_cosine_similarity(user_pref, course_emb))
    similarities = np.array(similarities)

    # Rank courses by similarity
    top_courses_indices = similarities.argsort()[::-1]
  
    # Delete already rated courses from top recommendations and select the specified amount of recommendations
    top_indices = [int(idx) for idx in top_courses_indices if idx not in rated_courses]

    # Get the titles of previously liked courses
    liked_titles = [past_courses[idx]['title'] for idx in previously_liked_courses]

    # Delete all titles that are already in the previously liked courses
    cleaned_indices = [idx for idx in top_indices if current_courses[idx]['title'] not in liked_titles]

    response = ""
    response_end = ""
    to_recommend = []

    # If there are no more courses left that could be recommended, tell the user
    if len(cleaned_indices) == 0:
        response = f"There are no more courses left that I could recommend to you!"
        
    # If there are less then the specified amount of courses left to recommend, tell the user that these are the last courses they have not yet rated or mentioned to have previously liked
    elif len(cleaned_indices) < amount:
        response = f"There are only {len(cleaned_indices)} courses left that I could recommend to you. These are:  \n"
        response_end = f"You already rated all the other courses or have taken them in previous semesters! I hope I was able to help you finding new courses and that I will see you again in a few months. Have a great semester!"
        to_recommend = cleaned_indices

    else:          
        # Delete courses that do not match the current filters
        cleaned_courses = [current_courses[idx] for idx in cleaned_indices]
        filtered_courses = filter_courses(filter_dict, cleaned_courses)
        filtered_indices = [current_courses.index(c[0]) for c in filtered_courses]
        threshold = 0.6

        # If there are no courses left that match the current filters, ask the user to remove some filters
        if len(filtered_indices) == 0:
            response = f"There are no courses that match the currently set filters that you haven't rated or mentioned before! Please remove some filters by clicking on them in the list on the left side of the screen."
            more_info_counter = 0
            
        # If there are less then the specified amount of courses left to recommend, tell the user that these are the last courses they have not yet rated or mentioned to have previously liked
        elif len(filtered_indices) <= amount:
            response = f"There are only {len(filtered_indices)} courses that match the currently set filters that you haven't rated or mentioned before! These are:  \n"
            response_end = f"Please remove some filters by clicking on them in the list on the left side of the screen."
            to_recommend = filtered_indices
            more_info_counter = 0

        else:
            # Decrease threshold if filters are set
            if len(filtered_indices) < len(cleaned_indices):
                threshold = 0.5

            # Check if the similarity of any of the courses is above the threshold and select the corresponding response to return together with the list of courses to recommend
            print(f"--- Best matches: {[(current_courses[c]['title'], similarities[c]) for c in filtered_indices[:amount]]}")
            for course in filtered_indices[:amount]:
                if float(similarities[course]) >= threshold:
                    to_recommend.append(course)
            if len(to_recommend) > 1:
                response = "I found some courses you might like:  \n"
                response_end = f"To get more information about a course, you can click on its title in the list.  \n\nPlease tell me if these courses sound interesting to you.  \nIf you haven’t done that already, please check out the ‘Feedback Hints’ (click on the button below the chat) to find out how to properly give feedback. "
                more_info_counter = 0
            elif len(to_recommend) == 1:
                response = f"I found a course you might like:"
                response_end = f"Feedback would help a lot to improve further recommendations. Please tell me if this course sounds interesting or not. "
                more_info_counter = 0
            else:
                # If no similarity is above the threshold and the chatbot already asked more than 3 times for more information, show the currently best courses and ask for feedback
                if more_info_counter >= 3:
                    response = "I still do not have enough information to find courses that really fit what you're looking for. Maybe getting feedback on potential recommendations would help. What do you think about the following courses?"
                    to_recommend = filtered_indices[:amount]
                    response_end = "I would really appreciate feedback for at least one of these courses. Please check out the 'Feedback Hint' below the chat if you haven't already."
                    more_info_counter = 0
                # Otherwise, ask for more information
                else:
                    response = "I need some more information to generate good recommendations for you. Could you tell me more about what kind of course you are looking for? Or is there any course you liked in the past that you didn't tell me about yet? "
                    more_info_counter += 1

            # Check if the list of recommended courses is the same as the last recommendations
            if len(last_recommendations) > 0 and last_recommendations == to_recommend:
                response = "I couldn't find any new recommendations! Please give me some more information or give feedback for the last recommendations so that I can refine my search."
                response_end = ""
        
    return response, response_end, to_recommend, more_info_counter