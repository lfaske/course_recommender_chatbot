import streamlit as st
from recommender import detect_intent, update_user_preferences, recommend_courses, input_embedding, get_past_title, get_details, get_all_filter
import re
from chatbot_variables import confirmation_dict, confirmation_replies, abbreviations, hints, help_text
import time

st.title("Course Recommender Bot")


###--- Initialize session_state variables ---###

# Initialize the chat history with a welcome message from the chatbot
if "messages" not in st.session_state:
    welcome_msg = "Hey, nice to meet you!  \nI'm CouRe, a chatbot designed to help you find interesting courses for the upcoming semester. You can start by either naming a course you liked in the past or describe what kind of course you are looking for. For more detailed instructions, click on the *Instructions* button below."
    #with st.chat_message("assistant"):
    #    first_msg = st.write_stream(type_response(welcome_msg))
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg, "content_end": "", "new_recommendations": []}]

# Initialize the current state
if "current_state" not in st.session_state:
    st.session_state.current_state = 'default'

# Initialize the user profile
if "user_profile" not in st.session_state:
    all_filter_attr = get_all_filter()
    st.session_state.user_profile = {
        'preferences': None,
        'previously_liked_courses': [],
        'rated_courses': [],
        'last_recommendations': [],
        'more_info_counter': 0
    }

# Initialize the dictionary for filters
if "filter_dict" not in st.session_state:
    all_filters = get_all_filter()
    st.session_state.filter_dict = {key: {val: [] for val in all_filters[key]} if key == 'filter_time' else [] for key in all_filters}

# Initialize the dictionary for enabling and disabling the weekdays
if "allow_days" not in st.session_state:
    all_days = list(st.session_state.filter_dict['filter_time'].keys())
    st.session_state.allow_days = {}
    for day in all_days:
        st.session_state.allow_days[day] = False

# Initialize the instruction/hint buttons
if 'hint_button' not in st.session_state:
    st.session_state.hint_button = {
        'show_instructions': False,
        'show_hint': False,
        'show_free': False
    }

# Initialize the course details
if "show_details" not in st.session_state:
    st.session_state.show_details = {}


###--- Handle buttons and messages ---###

def toggle_hint(hint_key):
    """
    Toggles the instruction/hint buttons. First closes other button's field if opened

    Parameter:
        hint_key (str): The key of the pressed button
    """
    # Close the other hints before opening the clicked one
    for key in ['show_instructions', 'show_hint', 'show_free']:
        if key != hint_key:
            st.session_state.hint_button[key] = False
    # Toggle the selected hint
    st.session_state.hint_button[hint_key] = not st.session_state.hint_button[hint_key]

def type_response(msg):
    """
    Displays a message word by word instead of instantly displaying the whole content, making it look similar to typing

    Parameter:
        msg n(str): message to display
    """
    for word in msg.split(" "):
        yield word + " "
        time.sleep(0.025)

def chatbot_response(new_recommendations, chatbot_reply, chatbot_reply_end):
    """
    Creates the chatbot's response

    Parameter:
        new_recommendations (list): List of recommended courses
        chatbot_reply (str): The string to start the chatbot's response with
        chatbot_reply_end (str): The string to end the chatbot's response with
    """
    # List with each recommended title containing all details that should be displayed
    course_details = [{"title": course["title"], "details": course} for course in [get_details(c) for c in new_recommendations]]

    #with st.chat_message("assistant"):
    #    response = st.write_stream(type_response(chatbot_reply))
    #    response_end = st.write_stream(type_response(chatbot_reply_end))

    # Create the chatbot's reply, append it to the chat history and write it into the chat
    st.session_state.messages.append({"role": "assistant", "content": chatbot_reply, "content_end": chatbot_reply_end, "courses": course_details})
    #st.session_state.messages.append({"role": "assistant", "content": response, "content_end": response_end, "courses": course_details})
    msg_idx = len(st.session_state.messages[-1])-1
    with st.chat_message("assistant"):
        render_message(st.session_state.messages[-1], msg_idx)


def render_message(msg, msg_idx):
    """
    Renders a chat message

    Parameter:
        msg (str): Message to render
    """
    if msg["role"] == "assistant":
        # Display the beginning of the chatbot's message
        if msg_idx == len(st.session_state.messages[-1])-1:  # Only the last message should be 'typed'
            st.write_stream(type_response(msg["content"]))
        else:
            st.markdown(msg["content"])
        
        
        # Check if there are expander widgets (containing details of recommended courses) to render
        if "courses" in msg:
            # Display each recommended course as an expander (labeled with the number (position in the list) and the course title)
            for course_nr, course_details in enumerate(msg["courses"], 1):
                title = course_details["title"]
                details = course_details["details"]
                
                # Display details if expanded
                with st.expander(label=f"{course_nr}: {title}"):
                    st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                            <strong>Course type:</strong> {details['status']}<br>
                            <strong>Mode:</strong> {details['mode']}<br>
                            <strong>Language:</strong> {details['language']}<br>
                            <strong>ECTS:</strong> {details['ects']}<br>
                            <div style="margin-bottom: 10px;"><strong>SWS:</strong> {details['sws']}</div>
                            <div style="margin-bottom: 10px;"><strong>Prerequisites:</strong> <br>{details['prerequisites']}</div>
                            <div style="margin-bottom: 10px;"><strong>Lecturer:</strong> <br>{'<br>'.join(details['lecturer'])}</div>
                            <div style="margin-bottom: 10px;"><strong>Times:</strong> <br>{details['time']}</div>
                            <div style="margin-bottom: 10px;"><strong>Home Institute:</strong> <br>{details['home_institute']}</div>
                            <div style="margin-bottom: 10px;"><strong>Module Assignment:</strong> <br>{'<br>'.join(details['module'])}</div>
                            <div><strong>Description:</strong> <br>{details['description']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                time.sleep(0.075)

        # Display the end of the chatbot's message
        if msg["content_end"] != "":
            #st.markdown(msg["content_end"])
            if msg_idx == len(st.session_state.messages[-1])-1:
                st.write_stream(type_response(msg["content_end"]))
            else:
                st.markdown(msg["content_end"])
    else:
        # Display the user's message
        st.markdown(msg["content"])


# Display chat messages from history on app rerun
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg['role']):
        render_message(msg, i)


###--- Handle User Input ---###

if user_input := st.chat_input("Start typing ..."):
    print("\n\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n-+-+-+-+- NEW USER MESSAGE -+-+-+-+-\n")

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add the message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Expand abbreviations in input
    for abbrev, full_form in abbreviations.items():
        user_input = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, user_input, flags=re.IGNORECASE)

    chatbot_reply = ""
    chatbot_reply_end = ""  # E.g. content after presenting recommended courses
    new_recommendations = []
            
    # If the chatbot asked the user whether or not the detected course reference was correct, it waits for confirmation
    if st.session_state.current_state == 'confirmation':

        # Look for words marking confirmation or negation
        confirmation_keys = []
        for key, value_list in confirmation_dict.items():
            if any(value in user_input.lower() for value in value_list):
                confirmation_keys.append(key)

        # If a confirmation is found, set the current state back to default, update the user profile and generate recommendations
        if len(confirmation_keys) == 1 and confirmation_keys[0] == 'yes':
            st.session_state.current_state = 'default'
            chatbot_reply = confirmation_replies['yes']
            liked_course = st.session_state.user_profile['previously_liked_courses'][-1]

            # Update the user profile and generate recommendations
            st.session_state.user_profile['preferences'] = update_user_preferences(user_profile=st.session_state.user_profile['preferences'], rated_course=(liked_course, 'past'), liked=True)
            reply, reply_end, new_recommendations, st.session_state.user_profile['more_info_counter'] = recommend_courses(user_profile=st.session_state.user_profile, filter_dict=st.session_state.filter_dict)
            chatbot_reply += reply
            chatbot_reply_end += reply_end
            # Only save the returned list as last recommendations if it is not empty
            if len(new_recommendations) > 0:
                st.session_state.user_profile['last_recommendations'] = new_recommendations
            
        # If a negation is found, set the current state back to default and remove the last entry from the previously liked courses
        elif len(confirmation_keys) == 1 and confirmation_keys[0] == 'no':
            st.session_state.current_state = 'default'
            chatbot_reply = confirmation_replies['no']
            del st.session_state.user_profile['previously_liked_courses'][-1]

        # If neither is found, ask the user to first confirm or deny, before giving new information 
        else:
            chatbot_reply = confirmation_replies['other']
    
    # If the current state is default
    else: 
        # Detect the intent of the message
        detected_intent, correct_reply, detected_courses = detect_intent(user_input, st.session_state.user_profile['last_recommendations'])
        chatbot_reply += correct_reply

        # If the user referred to a liked course, add the course to the previously liked courses 
        if detected_intent == "liked_course_reference":
            detected_courses = detected_courses[0]
            st.session_state.user_profile['previously_liked_courses'].append(detected_courses[0])

            # If the certainty (similarity of the title to the user input) is not high enough: ask if correct
            if detected_courses[1] < 0.7:
                chatbot_reply = f"I'm not sure if I understood you correctly. You liked the course {get_past_title(detected_courses[0])}, is that correct? "
                st.session_state.current_state = "confirmation"

            # Otherwise, update the user profile and generate recommendations
            else:
                chatbot_reply = f"You liked the course {get_past_title(detected_courses[0])}.  \n"
                st.session_state.user_profile['preferences'] = update_user_preferences(user_profile=st.session_state.user_profile['preferences'], rated_course=(detected_courses[0], 'past'), liked=True)
                reply, reply_end, new_recommendations, st.session_state.user_profile['more_info_counter'] = recommend_courses(user_profile=st.session_state.user_profile, filter_dict=st.session_state.filter_dict)
                chatbot_reply += reply
                chatbot_reply_end += reply_end
                # Only save the returned list as last recommendations if it is not empty
                if len(new_recommendations) > 0:
                    st.session_state.user_profile['last_recommendations'] = new_recommendations

        # If the user gave a free description, update their preferences and generate new recommendations 
        elif detected_intent == "free_description":
            input_emb, st.session_state.filter_dict = input_embedding(user_input, st.session_state.filter_dict)  # Compute the embedding of the user's input and get the mentioned attributes for the filter
            
            # Check if the user set a time for a new day in the input
            for day, d_time in st.session_state.filter_dict['filter_time'].items():
                # If the user set a time for a day that is currently deactivated in the filter, activate it
                if d_time != [] and day in st.session_state.allow_days and not st.session_state.allow_days[day]:
                    st.session_state.allow_days[day] = True

            # Update the user profile and generate recommendations
            st.session_state.user_profile['preferences'] = update_user_preferences(user_profile=st.session_state.user_profile['preferences'], input_embedding=input_emb, liked=True)
            reply, reply_end, new_recommendations, st.session_state.user_profile['more_info_counter'] = recommend_courses(user_profile=st.session_state.user_profile, filter_dict=st.session_state.filter_dict)
            chatbot_reply += reply
            chatbot_reply_end += reply_end
            # Only save the returned list as last recommendations if it is not empty
            if len(new_recommendations) > 0:
                st.session_state.user_profile['last_recommendations'] = new_recommendations

        # If the user gave feedback, find out for which recommended course and update the user profile accordingly
        elif detected_intent == "feedback":
            if len(detected_courses) > 0:
                # Update the user profile with each rated course
                for (c, sentiment) in detected_courses:
                    st.session_state.user_profile['preferences'] = update_user_preferences(st.session_state.user_profile['preferences'], rated_course = (c, 'current'), liked = (sentiment == 'liked'))
                    st.session_state.user_profile['rated_courses'].append(c)
                
                # Generate new recommendations
                reply, reply_end, new_recommendations, st.session_state.user_profile['more_info_counter'] = recommend_courses(user_profile=st.session_state.user_profile, filter_dict=st.session_state.filter_dict)
                chatbot_reply += reply
                chatbot_reply_end += reply_end
                # Only save the returned list as last recommendations if it is not empty
                if len(new_recommendations) > 0:
                    st.session_state.user_profile['last_recommendations'] = new_recommendations
            
    chatbot_response(new_recommendations, chatbot_reply, chatbot_reply_end)


###--- Add a Sidebar ---###

st.sidebar.header("Filter Options", help = "To select a filter, you can either select them from the boxes below or write them in the chat. To remove a filter, just click on the 'x' next to it.")

# If the user clicks the button "Refresh", try to generate new recommendations
if st.sidebar.button("Refresh", help=help_text['refresh']):
    try:
        reply, reply_end, new_recommendations, st.session_state.user_profile['more_info_counter'] = recommend_courses(user_profile=st.session_state.user_profile, filter_dict=st.session_state.filter_dict)
        chatbot_reply = reply
        chatbot_reply_end = reply_end
        # Only save the returned list as last recommendations if it is not empty
        if len(new_recommendations) > 0:
            st.session_state.user_profile['last_recommendations'] = new_recommendations
        chatbot_response(new_recommendations, reply, reply_end)
    # If the user gave no information yet (neither through chat nor filters), it cannot generate any recommendations
    except:
        chatbot_response([], "I cannot recommend courses without knowing what you are looking for! Please use the chat first to tell me what kind of courses you are interested in or name a course you liked in the past.", "")


###--- Create the buttons for general instructions and hints for feedback at the bottom of the chat ---###

# Custom CSS for the box to display the hints in
st.markdown("""
    <style>
        .hint-box {
            border: 1px solid grey;
            padding: 10px;
            margin-top: 10px;
            border-radius: 8px;
            max-width: 80%
        }
    </style>
""", unsafe_allow_html=True)


# Place the hint buttons side by side
st.divider()
col1, col2, col3, col4 = st.columns([1.25, 2, 2.5, 2])
with col1:
    st.markdown("""
        <div style="margin-top: 7.5px;"><strong>Hints:</strong></div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("Instructions", key="button_1"):
        toggle_hint("show_instructions")
with col3:
    if st.button("Free Descriptions", key="button_2"):
        toggle_hint("show_free")
with col4:
    if st.button("Feedback", key="button_3"):
        toggle_hint("show_hint")
        
# Display the corresponding expanded text
if st.session_state.hint_button['show_instructions']:
    st.markdown(f"""
        <div class="hint-box">
            {hints['instructions']}
        </div>
    """, unsafe_allow_html=True)
elif st.session_state.hint_button['show_hint']:
    st.markdown(f"""
        <div class="hint-box">
            {hints['Feedback Hint']}
        </div>
    """, unsafe_allow_html=True)
elif st.session_state.hint_button['show_free']:
    st.markdown(f"""
        <div class="hint-box">
            {hints['Free Hint']}
        </div>
    """, unsafe_allow_html=True)


###--- Add Filters to Sidebar ---###

all_filter = get_all_filter()

# Updates the filter dictionary in session_state, called when a new filter is selected in the sidebar
def update_filters(filter):
    if isinstance(filter, str):
        st.session_state.filter_dict[filter] = st.session_state[filter]
    else:
        st.session_state.filter_dict[filter[0]][filter[1]] = st.session_state[f"{filter[0]}_{filter[1]}_{filter[2]}"]

# Maps each attribute to how it should be labeled in the sidebar
filter_names = {'status': 'Course Type', 'mode': 'Mode', 'ects': 'ECTS', 'sws': 'SWS', 'lecturer_short': 'Lecturer', 'module': 'Module', 'language': 'Language', 'filter_time': 'Time', 'home_institute': 'Home Institute'}

# Create a widget for each filter
for filter_key, filter_options in all_filter.items():
    f_name = filter_names[filter_key]
    f_help = help_text[filter_key] if filter_key in help_text else ""

    # For filter_time, create a checkbox for each day to enable or disable the day
    if filter_key == 'filter_time':
        st.sidebar.subheader("Days and Times", help=help_text['filter_time'])
        selected_options = {}
        for day, timeframes in filter_options.items():
            allow_day = st.sidebar.checkbox(f"Show courses on {day}?", value=st.session_state.allow_days[day], key=f"allow_{day}")
            
            # Place an expander that shows the timeframes selected for the weekday
            expander = st.sidebar.expander(f"Selected Times for {day}")
            if st.session_state.filter_dict[filter_key][day] != []:
                for day_time in st.session_state.filter_dict[filter_key][day]:
                    if day_time != []:
                        expander.write(f"{day_time[0]}:00 - {day_time[1]}:00")
            else:
                expander.write("")

            # Enable or disable the day according to the checkbox
            if allow_day:
                st.session_state.allow_days[day] = True
            else:
                st.session_state.allow_days[day] = False
                # If disabled, remove the timeframes from the expander
                expander.empty()

            # If the day is enabled, set the time filter to the previously selected times (if any; otherwise to the whole possible timeframe)
            if st.session_state.allow_days[day] == True:
                append_times = []
                if st.session_state.filter_dict[filter_key][day] != []:
                    append_times = st.session_state.filter_dict[filter_key][day]
                else:
                    append_times = [[timeframes[0][0], timeframes[0][1]]]
                if day in selected_options:
                    selected_options[day] += append_times
                else:
                    selected_options[day] = append_times
            else:
                selected_options[day] = []

    # Use a slider with ranges for ects and sws
    elif filter_key in ["ects", "sws"]:
        # If there is a range saved for the attribute, use it as default value
        filter_options = list(range(int(filter_options[0]), int(filter_options[-1]) + 1))
        if len(st.session_state.filter_dict[filter_key]) > 0:
            selected_range = [int(st.session_state.filter_dict[filter_key][0]), int(st.session_state.filter_dict[filter_key][-1])]
            # Check if both the selected minimum and the selected maximum are in the possible range
            if selected_range[0] < filter_options[0]:
                selected_range[0] = filter_options[0]
            elif selected_range[0] > filter_options[-1]:
                selected_range[0] = filter_options[-1]
            if selected_range[-1] < filter_options[0]:
                selected_range[-1] = filter_options[0]
            elif selected_range[-1] > filter_options[-1]:
                selected_range[-1] = filter_options[-1]
        else:
            selected_range = filter_options
        # Create a select_slider widget which can be used to set the ranges for ECTS and SWS
        selected_sws_ects = st.sidebar.select_slider(
            label=f"Select {f_name}", 
            options=filter_options,
            value=[selected_range[0], selected_range[-1]],
            key=filter_key
        )
        selected_options = list(selected_sws_ects)

    else:
        # Use st.multiselect and connect it to a session state variable via the `key`
        selected_options = st.sidebar.multiselect(
            label=f"Select {f_name}",
            options=filter_options,
            default=st.session_state.filter_dict[filter_key],  # Initially empty selection
            key=filter_key,  # Unique key for each widget
            help=f_help,
            on_change=update_filters,  # Trigger the filter update function
            args=(filter_key,),  # Pass the filter as an argument to the function
        )
    st.session_state.filter_dict[filter_key] = selected_options

print(f"\n~~~ NEW FILTER DICT: {st.session_state.filter_dict}")


