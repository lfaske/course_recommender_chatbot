# Define which user responses can mean yes and which can mean no
confirmation_dict = {
    'yes': ['yes', 'exactly', 'correct', 'yep', 'yeah', 'ja', 'right'],
    'no': ['no', 'nope', 'not', 'false', 'nein', 'wrong']
}

# Define how the chatbot replies to confirmation or rejection
confirmation_replies = {
    'yes': 'Great! ',
    'no': "I'm sorry for the misunderstanding. Could you rephrase your message then? ",
    'other': "I'm sorry, I didn't understand that. Please just answer with 'yes' or 'no'. We will continue finding great courses after that. "
}

# Define common abbreviations and their full forms
abbreviations = {
    "AI": "Artificial Intelligence",
    "KI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "NLP": "Natural Language Processing",
    "NLU": "Natural Language Understanding",
    "LLM": "Large Language Model",
    "LLMs": "Large Language Models",
    "CL": "Computational Linguistics",
    "CV": "Computer Vision",
    "Coxi": "Cognitive Science",
    "CogSci": "Cognitive Science",
    "Philo": "Philosophy",
    "Intro": "Introduction",
    "Scipy": "Scientific Programming in Python",
    "Info": "Informatics",
    "Neuroinfo": "Neuroinformatics",
    "Math": "Mathematics",
    "Neurobio": "Neurobiology",
    "HCI": "Human-Computer-Interactions",
    "MCI": "Mensch-Computer-Interaktion",
    "DL": "deep learning",
    "VR": "virtual reality"
}

# Define chatbot replies based on user's intent
intent_replies = {
    "greeting": "Hello! How can I assist you with course recommendations today?",
    "thanks": "You're welcome!",
    "free_description": "Thanks for sharing!  \n",
    "liked_course_reference": "That's a great course! Let me find some similar courses for you...",
    "feedback": "Thank you for your feedback! I'll use it to improve recommendations.",
    "no_info": "Let's find some courses you might like! Please tell me what kind of course you are looking for or which course you liked in the past.",
    "nonsense": "I'm sorry, but your message does not make sense to me. I can only understand messages in the context of recommending university courses. For more instructions click on the 'Instructions' button below the chat. Could you please rephrase your message to be more clear?",
    "other": "I'm sorry, but I do not understand what you want to tell me. Could you please clarify?"
}

# Title and text for the hint-button
hints = {
    'instructions':
        """<div style="padding-left: 10px; max-width: 100%;">
        <p>You can either tell me a course you liked in the past or simply describe what kind of course you are looking for. Please keep in mind the following tips:</p>
            <ul>
                <li>Please write your messages in <u>English</u>, as I do not understand other languages.
                <li><b>Keep it simple!</b> It's better to write multiple simple messages than to put all the information into a single one.</li>
                <li>Please don't put different types of input (such as a reference to a course you liked in the past, a free description of what you would like, or feedback on a recommended course) into a single message. Instead, split it up into multiple messages so that I can better understand you.</li>
                <li>When referencing a course from the past or giving a free description, please only tell me what you <u>like</u>! I cannot detect negative sentiments (such as <i>I don't like online courses</i>) for that kind of input.</li>
                <li>If you want to tell me more than one course you liked in the past, please do so in separate messages (1 course per message).</li>
                <li>Sometimes, I have a hard time understanding the intention of a message. If I don't understand you correctly, you can just start your message by stating your intention: 
                <ul>
                    <li>To give feedback, start with <i>Feedback:</i></li>
                    <li>To tell me a course you liked in the past, start with <i>Ref:</i></li>
                    <li>To give a free description, start with <i>Free:</i></li>
                </ul>
                </li>
            </ul>
        </div>""",
    'Free Hint':
        """<div style="padding-left: 10px; max-width: 80%;">
            <p>With free descriptions, you can tell me what kind of course you are looking for. You can either tell me topics you are interested in, or specify attributes the recommended courses must have.</p>
            <p>If you specify attributes over the chat, you will see the ones I detected in the sidebar on the left. There you can also delete selected attributes or select new ones.</p>
            <p>For example, a free description could look like this:</p>
            <ul>
                <li><i>I'm looking for a course taking place on mondays or thursdays between 8 am and 4 pm for my Bachelor's elective Philosophy module. Something about the mind and consciousness would be interesting.</i></li>
                <li><i>I want an online lecture or seminar about the anatomy of the human brain.</i></li>
                <li><i>I'm interested in NLP.</i></li>
            </ul>
            <p>There are a few limitations you should keep in mind when specifying attributes over the chat:</p>
            <ul>
                <li>If you want to specify <b>multiple modules</b>, please write only <u>one</u> per message. Alternatively, you can select modules (or any other attributes) in the sidebar.</li>
                <li>If you want a course with <b>recordings</b> (no matter if it is online, in person, or hybrid) you can just state that (e.g., <i>I want a course with recordings</i>)</li>
                <li>When defining <b>times</b>, please always state the days (single days, multiple days, range of days, or 'every day') <u>in front</u> of the timeframes (if you write a single time, e.g., <i>10 am</i> instead of a range, a 2 hour range starting with that time is selected). Times without days cannot be detected. If you write multiple days and only a single timeframe behind them, the timeframe is selected for each of the days. If you want to select different timeframes for the days, please always state the day before the corresponding timeframe. If you mention a day (or multiple days) without a timeframe, the whole day is selected. For example:
                <ul>
                    <li><i>Monday to Wednesday between 8 and 2</i> is interpreted as: Monday 8:00-14:00, Tuesday 8:00-14:00, Wednesday 8:00-14:00</li>
                    <li><i>on Monday or Friday</i> is interpreted as: Monday 7:00-21:00, Friday 7:00-21:00</li>
                    <li><i>Tuesday 10-12 or Friday 8-14</i> is interpreted as: Tuesday 10:00-12:00, Friday 8:00-14:00</li>
                    <li><i>Tuesday or Friday 8-14</i> is interpreted as: Tuesday 8:00-14:00, Friday 8:00-14:00</li>
                    <li><i>Tuesday at 8-14 or any other day from 10 to 6 pm</i> is interpreted as: Monday 10:00-18:00, Tuesday 8:00-14:00, Wednesday 10:00-18:00, Thursday 10:00-18:00, Friday 10:00-18:00, Saturday 10:00-18:00 (if there are any courses on Saturday)</li>
                </ul></li>
                <li>When filtering <b>times</b>, only courses that are offered weekly throughout the semester are considered. Courses such as block seminars are not supported by the time filter and can therefore not be recommended when times are selected.</li>
                <li>When defining <b>SWS</b> or <b>ECTS</b>, always write <i>sws</i> or <i>ects</i> behind the number or range (e.g., <i>I want a course with 4-6 ects</i>). You can only select a single range (or number) for each.</li>
            </ul>
            </div>
        """,
    'Feedback Hint': 
        """<div style="padding-left: 10px; max-width: 80%;">
                <ul>
                    <li>When giving feedback, please refer to the course you want to give feedback for by it's <u>position</u> in the list (e.g., <i>the first</i> or <i>course 4</i>).</li>
                    <li>You can either write the positions separately or give ranges (e.g., <i>courses 1, 2, 3 and 4</i> or <i>courses 1 to 4</i>). You can also tell me if you liked <i>all</i> or <i>none</i> of the recommendations (e.g., <i>I like all of them</i>).</li>
                    <li>If you want to give both positive and negative feedback, please make sure that you don't write them in the same sentence. Alternatively, you can divide the sentence using <i>but</i>. For example, you could write: <i>I liked the first and third recommendation, but not the second one.</i></li>
                </ul>
            </div>"""

}

# Text to add to certain widgets to help the user
help_text = {
    'module': "The naming of the modules consists of the following information:  \n- 'CS-': Cognitive Science (currently only courses from Cognitive Science modules are included)  \n- 'B'|'M': 'B' for modules from the Bachelor's program, 'M' for the Master's program  \n- 'P'|'W'|'WP': The type of the module - 'P' for compulsory modules, 'WP' for elective modules, 'W' for Distinguishing Elective Courses and Instruction for Working Scientifically  \n- '-XY': The short form of the area (e.g., 'AI' for Artificial Intelligence)  \n\nExample: The module *CS-BWP-NI* is the elective Neuroinformatics module for the Bachelor's program",
    'filter_time': "Select the days you want to get recommendations for *. You can set time ranges for a day or multiple days by writing them in the chat. Make sure to deactivate the day before writing the range in the chat, as it would otherwise just be added to the range of the whole day, changing nothing.  \n*) Due to a bug, you have to click twice for enabling/disabling a day.  \n* E.g., *I\'m free from Monday to Thursday between 10 am and 4 pm.*",
    'refresh': "Click this button after setting or removing filters to generate new recommendations without having to type anything in the chat."
}

# To find mentioned modules
module_dict = {
    'study_program': {
        'bachelor': 'B',
        'master': 'M'
    },
    'module': {
        'elective': 'WP',
        'compulsory': 'P'
        # 'BW' is only used for 'Instruction for Working Scientifically' ('Anleitung zum wissenschaftlichen Arbeiten') -> not important here
    },
    'area': {
        'ai': '-AI',
        'artificial intelligence': '-AI',
        'ni': '-NI',
        'neuroinformatics': '-NI', 
        'neuroinfo': '-NI',
        'cl': '-CL', 
        'linguistics': '-CL',
        'cnp': '-CNP', 
        'neuropsychology': '-CNP', 
        'psychology': '-CNP',
        'mat': '-MAT',
        'mathematics': '-MAT', 
        'math': '-MAT',
        'phil': '-PHIL', 
        'philosophy': '-PHIL', 
        'philo': '-PHIL',
        'working scientifically': '-IWS', 
        'iws': '-IWS',
        'methods': '-MCS',
        'mcs': '-MCS',
        'ic': '-IC', 
        'interdisciplinary': '-IC',
        'inf': '-INF', 
        'info': '-INF', 
        'informatics': '-INF',
        'computer science': '-INF',
        'study project': '-SP', 
        'sp': '-SP',
        'ns': '-NS', 
        'neuroscience': '-NS',
        'distinguishing elective': ''
    }
}