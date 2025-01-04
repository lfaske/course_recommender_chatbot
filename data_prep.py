from xml.dom.minidom import parse
from sentence_transformers import SentenceTransformer
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import re
import string
import numpy as np
import os
import difflib
import json
from data_prep_variables import abbreviations, intent_examples

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
std_emb_file = "embeddings.npz"
std_course_file = "courses.json"

def read_xml(filename):
    """
    Read the XML file and return a list of dictionaries, each containing the information of a course.
    
     Parameters:
        filename (str): the name to the xml file (saved in 'data' folder)
    Returns:
        list of dictionaries (each dictionary contains all relevant attributes of one course from the semester)
    """
    print(f"-- Reading xml file: {filename}")
    def get_single_value(node, tag, subtag=None):
        try:
            if not subtag:
                return node.getElementsByTagName(tag)[0].firstChild.data
            else:
                subnode = node.getElementsByTagName(tag)[0]
                try:
                    return subnode.getElementsByTagName(subtag)[0].firstChild.data
                except AttributeError:
                    return ''
        except IndexError:
            return ''

    def get_data_field(node, fieldname):
        datafields = node.getElementsByTagName('datenfelder')
        if not datafields:
            return ''
        for d in datafields[0].getElementsByTagName('datenfeld'):
            if d.getAttribute('key') == fieldname:
                return d.firstChild.data

    dom = parse(filename)  # parse an XML file by name
    domcourses = dom.getElementsByTagName('seminar')
    courses = []
    for node in domcourses:
        course = {}
        s_key = node.getAttribute('key')
        # Only add courses with new seminar keys to avoid duplicates (as with the 'Language and Communication Colloquium' in SoSe 2024)
        if s_key in [c['sem_key'] for c in courses]:
            continue
        course['sem_key'] = s_key
        
        # Dictionary with the German terms as keys & the English translations as values
        translated_attributes = {
            'voraussetzung': 'prerequisites',
            'heimateinrichtung': 'home_institute'
        }

        for k in ['ects', 'voraussetzung', 'heimateinrichtung']:
            if k in translated_attributes.keys():
                trans_k = translated_attributes[k]
            else:
                trans_k = k
            # Make sure ECTS are saved as strings
            value = get_single_value(node, k)
            if isinstance(value, int):
                course[trans_k] = str(value)
            else:
                course[trans_k] = value

        # Get the cleaned title
        title = get_single_value(node, 'titel')
        # Remove unnecessary information at the end of the titles (e.g., "(CS-"..., "(Lecture)", ...)
        if ' (Lecture' in title:
            title = title[:title.find(' (Lecture')]
        if ' (CS-' in title:
            title = title[:title.find(' (CS-')]
        # Remove extra spaces if there are any
        cleaned_title = " ".join(title.split())
        course['title'] = cleaned_title


        # Check if there are prerequisites mentioned in the description
        # There are 3 main formats of prerequisites in descriptions:
        # - 1. At the start in one line, separated by rest of description with "\n" ("Prerequisites: ...\n")
        # - 2. At the start in multiple lines, separated by rest of description with "\n\n" ("Prerequisites:\n...\n\n")
        # - 3. Somewhere inside the description in multiple lines, separated by rest of description with "\n\n" before and after the prerequisites ("\n\nPrerequisites:\n...\n\n")
        description = course['description'] = get_single_value(node, 'beschreibung').strip()
        prerequisites = course['prerequisites']
        found_start = ""
        found_prerequisites = ""
        for found_start in ["\n\nPrerequisites:", "\nPrerequisites:", "Prerequisites:"]:
            if found_start in description:
                #print(f"--- Found prerequisites in description! (for course '{cleaned_title}')")
                start_index = description.find(found_start)

                # If the prerequisites start with a newline, include it in the found start
                if description[start_index + len(found_start):].startswith("\n"):
                    found_start += "\n"

                # Find the end of the prerequisites: First check for a double newline; if that is not found, look for a single one
                end_index = description.find("\n\n", start_index + len(found_start))
                if end_index == -1:
                    end_index = description.find("\n", start_index + len(found_start))              

                found_prerequisites = description[start_index + len(found_start):end_index].strip()
                course['description'] = description.replace(description[start_index:end_index], '').strip()

                # Add the prerequisites to the corresponding attribute if they are not in there already
                if not (found_prerequisites in prerequisites) and not (found_prerequisites.lower() in ["none", "none!"]):
                    if prerequisites == "":
                        course['prerequisites'] = found_prerequisites
                    elif prerequisites[-1] == ".":
                        course['prerequisites'] = " ".join([prerequisites, found_prerequisites])
                    else:
                        course['prerequisites'] = ". ".join([prerequisites, found_prerequisites])
                break
            
        # Translate the status (if german)
        status_translations = {
            'Vorlesung und Übung': 'Lecture and Practice',
            'Seminar und Praktikum': 'Seminar and Practice',
            'Studienprojekt': 'Study Project'
        }
        status = get_single_value(node, 'status')
        if status in status_translations.keys():
            status = status_translations[status]
        course['status'] = status

        # Add 'teilnehmer' to prerequisites ('teilnehmer' includes information such as 'ab 4. Semester')
        participants = get_single_value(node, 'teilnehmer')
        if participants and not participants.isnumeric():  # In some cases, this field only contains a number (e.g., '24') -> useless as prerequisite
            if course['prerequisites']:  # Add a space, if there are already prerequisites for this course
                course['prerequisites'] += ' ' 
            course['prerequisites'] += participants
            
        # Add subtitles to description
        subtitle = get_single_value(node, 'untertitel')
        if course['description']:  # Add a space, if there is already a description for this course
            course['description'] += ' ' 
        course['description'] += subtitle
          
        # Get SWS
        course['sws'] = get_data_field(node, 'SWS')
        if course['sws'] and "\n" in course['sws']:
            course['sws'] = course['sws'].split('\n')[1]

        # Get and translate the shortened mode
        mode = get_data_field(node, 'Art der Durchführung')
        if mode:
            mode = mode.split('[')[1].split(']')[0]
            if 'präsenz' in mode:
                mode = mode.replace('präsenz', 'in person')
            if '+' in mode:
                mode = mode.replace('+', ' + recording')
            course['mode'] = mode
        else:
            course['mode'] = 'not specified'

        # Get and translate language
        language = get_data_field(node, 'Sprache')
        if isinstance(language, str):
            language = language.replace("\n", "")
            if 'deutsch' in language.lower() and 'englisch' in language.lower():
                course['language'] = 'German/English'
            elif 'deutsch' in language.lower():
                course['language'] = 'German'
            elif 'englisch' in language.lower():
                course['language'] = 'English'
            else:
                course['language'] = language.capitalize().strip()
        else:
            course['language'] = 'not specified'

        # Add 'Hinweise zur Veranstaltung' to description
        hinweise = get_data_field(node, 'Hinweise zur Veranstaltung')
        if hinweise:
            course['description'] += " "
            course['description'] += hinweise

        # Get the times (schedule)
        termine = get_single_value(node, 'termine', 'termin').replace('..', '.')
        if not termine:
            termine = 'No dates fixed yet'
        course['time'] = termine

        # Get a dictionary with the times for filtering
        days = {'Monday': 'Mon.', 'Tuesday': 'Tue.', 'Wednesday': 'Wed.', 'Thursday': 'Thu.', 'Friday': 'Fri.', 'Saturday': 'Sat.'}
        course['filter_time'] = {}
        c_time = termine
        if any(char.isdigit() for char in c_time):

            # Split 'time' attribute in lines and remove times for tutorials etc. (as those are usually not mandatory)
            split_input = c_time.split("\n")
            delete_words = ["Tutorial", "Practice Sessions", "Tutorium", "Übung"]
            selected_lines = [line for line in split_input if not any(word in line for word in delete_words)]
            selected_times = "\n".join(selected_lines)

            # Remove punctuation (except . and :) and split it into words
            translation_table = str.maketrans('', '', string.punctuation.replace(':', '').replace('.', ''))
            cleaned_time = selected_times.translate(translation_table)
            split_time = cleaned_time.split()
            found_times = {}
            for idx, part in enumerate(split_time):
                # If a day is found and the next two parts are times, add it to the found times
                if part in [list(days.keys()) + list(days.values())][0]:
                    # Make sure that each day is written as abbreviation
                    if part in days:
                        part = days[part]
                    if bool(re.match(r"^\d{2}:\d{2}$", split_time[idx+1])) and bool(re.match(r"^\d{2}:\d{2}$", split_time[idx+2])):
                        # Transform time into int
                        start_time = int(split_time[idx+1].split(":")[0])
                        end_time = int(split_time[idx+2].split(":")[0])

                        # If end time is not a full hour, round it up
                        if not bool(re.match(r"^\d{2}:00$", split_time[idx+2])):
                            end_time += 1
                        
                        # If there have already been times detected for the day in the course, check for overlap
                        if part in found_times:
                            overlap = []
                            for times in found_times[part]:
                                # Start or end in existing timeframe (or equal) -> select smallest start + biggest end
                                if (start_time >= times[0] and start_time <= times[1]) or (end_time >= times[0] and end_time <= times[1]):
                                    overlap = [min(start_time, times[0]), max(end_time, times[1])]
                                    found_times[part].remove(times)
                                    found_times[part].append(overlap)

                            # If no overlap was detected, save new timeframe
                            if len(overlap) == 0:
                                found_times[part].append([start_time, end_time])
                        else:
                            found_times[part] = [[start_time, end_time]]

            # Save the found times
            course['filter_time'] = found_times

        # Get lecturers
        lecturer_node = node.getElementsByTagName('dozenten')
        all_lecturers=[]
        all_short=[]
        if not lecturer_node:
            course['lecturer'] = ['N.N.']
        else:
            for d in lecturer_node[0].getElementsByTagName('dozent'):
                nd = d.firstChild.data
                # remove trailing ", M.Sc." and ", Ph.D." from names
                if nd.endswith(", M. Sc."):
                    nd = nd[:-8]
                elif nd.endswith(", Ph.D."):
                    nd = nd[:-7]
                all_lecturers.append(nd)
                all_short.append(nd.split(' ')[-1])
            course['lecturer'] = all_lecturers[:]
            course['lecturer_short'] = all_short

        # Module
        lvgruppen_node = node.getElementsByTagName('lvgruppen')
        allmodules=[]
        for m in lvgruppen_node[0].getElementsByTagName('lvgruppe'):
            m = m.firstChild.data
            if m.startswith('Cognitive Science'):  # only cognitive science modules
                # Check if ECTS are included in this field
                if m.endswith('ECTS'):
                    # Save number of ETCS in corresponding column, as it's missing for some courses
                    ects = m[m.rfind(' - '):]
                    ects = int(re.search(r'\d+', ects).group())
                    if course['ects'] == '':
                        course['ects'] = str(ects)
                # Only include the short form and area (e.g., "Neuroinformatics") in the module field
                m = re.findall(r"(CS-[^,>]+)", m)[0]
                allmodules.append(m)
        course['module'] = allmodules[:]       
        
        # Get areas
        area_node = node.getElementsByTagName('bereiche')
        all_areas=[]
        if not area_node:
            course['area'] = ['not specified']
        else:
            for b in area_node[0].getElementsByTagName('bereich'):
                b = b.firstChild.data
                all_areas.append(b)
            course['area'] = all_areas[:]
            
        # Only include courses with cognitive science modules (for courses without cognitive science modules, course['module'] is empty)
        if course['module']:
            # Check if the title is already in the list of courses from the currently checked semester 
            if course['title'] in [c['title'] for c in courses]:
                same_title = [c for c in courses if c['title'] == course['title']][0]
                # If it has the same title as another course but the status is different (e.g., one is a lecture and the other a seminar), add the status to the title
                if course['status'] != same_title['status']:
                    course['title'] += f" ({course['status']})"
                    same_title['title'] += f" ({same_title['status']})"
            courses.append(course)
    print("--> Done reading!")
    return courses


def translate_courses(courses):
    """
    Translates the titles and descriptions of all German entries from a list of courses to English.
    (Other attributes are directly translated when reading the xml file)

    Parameters:
        courses (list): List of courses (as dictionaries)
    Returns:
        List with all courses in English
    """
    print("-- Translating courses...")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    trans_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    engl_courses = []
    for idx, course in enumerate(courses):
        engl_courses.append(course)

        # Translate the description for the embeddings and save it in a separate key; keep the original description for presenting it to the user
        emb_description = course['description']
        if course['description'] and detect(course['description']) == 'de':
            # If the description is too long to be translated (> 512 characters), split it into smaller parts 
            if len(course['description']) > 450:  # Somewhat lower than 512 as safe margin
                sentences = course['description'].split(". ") 
                parts, curr_part = [], ""
                for s in sentences:
                    if len(curr_part + s) < 450:
                        curr_part += s + ". "
                    else:
                        parts.append(curr_part.strip())
                        curr_part = s + ". "
                parts.append(curr_part.strip())  # Add last chunk
            else:
                parts = [course['description']]
            translated_parts = []
            for part in parts:
                tokenized_text = tokenizer(part, return_tensors="pt", padding=True)
                translated = trans_model.generate(**tokenized_text)
                translated_parts.append(tokenizer.decode(translated[0], skip_special_tokens=True))
            emb_description = " ".join(translated_parts)

            # If no language was given for a course, assume that the course is held in the same language as the description is written in
            if course['language'] in ["", "not specified"]:
                course['language'] = "German"
        elif course['description'] and (course['language'] in ["", "not specified"]) and detect(course['description']) == 'en':
            course['language'] = "English"

        # Translate the title for the embeddings
        if detect(course['title']) == 'de':
            tokenized_text = tokenizer(course['title'], return_tensors="pt", padding=True)
            translated = trans_model.generate(**tokenized_text)
            emb_title = tokenizer.decode(translated[0], skip_special_tokens=True)
            # If there is neither a language nor description for the course, set the language to the one detected in the title
            if course['language'] in ["", "not specified"]:
                course['language'] = "German"
        else:
            emb_title = course['title']
            if course['language'] in ["", "not specified"]:
                course['language'] = "English"
        
        # Add a new attribute that combines the title and description for the embedding
        engl_courses[idx]['title_descr'] = " - ".join([emb_title, emb_description])

    print("--> Done translating!")
    return engl_courses


def combine_strings(string1, string2):
    """
    Compares two given strings and combines them (sentence by sentence) if there are differences.

    Parameters:
        string1 (str): first string to compare
        string2 (str): string to compare to the first one
    Returns:
        all_sentences: String containing all sentences of both strings
    """
    # Split both strings into sentences
    string1 = string1.replace("\n", " ")
    sentences1 = string1.split(". ")
    string2 = string2.replace("\n", " ")
    sentences2 = string2.split(". ")
    
    # Make sure the last sentence is split correctly
    if sentences1[-1].endswith("."):
        sentences1[-1] = sentences1[-1][:-1]
    if sentences2[-1].endswith("."):
        sentences2[-1] = sentences2[-1][:-1]

    # Save sentences that have not been combined
    rest1 = sentences1.copy()

    # Compare each sentence
    combined_sentences = []
    for s1 in sentences1:
        found_same = False
        for s2 in sentences2:
            if found_same == True:
                continue
            words_count = 0
            different_words = 0
            diff = list(difflib.ndiff(s1.split(), s2.split()))
            for word in diff:
                words_count += 1
                if word.startswith("- "):  # Word in s1 but not in s2
                    different_words += 1
                elif word.startswith("+ "):  # Word in s2 but not in s1
                    different_words += 1
           
            # If the sentences are exactly the same or very similar, add the longer one to the combined string and continue with the next sentence
            if different_words == 0 or different_words/words_count < 0.5:
                combined_sentences.append(max([s1, s2], key = len))
                rest1.remove(s1)
                sentences2.remove(s2)
                found_same = True
            
    # Add the remaining sentences of both strings
    combined_sentences += rest1
    combined_sentences += sentences2

    all_sentences = ". ".join(combined_sentences)
    return all_sentences


def expand_abbreviations(text, abbreviations):
    """
    Function that expands common abbreviations

        Parameter: 
            text: text to check for abbreviations
            abbreviations: dictionary containing all abbreviations to search for
        Returns: text in which every abbreviation (that is in the dictionary) is replaced with its full form
    """
    for abbrev, full_form in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, text, flags=re.IGNORECASE)
    return text


def combine_courses(course_dict, semester_file):
    """
    Compares courses from different semesters (from given files) and merges those with the same title and status. 
    Used to create a list with all courses from previous semesters without duplicates.

    Parameters:
        course_dict (dict): dictionary containing courses
        semester_file (str): Name of a semester file with courses to compare and combine with the course_dict
    Returns:
        dictionary of combined courses
    """
    # Read and translate the courses from the xml file
    courses_to_add = read_xml(semester_file)
    courses_to_add = translate_courses(courses_to_add)

    rest1 = courses_to_add  # Contains all courses from courses_to_add that have not matched one from course_dict to append at the end    
    for c0 in course_dict:
        for c1 in courses_to_add:
            combined = False
            # If the title and status are the same, consider it to be the same course
            if c0['title'] == c1['title']:
                if c0['status'] == c1['status']:
                    combined = True
                    # If the descriptions or modules are different, combine them
                    if c0['description'] != c1['description']:
                        c0['description'] = combine_strings(c0['description'], c1['description'])
                    if c0['module'] != c1['module']:
                        c0['module'] = list(set(c0['module'] + c1['module']))
                else:
                    # If two courses have the same title but a different status, add the status to the title
                    c0['title'] += f" ({c0['status']})"
                    c1['title'] += f" ({c1['status']})"
            #If a course from courses_to_add was combined with one from the other file, it does not need to be appended later
            if combined:
                rest1.remove(c1)
    course_dict += rest1
    return course_dict


def all_attributes(courses):
    """
    Extracts all attributes (except from sem_key, title, description) with all possible values

    Parameter:
        courses (list): list with all courses to extract attributes from
    Returns:
        dictionary with all attributes as keys and all possible values of each attributes (as sets) as respective values
    """
    print(f"-- Extracting all attributes...")
    all_attr_sets = {'ects': set(), 'prerequisites': set(), 'status': set(), 'sws': set(), 'mode': set(), 'language': set(), 'time': set(), 'filter_time': {}, 'lecturer': set(), 'lecturer_short': set(), 'module': set(), 'area': set(), 'home_institute': set()}
    # Loop through all courses to find all attributes from each course
    for c in courses:
        for attr, val in c.items():
            if attr in all_attr_sets.keys():
                # If the attribute's value is a list, add each value of the list
                if isinstance(val, list):
                    for v in val:
                        all_attr_sets[attr].add(v)
                # If it's a dictionary (only the case for 'filter_time'), add each key with the corresponding values
                elif isinstance(val, dict):
                    for day, time in val.items():
                        if day in all_attr_sets[attr]:
                            all_attr_sets[attr][day] += time.copy()
                        else:
                            all_attr_sets[attr][day] = time.copy()
                # Otherwise, simply add the attribute's value
                else:
                    all_attr_sets[attr].add(val)

    # Merge overlapping timeframes of 'filter_time'
    merged_times = {}
    for day, times in all_attr_sets['filter_time'].items():
        times.sort(key=lambda x: x[0])

        # Initialize the merged list with the first interval
        merged = [times[0]]
        for current in times[1:]:
            previous = merged[-1]
            # If two timeframes overlap, merge them
            if current[0] <= previous[1]:
                previous[1] = max(previous[1], current[1])
            # Otherwise, add the interval to the result
            else:
                merged.append(current)
        merged_times[day] = merged

    # Sort filter_time by days
    weekday_order = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]
    all_attr_sets['filter_time'] = {day: sorted(merged_times[day]) for day in weekday_order if day in merged_times}

    # Sort the values of each attribute
    all_attr = {key: list(value) if isinstance(value, set) else value for key, value in all_attr_sets.items()}
    sorted_attr = {}
    for f_key, f_value in all_attr.items():
        if isinstance(f_value, list):
            # Sort values by numbers and remove empty values and 'not specified'
            if f_key in ['sws', 'ects']:
                int_values = [int(val) for val in f_value if val and val.isdigit()]
                int_values = sorted(int_values)
                sorted_attr[f_key] = [str(val) for val in int_values]
            else:
                values = [val for val in f_value if not (val in [None, 'not specified', ''])]
                sorted_attr[f_key] = sorted(values)
        else:
            sorted_attr[f_key] = f_value
    return sorted_attr


def save_courses():
    """
    Saves all courses from the xml files in the data folder (2 lists: all previous courses and the current ones) and the possible attribute values in a file

    Returns:
        dictionary containing all courses (keys: 'current' & 'past')
    """
    semester_files = ['data/' + file for file in os.listdir('data')]
    print(f"Saving courses from files {semester_files}...")

    # Find most current semester
    most_current = (None, [0])  # (index, semester)
    for sem in semester_files:
        # Extract the years from the file name
        year = re.findall(r'\d+', sem)
        year = [int(y) for y in year]
        # If the first year is bigger than the first of the currently most current semester, it is more current
        if year[0] > most_current[1][0]:
            most_current = (sem, year)
        # If the first year of both are the same, one is the SoSe and the other one is the WiSe starting in the same year
        # As the WiSe (the one having 2 years in the name) comes after the SoSe, it is more current
        elif (year[0] == most_current[1][0]) and (len(year) > len(most_current[1])):
            most_current = (sem, year)
    # Remove the most current semester from the list of semesters to get a list containing only past semesters
    semester_files.remove(most_current[0])

    # Get the course data for the current semester
    print(f"\nProcessing the most current semester ({most_current[0]})...")
    current_courses = read_xml(most_current[0])
    current_courses = translate_courses(current_courses)

    # Get the course data for the previous semesters
    print(f"\nProcessing the past semester(s) ({semester_files})...")
    prev_courses = read_xml(semester_files[0])
    prev_courses = translate_courses(prev_courses)
    # If there are multiple previous semesters, combine them to a single list without duplicated courses
    if len(semester_files) > 1:
        for sem_file in semester_files[1:]:
            prev_courses = combine_courses(prev_courses, sem_file)

    # Get all possible attribute values from the most current semester
    current_attr = all_attributes(current_courses)

    # Save the courses and attributes in a json file
    courses = {"current": current_courses, "past": prev_courses, "all_attr": current_attr}
    with open(std_course_file, "w") as file:
        json.dump(courses, file, indent=4)
    print("\n>> All courses and attributes were saved!\n")
    return courses

 
def attribute_embeddings(courses):
    """
    Creates embeddings for the course's attributes

    Parameters:
        courses (list): list of courses to embed
    Returns:
        list with all embeddings
    """
    print("-- Creating course embeddings...")
    attr_emb = {}
    for c in courses:
        attr_emb[c['title']] = {}
        # Get the embedding of each of these attributes for the course
        for attr in ['title', 'title_descr', 'module', 'status', 'mode', 'lecturer_short', 'area', 'home_institute']:
            curr_attr = str(c[attr])
            # In the combined title and description, expand all common abbreviations (such as 'AI') for a more accurate similarity calculation (in recommender.py)
            if attr == 'title_descr':
                curr_attr = expand_abbreviations(str(c[attr]), abbreviations)
            attr_emb[c['title']][attr] = model.encode(curr_attr)
    print("--> Embeddings ready!")
    return attr_emb


def save_embedding(new_emb_key, new_emb_val):
    """
    Saves an embedding in the embedding file
    
    Parameters:
        new_emb_key (str): Key (name) of the embedding
        new_emb_val (str): The embedding
    """
    # Check if the save file for embeddings already exists
    if os.path.exists(std_emb_file):
        # Load existing embeddings
        current_data = np.load(std_emb_file, allow_pickle=True)
        embeddings = {key: current_data[key] for key in current_data}
        current_data.close()
    else:
        # If the file does not exists, create an empty dictionary
        embeddings = {}

    # Add or update with the new embedding
    embeddings[new_emb_key] = new_emb_val

    # Save all embeddings in the save file
    np.savez(std_emb_file, **embeddings)
    print(f"--> Embedding '{new_emb_key}' has been saved/updated.")


def save_course_embeddings(courses = None):
    """
    Saves the embeddings of all courses in a given file

    Parameters:
        courses (dict): current and previous courses (keys: "current" & "past")
    """
    print("Saving all embeddings...")
    current_courses = []
    prev_courses = []

    # Load courses from the course file, if not given as parameter
    if courses is None:
        with open(std_course_file, "r") as file:
            courses = json.load(file)
            current_courses = courses["current"]
            prev_courses = courses["past"]
    else:
        current_courses = courses["current"]
        prev_courses = courses["past"]

    # Embed and save the current semester
    current_emb = attribute_embeddings(courses = current_courses)
    save_embedding("current_courses", current_emb)

    # Embed and save the previous semesters
    prev_emb = attribute_embeddings(courses = prev_courses)
    save_embedding("prev_courses", prev_emb)


def intent_embeddings():
    """
    Creates and saves embeddings for all intent categories (for the example sentences in data_prep_variables.py)
    """
    # Create and save the embeddings of the example sentences
    intent_embeddings = {intent: model.encode(sentences) for intent, sentences in intent_examples.items()}
    save_embedding("intent", intent_embeddings)


def prepare_data():
    """
    Computes, encodes and saves all relevant data (from the data folder)
    No need to manually run any other function from this file
    """
    # Get and save all courses from the xml files in the data folder
    courses = save_courses()
    # Create and save the embeddings of all courses
    save_course_embeddings(courses = courses)
    # Create and save the embeddings for the example sentences for each intent category
    intent_embeddings()
    print("Finished preprocessing!")


if __name__ == "__main__":
    prepare_data()


###--- Using individual save functions ---###

## Save courses
#save_courses()

## Save embeddings
#save_course_embeddings()
#intent_embeddings()