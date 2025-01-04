# 💬 Course Recommending Chatbot

A chatbot that recommends courses based on the users preferences.
The user can either describe what kind of course they are looking for (naming topics or defining necessary attributes) or name courses they liked in the past.
They can also give feedback for recommended courses to improve further recommendations.

[![Link to the Chatbot](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recommender-chatbot.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Preprocess the data (if courses.json and/or embeddings.npz is missing or to update the data, for example, when adding a new semester)

   ```
   $ python data_prep.py
   ```

3. Run the app

   ```
   $ streamlit run chatbot.py
   ```


### Attention!
When running `data_prep.py` to preprocess the data: 
- The course data of each semester has to be in an individual xml file. The name of the file must contain the year/years of the semester (one year for summer semesters, two years for winter semesters)
- All xml files have to be in the `data` folder. 
- The chatbot (and therefore also `data_prep.py`) needs the course data of at least two semesters (the most current semester to recommend courses from and at least one previous semester to enable referencing previously liked courses).
- The `data` folder must not contain any other files than the xml files with the course data.
