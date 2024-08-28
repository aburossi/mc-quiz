import streamlit as st
import time  # Import the time module to use sleep

# This must be the first Streamlit command
st.set_page_config(page_title="SmartExam Creator", page_icon="📝")

import os
import json
from PyPDF2 import PdfReader
from fpdf import FPDF
import dotenv
from openai import OpenAI

__version__ = "1.1.0"

# Main app functions
def stream_llm_response(messages, model_params, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_params.get("model", "gpt-4o"),
        messages=messages,
        temperature=model_params.get("temperature", 0.3),
        max_tokens=4096,
    )
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def summarize_text(text, api_key=st.secrets["OPENAI_API_KEY"]):
    prompt = (
        "Please summarize the following text to be concise and to the point:\n\n" + text
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    summary = stream_llm_response(messages, model_params={"model": "gpt-4o-mini", "temperature": 0.3}, api_key=api_key)
    return summary

def chunk_text(text, max_tokens=3000):
    sentences = text.split('. ')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_tokens:
            chunks.append(chunk)
            chunk = sentence + ". "
        else:
            chunk += sentence + ". "
    if chunk:
        chunks.append(chunk)
    return chunks

def generate_mc_questions(content_text, api_key=st.secrets["OPENAI_API_KEY"]):
    prompt = (
        """// Objective:
- You convert part of the {content} to JSON. Your output is always in German.
- Generate 15 multiple choice questions based on the key concepts and information from the provided JSON file. 
- Each question should have one correct answer and two plausible incorrect answers.

// Steps:
1. Extract Key Concepts and Information:
- Identify and extract key sections from the JSON that contain important concepts, definitions, and explanations.
- Focus on sections: "overview", "headings", "terminology", "faqs", and "key_points".

2. Formulate Questions:
- ALWAYS Create 15 questions that test the understanding of the extracted key sections.
- Ensure each question is clear and directly related to the key concepts.

3. Generate Answer Choices:
- For each question, generate one correct answer and two very plausible incorrect answers.
- Ensure that each answer is equally long. Add superficial information to wrong answers to ensure equal length.

4. Provide Feedback:
For the correct answer, provide "feedback" that includes further information in one sentence.
For each incorrect answer, provide "feedback" that includes the correct answer in one sentence.
For each question, provide "real_life_examples" 

5. Format:
The output is STRICTLY formatted according to //Output. Including "additional_info" and "real_life_examples".

// JSON Logic
Questions Array: Contains multiple question objects.
Question Object: Each object represents a single question and contains:
title: The title of the question.
type: The type of the question (e.g., MC).
question_text: The actual question text.
answers: An array of answer objects.
text: The text of the answer.
is_correct: A boolean indicating if the answer is correct.
feedback: Feedback for the answer, providing additional context and explanation.
points: Points awarded for a correct answer, e.g. '1'
penalty: Penalty points for an incorrect answer, e.g. '-0.5'
additional_info: Any extra information that might be useful, such as "real-life examples".

//Output
{
  "questions": [
    {
      "title": "Economy",
      "type": "MC",
      "question_text": "What drives the entire economy by motivating people to demand goods and services?",
      "answers": [
        {
          "text": "Needs",
          "is_correct": true,
          "feedback": "Correct! Needs are the primary motivation for economic activity. For example, the need for food drives agricultural production and grocery sales."
        },
        {
          "text": "Wants",
          "is_correct": false,
          "feedback": "Wrong! While wants also influence economic activity, needs are more fundamental. The correct answer is needs. Needs are the primary motivation for economic activity. For example, the need for food drives agricultural production and grocery sales."
        },
        {
          "text": "Desires",
          "is_correct": false,
          "feedback": "Wrong! Desires are similar to wants and do influence economic activity, but needs are the fundamental drivers. The correct answer is needs. Needs are the primary motivation for economic activity. For example, the need for food drives agricultural production and grocery sales."
        }
      ],
      "points": 1,
      "penalty": -0.5,
      "additional_info": {
        "real_life_example": "For example, the need for food drives agricultural production and grocery sales."
      }
    },
    {
      "title": "Supply and Demand",
      "type": "MC",
      "question_text": "What is the outcome when supply and demand in a market balance each other?",
      "answers": [
        {
          "text": "Market equilibrium",
          "is_correct": true,
          "feedback": "Correct! Market equilibrium occurs when the quantity supplied equals the quantity demanded. For example, when the supply of smartphones matches consumer demand, prices remain stable."
        },
        {
          "text": "Price increase",
          "is_correct": false,
          "feedback": "Wrong! A price increase happens when demand exceeds supply, not when they are balanced. The correct answer is market equilibrium. Market equilibrium occurs when the quantity supplied equals the quantity demanded. For example, when the supply of smartphones matches consumer demand, prices remain stable."
        },
        {
          "text": "Surplus",
          "is_correct": false,
          "feedback": "Wrong! A surplus occurs when supply exceeds demand, not when they are balanced. The correct answer is market equilibrium. Market equilibrium occurs when the quantity supplied equals the quantity demanded. For example, when the supply of smartphones matches consumer demand, prices remain stable."
        }
      ],
      "points": 1,
      "penalty": -0.5,
      "additional_info": {
        "real_life_example": "For example, when the supply of smartphones matches consumer demand, prices remain stable."
      }
    }
  ]
}
"""
    )
    messages = [
        {"role": "user", "content": content_text},
        {"role": "user", "content": prompt},
    ]
    response = stream_llm_response(messages, model_params={"model": "gpt-4o-mini", "temperature": 0.3}, api_key=api_key)
    return response

def parse_generated_questions(response):
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        json_str = response[json_start:json_end]

        questions = json.loads(json_str)
        return questions
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        st.error("Response from OpenAI:")
        st.text(response)
        return None

def get_question(index, questions):
    return questions[index]

def initialize_session_state(questions):
    session_state = st.session_state
    session_state.current_question_index = 0
    session_state.quiz_data = get_question(session_state.current_question_index, questions)
    session_state.correct_answers = 0

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Generated Exam', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, title)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def generate_pdf(questions):
    pdf = PDF()
    pdf.add_page()

    for i, q in enumerate(questions):
        question = f"Q{i+1}: {q['question']}"
        pdf.chapter_title(question)

        choices = "\n".join(q['choices'])
        pdf.chapter_body(choices)

        correct_answer = f"Correct answer: {q['correct_answer']}"
        pdf.chapter_body(correct_answer)

        explanation = f"Explanation: {q['explanation']}"
        pdf.chapter_body(explanation)

    return pdf.output(dest="S").encode("latin1")

# Integration with the main app
def main():
    # Load your OpenAI API key from the environment variable
    dotenv.load_dotenv()
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    # Initialize app_mode if it doesn't exist
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Upload PDF & Generate Questions"
    
    st.sidebar.title("SmartExam Creator")
    
    app_mode_options = ["Upload PDF & Generate Questions", "Take the Quiz", "Download as PDF"]
    st.session_state.app_mode = st.sidebar.selectbox("Choose the app mode", app_mode_options, index=app_mode_options.index(st.session_state.app_mode))
    
    st.sidebar.markdown("## About")
    st.sidebar.video("https://youtu.be/zE3ToJLLSIY")
    st.sidebar.info(
        """
        placeholder
        """
    )
    
    if st.session_state.app_mode == "Upload PDF & Generate Questions":
        pdf_upload_app()
    elif st.session_state.app_mode == "Take the Quiz":
        if 'mc_test_generated' in st.session_state and st.session_state.mc_test_generated:
            if 'generated_questions' in st.session_state and st.session_state.generated_questions:
                mc_quiz_app()
            else:
                st.warning("No generated questions found. Please upload a PDF and generate questions first.")
        else:
            st.warning("Please upload a PDF and generate questions first.")
    elif st.session_state.app_mode == "Download as PDF":
        download_pdf_app()

def pdf_upload_app():
    st.title("Upload Your Lecture - Create Your Test Exam")
    st.subheader("Show Us the Slides and We do the Rest")

    content_text = ""
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        content_text += pdf_text
        st.success("PDF content added to the session.")
    
    if len(content_text) > 3000:
        content_text = summarize_text(content_text)

    if content_text:
        st.info("Generating the exam from the uploaded content. It will take just a minute...")
        chunks = chunk_text(content_text)
        questions = []
        for chunk in chunks:
            response = generate_mc_questions(chunk)
            parsed_questions = parse_generated_questions(response)
            if parsed_questions:
                questions.extend(parsed_questions)
        if questions:
            st.session_state.generated_questions = questions
            st.session_state.content_text = content_text
            st.session_state.mc_test_generated = True
            st.success("The game has been successfully created! Switch the Sidebar Panel to solve the exam.")
            
            # Wait 2 seconds and switch to quiz mode
            time.sleep(2)
            st.session_state.app_mode = "Take the Quiz"
            st.rerun()
            
        else:
            st.error("Failed to parse the generated questions. Please check the OpenAI response.")
    else:
        st.warning("Please upload a PDF to generate the interactive exam.")

def submit_answer(i, quiz_data):
    user_choice = st.session_state[f"user_choice_{i}"]
    st.session_state.answers[i] = user_choice
    if user_choice == quiz_data['correct_answer']:
        st.session_state.feedback[i] = ("Correct", quiz_data.get('explanation', 'No explanation available'))
        st.session_state.correct_answers += 1
    else:
        st.session_state.feedback[i] = ("Incorrect", quiz_data.get('explanation', 'No explanation available'), quiz_data['correct_answer'])

def mc_quiz_app():
    st.title('Multiple Choice Game')
    st.subheader('Here is always one correct answer per question')

    questions = st.session_state.generated_questions

    if questions:
        if 'answers' not in st.session_state:
            st.session_state.answers = [None] * len(questions)
            st.session_state.feedback = [None] * len(questions)
            st.session_state.correct_answers = 0

        for i, quiz_data in enumerate(questions):
            st.markdown(f"### Question {i+1}: {quiz_data['question']}")

            if st.session_state.answers[i] is None:
                user_choice = st.radio("Choose an answer:", quiz_data['choices'], key=f"user_choice_{i}")
                st.button(f"Submit your answer {i+1}", key=f"submit_{i}", on_click=submit_answer, args=(i, quiz_data))
            else:
                st.radio("Choose an answer:", quiz_data['choices'], key=f"user_choice_{i}", index=quiz_data['choices'].index(st.session_state.answers[i]), disabled=True)
                if st.session_state.feedback[i][0] == "Correct":
                    st.success(st.session_state.feedback[i][0])
                else:
                    st.error(f"{st.session_state.feedback[i][0]} - Correct answer: {st.session_state.feedback[i][2]}")
                st.markdown(f"Explanation: {st.session_state.feedback[i][1]}")

        if all(answer is not None for answer in st.session_state.answers):
            score = st.session_state.correct_answers
            total_questions = len(questions)
            st.write(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh;">
                    <h1 style="font-size: 3em; color: gold;">🏆</h1>
                    <h1>Your Score: {score}/{total_questions}</h1>
                </div>
            """, unsafe_allow_html=True)

def download_pdf_app():
    st.title('Download Your Exam as PDF')

    questions = st.session_state.generated_questions

    if questions:
        for i, q in enumerate(questions):
            st.markdown(f"### Q{i+1}: {q['question']}")
            for choice in q['choices']:
                st.write(choice)
            st.write(f"**Correct answer:** {q['correct_answer']}")
            st.write(f"**Explanation:** {q['explanation']}")
            st.write("---")

        pdf_bytes = generate_pdf(questions)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="generated_exam.pdf",
            mime="application/pdf"
        )

if __name__ == '__main__':
    main()
