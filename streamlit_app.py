import streamlit as st
from langchain_community.llms import OpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

llm = OpenAI(openai_api_key=st.secrets["OpenAIkey"])

sentiment_template = """Analyze the sentiment of this feedback:
'{feedback}'

Only respond with "positive" or "negative".
"""
cause_template = """If the experience is negative, determine the cause:
'{feedback}'

Only respond with "airline fault" or "external factors".
"""

sentiment_chain = PromptTemplate.from_template(sentiment_template) | llm | StrOutputParser()
cause_chain = PromptTemplate.from_template(cause_template) | llm | StrOutputParser()


positive_response = PromptTemplate.from_template("Thank you for choosing our airline! We're glad you had a good experience.") | llm
airline_fault_response = PromptTemplate.from_template("We apologize for the inconvenience. Our customer service team will contact you soon to resolve the issue.") | llm
external_factors_response = PromptTemplate.from_template("We’re sorry for your experience. Unfortunately, the airline is not liable for events outside of our control.") | llm

feedback_branch = RunnableBranch(
    (lambda x: "positive" in x["sentiment"].lower(), positive_response),
    (lambda x: "negative" in x["sentiment"].lower() and "airline fault" in x["cause"].lower(), airline_fault_response),
    external_factors_response,
)

full_chain = {"sentiment": sentiment_chain, "cause": cause_chain, "feedback": lambda x: x["feedback"]} | feedback_branch


st.title("Airline Feedback App")
st.header("Share with us your experience of the latest trip:")

user_feedback = st.text_area("Your feedback")

if st.button("Submit Feedback"):
    if user_feedback:
        result = full_chain.invoke({"feedback": user_feedback})
        st.write(result)
    else:
        st.write("Please enter your feedback.")