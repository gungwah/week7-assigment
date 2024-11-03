import streamlit as st
import langchain as langchain
from langchain_community.llms import OpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.prompts import PromptTemplate
import os

langchain.debug = True #i deploy this app on my local so that i can see the result of each chain whether it performs as expected or no

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


positive_response = PromptTemplate.from_template("thank them for their feedback and for choosing to fly with the airline.") | llm | StrOutputParser()
airline_fault_response = PromptTemplate.from_template("display a message offering sympathies and inform the user that customer service will contact them soon to resolve the issue or provide compensation.") | llm | StrOutputParser()
external_factors_response = PromptTemplate.from_template("display a message implying sympathy but explain that the airline is not liable in such situations.") | llm | StrOutputParser()

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