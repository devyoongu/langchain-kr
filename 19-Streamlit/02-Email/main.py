import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
import glob
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
from langchain_community.utilities import SerpAPIWrapper


# API KEY 정보로드
load_dotenv()

st.title("나만의 Email 요약 챗GPT💬")

os.environ["SERPAPI_API_KEY"] = (
    "ad0069e2c4bf1b7160774cd361ba2e9d9de33566b4d5c42769b43cb6554efd1c"
)


class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    company: str = Field(description="메일을 보낸 사람의 회사")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    task_input = st.text_input("TASK 입력", "")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# email 내용을 EmailSummary로 파싱
def create_email_parsing_chain():

    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

    EMAIL CONVERSATION:
    {email_conversation}

    FORMAT:
    {format}
    """
    )

    # format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
    prompt = prompt.partial(format=output_parser.get_format_instructions())

    # GPT
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 체인 생성
    chain = prompt | llm | output_parser

    return chain


# email.yml 형태로 파싱
def create_report_chain():
    report_prompt = load_prompt("prompts/email.yaml", encoding="utf-8")

    output_parser = StrOutputParser()

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    chain = report_prompt | llm | output_parser

    return chain


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # 1) email 을 파싱하는 chain 을 생성 및 실행
    email_chain = create_email_parsing_chain()
    # EmailSummary class 형태로 저장
    answer = email_chain.invoke({"email_conversation": user_input})

    # 2) 보낸 사람의 추가 정보수집(검색)
    params = {
        "engine": "google",
        "gl": "kr",
        "hl": "ko",
    }  # 검색 파라미터 설정
    search = SerpAPIWrapper(params=params)  # 검색 객체 생성
    search_query = f"{answer.person} {answer.company} {answer.email}"
    search_result = search.run(search_query)
    search_result = eval(search_result)

    if isinstance(search_result, list):
        search_result_string = " ".join(search_result)
    elif isinstance(search_result, str):
        search_result_string = search_result
    else:
        search_result_string = str(search_result)

    # 3) 이메일 요약 리포트 생성
    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result_string,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }

    # 스트리밍 호출
    response = report_chain.stream(report_chain_input)
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
