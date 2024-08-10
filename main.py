from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
from pywebio.pin import *
from pywebio import start_server,config
from openai import OpenAI
from pywebio import start_server
#创建聊天模型
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from serpapi import GoogleSearch
import re
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import asyncio
import nest_asyncio
from langchain.agents import load_tools, initialize_agent, AgentType
from pywebio.session import go_app, run_async, hold,run_js
from datetime import datetime
import time
# 导入 Langchain 库的 ChatOpenAI 类，用于与 OpenAI 聊天模型进行交互。
from langchain.chat_models import ChatOpenAI  

# 导入 PromptTemplate 模块，用于创建和管理提示模板。
from langchain.prompts import PromptTemplate  
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
# 定义主题分类器模型
from langchain.pydantic_v1 import BaseModel
from typing import Literal
# 创建最终的处理链
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

def apply_custom_css():
    run_js("""
        document.head.innerHTML += `
        document.querySelector('input[type="text"]').style.backgroundColor = '#FFE4C4';
        document.querySelector('input[type="file"]').parentElement.style.backgroundColor = '#FFE4C4';
            <style>
                body {
                    font-family: "Arial", sans-serif;
                    font-size: 18px;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    height: 90vh;
                    justify-content: space-between;
                    background-color: #FFE4C4;  /* 可以调整背景颜色 */
                }
                .footer {
                    text-align: center;
                    width: 100%;
                }
                .application-index > li > a {
                    font-size: 24px;
                    font-weight: bold;
                }
                .application-index > ul {
                    margin-left: 20px;
                }
                .application-index > ul > li > a {
                    font-size: 18px;
                }
                .notice-board {
                    background-color: #f9f9f9;
                    padding: 15px;
                    margin-left: 1px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 50px;
                    width: 90%;
                }
            </style>
        `;
    """)

# 假设上次登录时间为一个变量
last_login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

arxiv_re=""
#设定AI的角色和目标
role_template = "你是一个在线笔记AI助手, 你的目标是帮助用户能够对笔记中的内容进行解释，总结，拓展"

cot_template = """
作为一个在线笔记AI助手,我的目标是尽可能帮助用户全面了解关于他提供内容的有用信息。 
我会按部就班的思考，先理解用户的需求，用尽可能短的话语高度概括答案，然后考虑与他提供的内容相关的知识点，最后根据这个需求，给出我的解答。
同时，我也会向客户说明我的信息来源,并保证来源尽量可靠,提供相关知识链接,尤其是那些高引用量的链接。我要用markdown格式返回结果
示例: 
人类：傅里叶变换的原理是什么。 
AI:首先，我理解你正在寻找有关傅里叶变换背后涉及的理论。
傅里叶变换(Fourier transform)是一种线性的积分变换。因其基本思想首先由法国学者傅里叶系统地提出，所以以其名字来命名以示纪念。
简单而言,它是一种从时域转换为频域的变化。我的理由主要来源于预训练时的数据库和网络信息,详细资料你可以参考:https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/。
"""

system_prompt_role=SystemMessagePromptTemplate.from_template(role_template)
system_prompt_cot=SystemMessagePromptTemplate.from_template(cot_template)

#用户提问
human_trmplate="{question}"
human_prompt=HumanMessagePromptTemplate.from_template(human_trmplate)

#将以上所有信息结合为一个聊天提示
chat_prompt= ChatPromptTemplate.from_messages([system_prompt_role,system_prompt_cot,human_prompt])

llm=ChatOpenAI(
    model='gpt-4o',
    temperature=0,
)

def extract_content(text):
    # 正则表达式模式，匹配 web: 后面的所有内容
    pattern = re.compile(r'web:\s*(.*)')
    # 查找匹配的模式
    match = pattern.search(text)
    if match:
        content = match.group(1).strip()
        return content
    return None

def extract_arxiv_id(text):
    # 使用正则表达式查找 arXiv 链接中的标识符
    match = re.search(r"https://arxiv\.org/abs/(\d+\.\d+)", text)
    if match:
        # 返回捕获组中的标识符
        return match.group(1)
    return None

def output(references,ask):
    # 将搜索结果添加到提示中
    references_text = "\n\n".join(references)
    chat_prompt= ChatPromptTemplate.from_messages([system_prompt_role,system_prompt_cot,human_prompt])
    prompt_text = f"以下是我找到的相关参考资料：\n{references_text}\n\n"
    if ask.lower().startswith("web"):
        prompt_2 = chat_prompt.format_prompt(question=prompt_text + "请完善我提供的资料：{}".format(ask))
    else:
        prompt_2 = chat_prompt.format_prompt(question=arxiv_re+prompt_text + "请筛选这些资料并参考具体链接，重点参考前半句信息，结合你原有的知识回答以下问题,优先提供可靠的来源：{}".format(ask))
    #接收用户的询问，返回回答结果
    with put_loading():
        put_text("Generating...")
        response = llm.invoke(prompt_2)
        time.sleep(2)  
    with use_scope('refresh', clear=True):
        put_markdown("\n**refreshed Answer:**\n")
        put_scrollable(put_markdown(response.content), height=375)
    

def web(ask):
    order=extract_content(ask)
    nest_asyncio.apply()


    # 实例化浏览器
    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()


    # 初始化ChatOpenAI实例
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
    )

    # 初始化代理，设置代理类型为结构化工具对话代理
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations = 15,
    )
    with put_loading():
        put_text("Generating...")
        # 定义异步主函数
        async def main():
            try:
                references = await agent_chain.arun("{} , but no more than 100 words overall, return in a markdown format".format(order))
                # put_text("Generating")
                return references
            except Exception as e:
                put_text(f"An error occurred: {e}")
                return None
        # 运行异步事件循环
        loop = asyncio.get_event_loop()
        references=loop.run_until_complete(main())
        time.sleep(2)
    with use_scope('web', clear=True):
        put_markdown("\n **Supplementary Information** \n")
        put_markdown(references)
    return references


def arxiv_sum(references):
    text = references[0]
    # 提取 arXiv 标识符
    arxiv_id = extract_arxiv_id(text)
    if arxiv_id:
        tools = load_tools(["arxiv"])

        # 初始化代理链
        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True  # 输出详细过程
        )
        with put_loading():
            put_text("Generating...")
            arxiv_re=agent_chain.run("paper_id = {}".format(arxiv_id))
            time.sleep(2)
        with use_scope('arxiv', clear=True):
            put_markdown("\n**arxiv summarize:**\n")
            put_scrollable(put_markdown(arxiv_re) ,height=70)
        return arxiv_re
    with use_scope('arxiv', clear=True):
        put_text("No arXiv ID found.")

def google_search(ask):
    references = []
    # 设置你的 SerpApi API 密钥
    api_key = "2de8a8d9aa7f5bd315a5298d7ee796c6c7d5d3b5c055f3036af5f29a3d71aaf8"

    # 定义搜索参数
    params = {
        "engine": "google",
        "q": "{} ".format(ask),
        "api_key": api_key,
    }

    # 创建 GoogleSearch 对象
    search = GoogleSearch(params)
    with put_loading():
        put_text("Generating...")
        # 执行搜索并获取结果
        results = search.get_dict()
    
        # 处理和输出搜索结果
        references = []
        for result in results.get('organic_results', []):
            title = result.get('title')
            snippet = result.get('snippet')
            link = result.get('link')
            references.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
        content = []
        for reference in references:
            content.append(put_markdown(reference))
            content.append(put_text())
        time.sleep(2)  
    with use_scope('google', clear=True):
        put_markdown("\n**以下是Google Search的原始答案:**\n ")
        put_scrollable(content, height=75)
    if ask.lower().startswith("arxiv"):
        references+=arxiv_sum(references)
    output(references,ask)

def ans(ask):
    clear('web')
    clear('arxiv')
    clear('google')
    with put_loading():
        put_text("Generating...")
        prompt = chat_prompt.format_prompt(question="{}".format(ask))
        #接收用户的询问，返回回答结果
        response = llm.invoke(prompt)
        time.sleep(2)  
    with use_scope('GPT_ori', clear=True):
        put_markdown("**以下是GPT4o的原始答案:**\n")
        put_scrollable(put_markdown("{}".format(response.content)), height=75)
    if not ask.lower().startswith("web"):
        google_search(ask)
    else:
        ref=web(ask)
        output(ref,ask)


def add_footer():
    put_html('<div class="footer">2024© Wenqi Guan</div>')

@config(description="欢迎页面 - 在线笔记网站的介绍")
def welcome():
    # 创建一个时钟的 HTML 和 CSS
    clock_html = '''
    <div id="clock" style="position: fixed; top: 50px; right: 50px; font-family: 'Elephant', sans-serif; font-size: 36px; color: #000000; background: #FFE4E1; padding: 30px; border-radius: 50px; text-align: center; box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);">
        <div id="time"></div>
        <div id="quote" style="font-size: 18px; font-family: 'STXingkai';margin-top: 5px; color: #000000;">
            Never leave that until tomorrow, which you can do today. <br>
            <span style="display:inline-block; width: 300px;"></span>— Benjamin Franklin
        </div>
    </div>
    <style>
        #clock {
            border: 5px solid #87CEEB;
            border-radius: 15px;
        }
        #time {
            font-family: 'Elephant', monospace;
            font-size: 48px;
            color: #000000;
        }
        @font-face {
            font-family: 'Elephant';
            src: url('https://fonts.gstatic.com/s/digital7.woff2') format('woff2');
        }
    </style>
    <script>
        function updateTime() {
            var now = new Date();
            var hours = String(now.getHours()).padStart(2, '0');
            var minutes = String(now.getMinutes()).padStart(2, '0');
            var seconds = String(now.getSeconds()).padStart(2, '0');
            document.getElementById('time').innerHTML = hours + ':' + minutes + ':' + seconds;
        }
        setInterval(updateTime, 1000);
        updateTime();
    </script>
    '''
    put_html(clock_html)
    response=llm.invoke("给热爱学习的自己一句自我勉励的话语")
    apply_custom_css()
    put_markdown("# 欢迎来到在线笔记网站")
    put_markdown("""
    这是一个简单而强大的在线笔记网站，您可以在这里创建、编辑和管理您的笔记。
    
    功能包括：
    - 推理模块
    - 创建和编辑现有笔记
    - 笔记检索
    - AI助手
    """)
    put_buttons(['推理模块', '编辑笔记','笔记检索','AI助手'], [lambda: go_app('deduction', new_window=False), lambda: go_app('view_notes', new_window=False),lambda: go_app('retriever', new_window=False),lambda: go_app('AI_search', new_window=False)])
    put_row([
        put_image('https://s2.loli.net/2024/08/02/ozXA47Sysg2huIc.png', width='87%'),
        put_html(f"""
            <div class="notice-board">
                <h3>公告栏</h3>
                <p><strong>上次登录时间：</strong>{last_login_time}</p>
                <p><strong>最近更新：</strong>"2024-08-04 by Guan"</p>
                <p><span style="font-size: 20px;">{response.content}</span></p>
                <p><strong>Just for Test</strong></p>
                <p><strong>如想试用,可连接404Wifi后登陆192.168.1.117:8080</strong></p>
            </div>
        """)
    ])
    add_footer()


@config(description="推理模块 - 专注逻辑推理,数学运算")
def deduction():
    put_buttons(['返回欢迎页面'], [lambda: go_app('welcome', new_window=False)])
    def ded(question):
        SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
        SPARKAI_APP_ID = '4f222d95'
        SPARKAI_API_SECRET = 'N2UxNmFlMmU0OGRjZWQxMzBjZjNhZmVm'
        SPARKAI_API_KEY = '286628a26f1bdf2bae0a541008e3f936'
        SPARKAI_DOMAIN = '4.0Ultra'
        with put_loading():
            put_text("Generating...")
            llm=ChatOpenAI(
            model='gpt-4o',
            temperature=0,)
            response = llm.invoke("{}".format(question))
            time.sleep(1)  
        with use_scope('GPT_or', clear=True):
            put_markdown("**以下是GPT4o的原始答案:**\n")
            put_scrollable(put_markdown("{}".format(response.content)), height=100)
        with put_loading():
            put_text("Generating...")
            # 定义物理问题的提示模板
            physics_template = """
            你是一位非常聪明的物理教授。你擅长以简明易懂的方式回答物理问题。当你不知道某个问题的答案时，你会承认自己不知道，同时审视他人的答案，选择采纳或不予理睬。

            以下是一个问题：
            {input}
            """
            physics_prompt = PromptTemplate.from_template(physics_template)

            # 定义数学问题的提示模板
            math_template = """
            你是一个非常优秀的数学家。你擅长回答数学问题。你之所以这么厉害，是因为你能够先自己思考，再参考其他思路，在思考后加工并完善你的答案
            这里有一个问题：
            {input}
            """
            math_prompt = PromptTemplate.from_template(math_template)

            # 定义通用问题的提示模板
            general_prompt = PromptTemplate.from_template(
                "您是一个很有帮助的助手。尽可能准确地回答问题。"
            )

            # 创建基于条件的提示分支
            prompt_branch = RunnableBranch(
                (lambda x: x["topic"] == "物理", physics_prompt),
                (lambda x: x["topic"] == "数学", math_prompt),
                general_prompt
            )
            class TopicClassifier(BaseModel):
                "分类用户问题的主题"
                topic: Literal["物理", "数学", "通用"]
                "用户问题的主题。其中之一是'数学','物理'或'通用'。"
            # 创建主题分类器函数
            classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
            # 创建 ChatOpenAI 实例并绑定主题分类器函数
            llm = ChatOpenAI(model_name="gpt-4o").bind(
                functions=[classifier_function], function_call={"name": "TopicClassifier"}
            )

            # 创建解析器
            parser = PydanticAttrOutputFunctionsParser(
                pydantic_schema=TopicClassifier, attr_name="topic"
            )

            # 创建分类链
            classifier_chain = llm | parser

            # 创建最终的处理链
            final_chain = (
                RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
                | prompt_branch
                | ChatOpenAI()
                | StrOutputParser()
            )
            spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
            )
            messages = [ChatMessage(
                role="user",
                content='{}，第一句请先重复说一遍问题，第二句以”我按照以下方式推理”为开头通过逐层推理得到答案，要说明思考过程。最后一句话以"你怎么看这个问题"结束'.format(question)
            )]
            handler = ChunkPrintHandler()
            a = spark.generate([messages], callbacks=[handler])
            time.sleep(1)
        with use_scope('Other', clear=True):
            put_markdown("**Other Thought:**\n")
            put_scrollable(put_markdown("{}".format(a.generations[0][0].text)), height=100)
        with put_loading():
            put_text("Generating...")
            response = final_chain.invoke({"input": "{},参考提供的思路，逐步思考问题".format(a.generations[0][0].text)})
            time.sleep(1)
        with use_scope('router_chain', clear=True):
            put_markdown("**以下是Router_Chain的答案:**\n")
            put_scrollable(put_markdown("{}".format(response)), height=350)
    put_input('ask', label='请输入您的问题：', type=TEXT, placeholder='Type in here', 
        help_text='目前仅针对数理问题有较好表现，但仍处于测试阶段')
    # 创建提交按钮
    put_buttons(['确认', '重置'], [lambda: ded(pin.ask), lambda: pin_update('ask', value='')])
    hold()

global reop
reop=False

@config(description="查看和编辑笔记 - 编辑笔记")
def view_notes():
    global reop
    def reopen():
        global reop
        try:
            with open("./temp.txt", 'r', encoding='utf-8') as f:
                content = f.read()
            res = textarea('Text area', code={
                'mode': "markdown",
                'theme': 'lightula'
            }, value=content)
            put_markdown(res)
            reop=True
        except FileNotFoundError:
            put_error("no temp file")
            reop=False
    def cf(file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            pass  # 不写入任何内容
    def create_file():
        put_input('path', label='请输入保存的路径：', type=TEXT, placeholder='Type in here', 
        help_text="sample: ./final/filename.txt 是服务器用于保存文件的路径")
        while True:
            action = actions('目前仅支持.txt文件格式创建', buttons=['确认', '重置'])
            
            if action == '确认':
                cf(pin.path)
                break
            elif action == '重置':
                pin_update('path', value='')
        global reop
        reop=False


    action = actions('选择一个操作', buttons=['返回欢迎页面', '恢复缓存','打开一个文件','创建新文件'])
    if action == '返回欢迎页面':
        go_app('welcome', new_window=False)
    elif action == '恢复缓存':
        reopen()
    elif action== '创建新文件':
        create_file()
    else:
        reop=False

    if reop==False:
        # 上传文件并读取内容
        txts = file_upload("Select some txts:", accept=".txt", multiple=True)
        content = ""
        for txt in txts:
            content += txt['content'].decode('utf-8') + "\n"

        global res
        # 将内容显示在 textarea 中
        res = textarea('Text_area', code={
            'mode': "markdown",
            'theme': 'lightula'
        }, value=content)  # 使用 value 参数设置初始值

    # 打印 textarea 的内容
    put_markdown(res)

    def save(file_path):
        global res
        global reop
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(res)
        put_markdown("笔记已保存")
        reop=False
        clear()
        view_notes()
    # 添加保存、删除、暂存按钮
    def save_note():
        global reop
        put_input('path', label='请输入保存的路径：', type=TEXT, placeholder='Type in here', 
        help_text='./final/filename.txt 是服务器用于保存文件的路径')
        # 创建提交按钮
        put_buttons(['确认', '重置'], [lambda: save(pin.path), lambda: pin_update('path', value='')])
        reop=False

    def temp_save():
        global reop
        save("./temp.txt")
        reop=False

    def delete_all():
        global reop
        reop=False
        clear()
        view_notes()

    put_buttons(['保存', '取消', '暂存'], [save_note,delete_all, temp_save])

    hold()  # 保持当前界面

@config(description="笔记检索页面 - 复习检索已有笔记")
def retriever():
    # llm_model = OpenAI(
    # temperature=0,
    # )
    put_buttons(['返回欢迎页面'], [lambda: go_app('welcome', new_window=False)])
    # txts = file_upload("Select some txts:", accept=".txt", multiple=True)
    # files_content = {}

    # # for file in txts:
    # #     file_name = file['filename']  
    # #     file_content = file['content'].decode('utf-8')  
    # #     files_content[file_name] = file_content
    
    # choice = radio("Please select an option:", options=['Strict', 'Broad'])
    put_markdown("**Todo**")
    # if choice==Broad:

    hold()
    

@config(description="AI助手界面 - 享受强大的AI")
def AI_search():
    put_buttons(['返回欢迎页面'], [lambda: go_app('welcome', new_window=False)])
    put_input('ask', label='请输入你的问题：', type=TEXT, placeholder='Type in here', 
              help_text='web:+ command ; arxiv:+website(http:...)')
    # 创建提交按钮
    put_buttons(['提交', '重置'], [lambda: ans(pin.ask), lambda: pin_update('ask', value='')])
    hold()  # 保持当前界面


if __name__ == '__main__':
    start_server({'welcome': welcome, 'deduction': deduction, 'view_notes': view_notes, 'retriever': retriever,'AI_search': AI_search}, port=8080, debug=True)


