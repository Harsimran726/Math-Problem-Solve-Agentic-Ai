import json
import logging
import re
import sys
import io
import os
from datetime import datetime

from langchain_core.tools import Tool
from states import AgentInput, ParseAgentOutput, VerifyAgentOutput, ExplainAgentOutput, SolveAgentOutput
from ocrmodels import extract_math_from_image
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────
logger = logging.getLogger("math_solver")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)


def _ts() -> str:
    """Short ISO timestamp for log entries stored in state."""
    return datetime.now().strftime("%H:%M:%S")


# shared conversation memory across agents
shared_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


def get_retreival_context(query):
    try:
        vector_store = FAISS.load_local("faiss_index", embedding)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        context = retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in context])
        return context_text
    except Exception as e: 
        logger.error(f"Error getting retreival context: {e}")
        return f"I don't find the content that you want"
retrieval_tool = Tool(
    name="retrival_tool",
    description="Essential for Math Material, Formulas, concept understanding..",
    func=get_retreival_context 
)
tools = [retrieval_tool]


# ── Helpers ────────────────────────────────────────────────────────────

def safe_parse_json(item):
    """Robust JSON parser: handles dicts, strings, and triple-quoted JSON."""
    logger.debug("safe_parse_json received type=%s", type(item).__name__)
    try:
        if isinstance(item, dict):
            return item
        if isinstance(item, list):
            return item
        if isinstance(item, str):
            s = item.strip()
            if s.startswith("'''json") and s.endswith("'''"):
                s = s[7:-3].strip()
                return json.loads(s)
            if s.startswith("```json") and s.endswith("```"):
                s = s[7:-3].strip()
                return json.loads(s)
            if s.startswith("'") and s.endswith("'"):
                s = s[1:-1]
                return json.loads(s)
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s
        return item
    except Exception as e:
        logger.warning("safe_parse_json failed: %s — returning raw", e)
        return item


def python_calculator_tool(llm_response: str) -> dict:
    """Extract ```python blocks from LLM output, run them, and return result."""
    match = re.search(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
    if not match:
        return {"status": 404, "message": "No valid Python code found. Wrap code in ```python blocks."}

    code_to_run = match.group(1)
    logger.info("python_calculator_tool executing %d chars of code", len(code_to_run))

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code_to_run, {})
        result = redirected_output.getvalue().strip()
        if not result:
            return {"status": 404, "message": "Code ran but produced no output. Did you forget print()?"}
        return {"status": 200, "message": f"Execution Result:\n{result}"}
    except Exception as e:
        return {"status": 300, "message": f"Execution Failed:\n{str(e)}"}
    finally:
        sys.stdout = old_stdout


# ── Agents ─────────────────────────────────────────────────────────────

class parser_agent:
    @staticmethod
    def agent_parser(agent_input, parse_output) -> dict:
        """Parse user input into a structured math problem. Returns dict with status & parsed output."""
        log_lines = []
        try:
            log_lines.append(f"[{_ts()}] 🔍 Parser Agent started")
            logger.info("Parser Agent started — text=%s, has_image=%s",
                        bool(getattr(agent_input, 'text_input', None)),
                        bool(getattr(agent_input, 'image_input', None)))

            if getattr(agent_input, 'image_input', None):
                log_lines.append(f"[{_ts()}] 📷 Extracting math from image via OCR")
                logger.info("Running OCR on uploaded image")
                image_text = extract_math_from_image(agent_input.image_input)
                parser_input = f"<QUESTION EXTRACT FROM IMAGE>{image_text}\n<USER QUERY>{agent_input.text_input}"
                log_lines.append(f"[{_ts()}] ✅ OCR complete — extracted {len(str(image_text))} chars")
            else:
                parser_input = f"<USER QUERY>{agent_input.text_input}"

            system_prompt = '''
            <Persona>
            You are the ELITE Math Teacher of IIT DELHI, Where you have experience to analyse the Math problem and generate the Easy to understand Statement Problem.
            </Persona>

            <Tone_Style>
            - Keep your tone High Pitch, Calm but like Genius Mindset.
            - use first person as yourself where you explain the Problem Statement in easy language.
            </Tone_Style>
            You can use the tools also retrieval_tool
            <Workflow>
            -> Understand the user given Question, User thought after that generate a High Quality Problem Statement that you can show user and ask it that you want me to solve?.. when you have low confidence score <60. otherwise don't ask user just show problem statement you generate.
            </Workflow>
            '{input}'
            <Output>
            - Provide in the given below JSON DICT Format (`need_clarification` MUST be a boolean true or false):
            "{{
                "topic": "Put here Topic You identified",
                "problem_text": "put here problem statement",
                "constraints": ["Put here constraints like x>0"],
                "variable": ["Put here all variables in this list"],
                "need_clarification": true or false
            }}"
            </Output>
            '''

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                
                ("assistant", "{agent_scratchpad}")
            ])

            llm = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("openai_api_key"))
            executor = AgentExecutor(
                agent=create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt_template),
                tools=tools,
                verbose=True,
                memory=shared_memory,
            )

            log_lines.append(f"[{_ts()}] 🤖 Calling LLM for parsing")
            logger.info("Invoking parser LLM")
            response = executor.invoke({"input": parser_input})
            raw_output = response['output']
            logger.debug("Parser raw output: %s", raw_output[:200])

            parsed = safe_parse_json(raw_output)
            if isinstance(parsed, dict):
                parse_output.topic = parsed.get('topic')
                parse_output.problem_text = parsed.get('problem_text')
                parse_output.constraints = parsed.get('constraints') or parsed.get('constriants')
                parse_output.variables = parsed.get('variables') or parsed.get('variable')
                nc = parsed.get('need_clarification', False)
                parse_output.need_clarification = nc if isinstance(nc, bool) else str(nc).lower() == 'true'
            else:
                parse_output.problem_text = str(parsed)
                parse_output.need_clarification = True

            status = 200 if not parse_output.need_clarification else 400
            log_lines.append(f"[{_ts()}] {'⚠️ Clarification needed' if parse_output.need_clarification else '✅ Parsed successfully'} — topic={getattr(parse_output, 'topic', '?')}")
            logger.info("Parser result: need_clarification=%s, topic=%s",
                        parse_output.need_clarification, getattr(parse_output, 'topic', None))
            return {"status": status, "message": parse_output, "logs": log_lines}

        except Exception as e:
            logger.exception("Parser Agent failed")
            log_lines.append(f"[{_ts()}] ❌ Parser Agent FAILED: {e}")
            return {"status": 300, "message": f"FAILED IN AGENT PARSER DUE TO {e}", "logs": log_lines}


class intent_router_agent:
    @staticmethod
    def intent_agent(query: dict):
        try:
            pass
        except Exception as e:
            return {"status": 300, "message": f"FAILED IN INTENT AGENT DUE TO {e}"}


class solver_agent:
    @staticmethod
    def solve_agent(solve_agent_out, parse_agent_out) -> dict:
        """Solve the parsed math problem with Python code execution."""
        log_lines = []
        try:
            log_lines.append(f"[{_ts()}] 🧮 Solver Agent started — topic={getattr(parse_agent_out, 'topic', '?')}")
            logger.info("Solver Agent started for topic=%s", getattr(parse_agent_out, 'topic', None))

            system_prompt = '''
You are an elite Mathematical Solver Agent designed to solve JEE-level problems in Algebra, Calculus, Probability, and Linear Algebra. 
'{input}'
You MUST NOT perform complex calculations, algebraic manipulations, or matrix operations in your head. You will hallucinate and fail. 
Instead, you must write Python code using libraries like `sympy`, `numpy`, `scipy`, or `math` to compute the exact answer.

YOUR WORKFLOW:
1. **Understand:** Read the structured problem and constraints.
2. **Plan:** Write out a brief, step-by-step mathematical strategy.
3. **Execute:** Write Python code to execute your plan. 
   - You MUST wrap your code in exact ```python ... ``` blocks.
   - You MUST use `print()` statements to output the exact variables or results you need to see.
4. **Evaluate:** I will execute your code and return the console output to you if any error comes. 
   - If the output contains an error, analyze the error, rewrite the corrected code, and try again.

RULES:
- Do not ask the user for permission. Just write the code.
- Ensure all variables in `sympy` are properly defined (e.g., `x = sp.Symbol('x')`).
- If solving a matrix problem, use `numpy` or `sympy.Matrix`.
            '''
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            llm = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("openai_api_key"))
            executor = AgentExecutor(
                agent=create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt_template),
                tools=tools,
                verbose=True,
                memory=shared_memory,
            )

            query = f"<Problem_statement>{parse_agent_out.problem_text} and Topic: {parse_agent_out.topic}"
            log_lines.append(f"[{_ts()}] 🤖 Calling Solver LLM")
            logger.info("Invoking solver LLM")
            response = executor.invoke({"input": query})

            llm_output = response.get("output", "")
            math_solution = python_calculator_tool(llm_output)
            logger.info("Calculator result status=%s", math_solution.get("status"))

            status_code = math_solution.get("status")
            if status_code in (404, 300):
                error = math_solution.get("message")
                log_lines.append(f"[{_ts()}] ⚠️ Code execution error — retrying")
                logger.warning("Solver code error, feeding back: %s", error[:100])
                executor.invoke({"input": f"RECEIVED THE ERRORS:- {error}"})

            solve_agent_out.answer = math_solution
            solve_agent_out.solution = llm_output

            final_status = 200 if status_code == 200 else 300
            log_lines.append(f"[{_ts()}] {'✅' if final_status == 200 else '⚠️'} Solver completed (status={final_status})")
            return {"status": final_status, "message": solve_agent_out, "logs": log_lines}

        except Exception as e:
            logger.exception("Solver Agent failed")
            log_lines.append(f"[{_ts()}] ❌ Solver Agent FAILED: {e}")
            return {"status": 300, "message": f"FAILED IN SOLVE AGENT DUE TO {e}", "logs": log_lines}


class verifier_agent:
    @staticmethod
    def verify_agent(solve_agent_out, verify_out, parse_agent_out) -> dict:
        """Verify the solution's correctness with a confidence score."""
        log_lines = []
        try:
            log_lines.append(f"[{_ts()}] 🔎 Verifier Agent started")
            logger.info("Verifier Agent started")

            system_prompt = '''
            <persona>
            You are an elite Mathematical Teacher of Harvard university, Experience to Verify the Solutions of JEE-level problems in Algebra, Calculus, Probability, and Linear Algebra. 
            </persona>
            '{input}'
            <context>
            You will receive the Problem Statement, Solution (Python Script, Answer) and need to verify the solution carefully.
            </context>
            <Workflow>
            -> Understand the Problem Statement carefully by breakdown into small parts to understand more. 
            -> Analyse the Solution of Problem Statement which is Python Script & Output of Python Script. 
            -> Prepare your answer and combine with given solution. return the Confidence_score out of 100, how much the answer accurate. 
            </Workflow>

            <Output>
            -> Return in the following JSON Dict format:
                "{{
                "confidence_score": <number>,
                "message": "Put here your thoughts in details about Problem Statement and Solution"
                }}"
            </Output>
            '''
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            llm = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("openai_api_key"))
            executor = AgentExecutor(
                agent=create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt_template),
                tools=tools,
                verbose=True,
                memory=shared_memory,
            )

            query = f"<Problem_statement>{parse_agent_out.problem_text} and TOPIC: {parse_agent_out.topic}<solution>{solve_agent_out.solution}<Answer>{solve_agent_out.answer}"
            log_lines.append(f"[{_ts()}] 🤖 Calling Verifier LLM")
            logger.info("Invoking verifier LLM")
            response = executor.invoke({"input": query})

            parsed = safe_parse_json(response.get('output'))
            if isinstance(parsed, dict):
                verify_out.confidence_score = parsed.get('confidence_score')
                verify_out.message = parsed.get('message')
            else:
                verify_out.confidence_score = None
                verify_out.message = str(parsed)

            score = verify_out.confidence_score
            log_lines.append(f"[{_ts()}] {'✅' if score and score > 60 else '⚠️'} Verification score={score}")
            logger.info("Verifier result: confidence_score=%s", score)
            return {"status": 200, "message": verify_out, "logs": log_lines}

        except Exception as e:
            logger.exception("Verifier Agent failed")
            log_lines.append(f"[{_ts()}] ❌ Verifier Agent FAILED: {e}")
            return {"status": 300, "message": f"FAILED IN VERIFY AGENT DUE TO {e}", "logs": log_lines}


class explanation_agent:
    @staticmethod
    def explain_agent(explain_out, verify_out, parse_agent_out, solve_agent_out) -> dict:
        """Generate a clear, step-by-step explanation of the solution."""
        log_lines = []
        try:
            log_lines.append(f"[{_ts()}] 📖 Explanation Agent started")
            logger.info("Explanation Agent started")

            system_prompt = """You are an elite Mathematical Teacher at Harvard University. You explain JEE-level math solutions in a clear, step-by-step manner that any student can follow.

FORMATTING RULES (VERY IMPORTANT):
- Write ALL mathematical expressions using LaTeX notation:
  - Inline math: $x^2 + 3x + 2 = 0$
  - Display/block math: $$\\int_0^1 x^2 \\, dx = \\frac{{1}}{{3}}$$
- Use Markdown headers (##, ###) to structure your explanation
- Use numbered lists for steps
- Use **bold** for key terms and concepts
- Use code blocks for any Python/computational steps

You MUST respond with ONLY a valid JSON object (no markdown fences, no extra text) in this exact format:
{{"problem_statement": "## Problem Statement\\n\\nExplain the problem in Markdown with LaTeX math like $ax^2+bx+c=0$. Break it into 2-3 clear steps with formulas.", "solution": "## Step-by-Step Solution\\n\\n### Step 1: ...\\n\\nUse LaTeX for every formula: $x = \\\\frac{{-b \\\\pm \\\\sqrt{{b^2-4ac}}}}{{2a}}$\\n\\n### Step 2: ...\\n\\nShow the final answer clearly with $$\\\\boxed{{answer}}$$", "context": "## Key Takeaway\\n\\nA short paragraph explaining the concept, when to use this method, and any important formulas to remember like $formula$."}}

Respond with ONLY the JSON. No other text."""

            llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv("openai_api_key"), temperature=0.8)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            query = (
                f"<Problem_statement>{parse_agent_out.problem_text} TOPIC: {parse_agent_out.topic}\n"
                f"<Solution>{solve_agent_out.solution}"
                f"<Verify_Solution_Score>{verify_out.confidence_score} , MESSAGE:- {verify_out.message}"
            )
            executor = AgentExecutor(
                agent=create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt_template),
                tools=tools,
                verbose=True,
                memory=shared_memory,
            )
            log_lines.append(f"[{_ts()}] 🤖 Calling Explanation LLM")
            logger.info("Invoking explanation LLM")
            response = executor.invoke({"input": query})
            raw_output = response.get('output')

            parsed = safe_parse_json(raw_output)
            if isinstance(parsed, dict):
                explain_out.problem_statement = parsed.get('problem_statement', '')
                explain_out.solution = parsed.get('solution', '')
                explain_out.context = parsed.get('context', '')
            else:
                explain_out.problem_statement = str(parsed)
                explain_out.solution = ''
                explain_out.context = ''

            log_lines.append(f"[{_ts()}] ✅ Explanation generated successfully")
            logger.info("Explanation Agent completed")
            return {"status": 200, "message": explain_out, "logs": log_lines}

        except Exception as e:
            logger.exception("Explanation Agent failed")
            log_lines.append(f"[{_ts()}] ❌ Explanation Agent FAILED: {e}")
            return {"status": 300, "message": f"FAILED IN EXPLANATION AGENT DUE TO {e}", "logs": log_lines}