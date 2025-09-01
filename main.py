from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os

# LangChain + Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Basic routes
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/resume")
def download_resume():
    return FileResponse("resume.pdf", filename="Akash_Swain_Resume.pdf")

# ---------------------------
# Chat Assistant (LangChain + Groq)
# ---------------------------
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = None
if GROQ_API_KEY:
    # Smaller, faster model for low latency. Switch to 70B if needed.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

# You can expand this context or load it from files (e.g., resume/pdf parsed)
AKASH_CONTEXT = (
    "Name: Akash Swain\n"
    "Phone: +91 63729919712\n"
    "Email: aswain414@gmail.com\n"
    "GitHub: https://github.com/aswain414\n"
    "LinkedIn: https://www.linkedin.com/in/akash-swain-20645813a/\n"
    "Location: Originally from Bhubaneswar, Odisha; currently based in Pune, India.\n"
    "\n"
    "Current Title: Solutions Developer (ER&D Department)\n"
    "Current Company: Tata Technologies (Client: Tata Motors), Pune\n"
    "Experience: 5+ years in AI/ML engineering and full-stack development.\n"
    "\n"
    "Previous Company: Silicon Techlab Pvt. Ltd.\n"
    "Role: AI/ML Engineer & Full-Stack Developer\n"
    "\n"
    "Education:\n"
    "- B.Tech in Computer Science and Engineering, Centurion University of Technology and Management (2016–2020), CGPA: 8.17.\n"
    "- 12th (Science), CHSE Odisha, 2016.\n"
    "- 10th, BSE Odisha, 2013.\n"
    "\n"
    "Skills:\n"
    "- Programming: Python, PHP, JavaScript\n"
    "- Frameworks: Django, FastAPI, Flask, Laravel, React.js\n"
    "- AI/ML: NLP, Generative AI, LangChain, LangGraph, Transformers, Machine Learning, Deep Learning\n"
    "- Tools: Power BI, Metabase, Streamlit\n"
    "- Databases & APIs: REST APIs, JSON/XML, SQL\n"
    "- Other: Automation, data visualization, analytics\n"
    "\n"
    "Projects:\n"
    "- AutoQuery AI: Built an AI system that converts natural language questions into optimized database queries using LLMs, helping non-technical users fetch data easily.\n"
    "- KATS (Knowledge-Aided Ticketing System): Developed a system to detect duplicate issues in support tickets, reducing redundancy and improving resolution speed.\n"
    "- Handwritten Image to Text: Implemented an OCR-based AI tool to extract accurate text from handwritten notes and scanned images, useful for digitization of records.\n"
    "- LQOS (Line Quality Optimization System): Designed a monitoring and analytics solution to identify and fix quality issues in production lines, improving efficiency.\n"
    "- Audit Management Tool: Created an end-to-end digital audit management platform to plan, track, and report audits, reducing paperwork and manual effort.\n"
    "\n"
    "Achievements:\n"
    "- Delivered AI-powered solutions for automotive engineering at Tata Motors.\n"
    "- Built scalable data analysis pipelines and automation workflows.\n"
    "- Successfully deployed AI/NLP solutions in real-world business environments.\n"
    "\n"
    "Languages: English, Hindi, Odia\n"
    "Interests: Coding challenges, WWE\n"
    "\n"
    "Goal: To design and deliver innovative AI-driven products with Python and Generative AI that solve real-world problems, automate workflows, and create measurable business impact.\n"
)


SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions about Akash Swain. "
    "Base your answers only on the provided context. If something is unknown, say you don't have that information. "
    "Keep answers concise and professional."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS + "\n\nContext:\n{context}"),
    ("human", "{question}"),
])

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server.")

    chain = prompt | llm
    result = chain.invoke({"context": AKASH_CONTEXT, "question": payload.message})
    answer = getattr(result, "content", str(result))
    return {"answer": answer}