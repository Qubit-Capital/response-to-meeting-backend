import csv
import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, Optional
from dotenv import load_dotenv
import io
from fastapi.responses import Response, JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone

load_dotenv()

app = FastAPI(
    title="Email Classification AI Agent",
    description="An AI-powered email classification and response generation system",
    version="2.0.0"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["email_classification_db"]

categories_collection = db["categories"]
user_needs_collection = db["user_needs"]
email_classifications_collection = db["email_classifications"]

class Category(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_custom: bool = False

    class Config:
        allow_population_by_field_name = True

class UserNeed(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    category_id: str
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_custom: bool = False

    class Config:
        allow_population_by_field_name = True

class EmailClassification(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    email_id: str
    category_id: str
    user_need_id: str
    instruction: str
    confidence_score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback: Optional[str] = None
    is_corrected: bool = False

    class Config:
        allow_population_by_field_name = True

class EmailData(BaseModel):
    sent_message_text: str
    reply_message_text: str

class ClassificationResult(BaseModel):
    category: str
    user_need: str
    confidence_score: float

class InstructionResult(BaseModel):
    instruction: str

def get_categories() -> List[Category]:
    return [Category(**cat) for cat in categories_collection.find()]

def get_user_needs() -> List[UserNeed]:
    return [UserNeed(**need) for need in user_needs_collection.find()]

def create_or_get_category(name: str, description: str = "") -> Category:
    existing_category = categories_collection.find_one({"name": name})
    if existing_category:
        return Category(**existing_category)
    
    new_category = Category(name=name, description=description, is_custom=True)
    result = categories_collection.insert_one(new_category.dict(by_alias=True, exclude_none=True))
    new_category.id = str(result.inserted_id)
    return new_category

def create_or_get_user_need(category_id: str, name: str, description: str = "") -> UserNeed:
    existing_user_need = user_needs_collection.find_one({"category_id": category_id, "name": name})
    if existing_user_need:
        return UserNeed(**existing_user_need)
    
    new_user_need = UserNeed(category_id=category_id, name=name, description=description, is_custom=True)
    result = user_needs_collection.insert_one(new_user_need.dict(by_alias=True, exclude_none=True))
    new_user_need.id = str(result.inserted_id)
    return new_user_need

def analyze_email(sent_text: str, reply_text: str) -> ClassificationResult:
    categories = get_categories()
    user_needs = get_user_needs()
    
    categories_str = "\n".join([f"- {cat.name}" for cat in categories])
    user_needs_str = "\n".join([f"- {need.name}" for need in user_needs])
    
    prompt = f"""
    Analyze the following email exchange and provide:

    1. The most appropriate category for the reply. If no existing category fits, suggest a new one.
    2. The specific user need or intent behind the reply. If no existing user need fits, suggest a new one.
    3. A confidence score between 0 and 1 for your classification.

    Existing Categories:
    {categories_str}

    Existing User Needs:
    {user_needs_str}

    Email exchange:
    Sent message: {sent_text}
    Email reply: {reply_text}

    Provide the information in the required structure.
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes email exchanges."},
                {"role": "user", "content": prompt}
            ],
            response_format=ClassificationResult
        )

        result = completion.choices[0].message.parsed
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")

def generate_instruction(category: str, user_need: str, sent_text: str, reply_text: str) -> InstructionResult:
    prompt = f"""
    Based on the following email exchange and its classification, generate a detailed instruction for creating an appropriate follow-up email:

    Category: {category}
    User Need: {user_need}

    Email exchange:
    Sent message: {sent_text}
    Email reply: {reply_text}

    Provide a specific and contextual instruction for crafting a follow-up email.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates email response instructions."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "generate_instruction",
                "description": "Generate a detailed instruction for email follow-up",
                "parameters": InstructionResult.schema()
            }],
            function_call={"name": "generate_instruction"}
        )

        result = response.choices[0].message.function_call.arguments
        return InstructionResult.parse_raw(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating instruction: {str(e)}")

@app.post("/classify-email/", response_model=EmailClassification)
async def classify_email_endpoint(email_data: EmailData):
    classification = analyze_email(email_data.sent_message_text, email_data.reply_message_text)
    
    category = create_or_get_category(classification.category)
    user_need = create_or_get_user_need(category.id, classification.user_need)
    
    instruction_result = generate_instruction(classification.category, classification.user_need, email_data.sent_message_text, email_data.reply_message_text)
    
    email_classification = EmailClassification(
        email_id=str(ObjectId()),
        category_id=category.id,
        user_need_id=user_need.id,
        instruction=instruction_result.instruction,
        confidence_score=classification.confidence_score
    )
    
    email_classifications_collection.insert_one(email_classification.dict(by_alias=True, exclude_none=True))
    
    return email_classification

@app.post("/process-csv/")
async def process_csv_endpoint(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    content = await file.read()
    csv_data = content.decode('utf-8').splitlines()
    csv_reader = csv.DictReader(csv_data)

    output_rows = []
    fieldnames = csv_reader.fieldnames + ['classification_category', 'user_need', 'follow_up_instruction', 'confidence_score']

    for row in csv_reader:
        sent_text = row['sent_message_text']
        reply_text = row['reply_message_text']

        background_tasks.add_task(process_email, sent_text, reply_text)

        output_rows.append(row)

    output = io.StringIO()
    csv_writer = csv.DictWriter(output, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(output_rows)
    csv_content = output.getvalue()
    output.close()

    output_filename = f"processed_{file.filename}"
    output_path = os.path.join("output", output_filename)
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        f.write(csv_content)

    response = Response(content=csv_content, media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={output_filename}"

    return response

async def process_email(sent_text: str, reply_text: str):
    email_data = EmailData(sent_message_text=sent_text, reply_message_text=reply_text)
    await classify_email_endpoint(email_data)

def init_db():
    # Clear existing data (optional, remove in production)
    categories_collection.delete_many({})
    user_needs_collection.delete_many({})
    

    if categories_collection.count_documents({}) == 0:
        initial_categories = [
            Category(name="Not interested", description="Not interested in any type of service/collaboration"),
            Category(name="Out of office", description="Out of office, please contact later"),
            Category(name="Schedule call", description="Would like to schedule a call"),
            Category(name="Request information", description="Request information about services or company"),
            Category(name="Open to explore", description="Open to explore potential partnerships/services"),
        ]
        categories_collection.insert_many([cat.dict(by_alias=True, exclude_none=True) for cat in initial_categories])

    if user_needs_collection.count_documents({}) == 0:
        initial_user_needs = [
            UserNeed(category_id=str(categories_collection.find_one({"name": "Not interested"})["_id"]), name="Polite rejection", description="Politely reject the offer"),
            UserNeed(category_id=str(categories_collection.find_one({"name": "Out of office"})["_id"]), name="Provide alternative contact", description="Provide alternative contact information"),
            UserNeed(category_id=str(categories_collection.find_one({"name": "Schedule call"})["_id"]), name="Propose time slots", description="Propose available time slots for a call"),
            UserNeed(category_id=str(categories_collection.find_one({"name": "Request information"})["_id"]), name="Specific service details", description="Request details about specific services"),
            UserNeed(category_id=str(categories_collection.find_one({"name": "Open to explore"})["_id"]), name="Further discussion", description="Express interest in further discussion"),
        ]
        user_needs_collection.insert_many([need.dict(by_alias=True, exclude_none=True) for need in initial_user_needs])

@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint that provides information about the API.
    """
    return JSONResponse(
        status_code=200,
        content={
            "name": "Email Classification AI Agent",
            "version": "2.0.0",
            "description": "This API provides email classification and response generation capabilities.",
            "endpoints": [
                {
                    "path": "/",
                    "method": "GET",
                    "description": "Get information about the API"
                },
                {
                    "path": "/classify-email/",
                    "method": "POST",
                    "description": "Classify a single email exchange and generate a response instruction"
                },
                {
                    "path": "/process-csv/",
                    "method": "POST",
                    "description": "Process a CSV file containing multiple email exchanges"
                }
            ],
            "documentation": "/docs",
            "openapi_schema": "/openapi.json"
        }
    )

if __name__ == "__main__":
    init_db()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
