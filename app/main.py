import json
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient, DESCENDING, ASCENDING
from bson import ObjectId
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Any

load_dotenv()

app = FastAPI()

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["response-to-meeting"]

# Collections
email_collection = db["emails"]
cases_collection = db["cases"]
categories_collection = db["categories"]
user_needs_collection = db["user_needs"]
instruction_templates_collection = db["instruction_templates"]

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MongoBaseModel(BaseModel):
    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

class Category(MongoBaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_custom: bool = False

class UserNeed(MongoBaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    category_id: PyObjectId
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_custom: bool = False

class InstructionTemplate(MongoBaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    category_id: PyObjectId
    user_need_id: PyObjectId
    template: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Case(MongoBaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    email_id: PyObjectId
    category_id: PyObjectId
    user_need_id: PyObjectId
    instruction_template_id: PyObjectId
    confidence_score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CaseDetails(MongoBaseModel):
    case: Case
    category: Category
    user_need: UserNeed
    instruction: InstructionTemplate

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

def get_or_create_category(name: str, description: str = "") -> Category:
    existing_category = categories_collection.find_one({"name": name})
    if existing_category:
        return Category(**existing_category)
    
    new_category = Category(name=name, description=description, is_custom=True)
    result = categories_collection.insert_one(new_category.dict(exclude={"id"}, by_alias=True))
    new_category.id = result.inserted_id
    return new_category

def get_or_create_user_need(category_id: PyObjectId, name: str, description: str = "") -> UserNeed:
    existing_user_need = user_needs_collection.find_one({"category_id": category_id, "name": name})
    if existing_user_need:
        return UserNeed(**existing_user_need)
    
    new_user_need = UserNeed(category_id=category_id, name=name, description=description, is_custom=True)
    result = user_needs_collection.insert_one(new_user_need.dict(exclude={"id"}, by_alias=True))
    new_user_need.id = result.inserted_id
    return new_user_need

def get_or_create_instruction_template(category_id: PyObjectId, user_need_id: PyObjectId) -> Optional[InstructionTemplate]:
    existing_template = instruction_templates_collection.find_one({
        "category_id": category_id,
        "user_need_id": user_need_id
    })
    if existing_template:
        return InstructionTemplate(**existing_template)
    
    category = categories_collection.find_one({"_id": category_id})
    user_need = user_needs_collection.find_one({"_id": user_need_id})
    
    if not category or not user_need:
        print(f"Error: Category or User Need not found. Category ID: {category_id}, User Need ID: {user_need_id}")
        return None
    
    instruction = generate_instruction(category["name"], user_need["name"])
    
    if not instruction:
        print("Error: Failed to generate instruction")
        return None
    
    new_template = InstructionTemplate(
        category_id=category_id,
        user_need_id=user_need_id,
        template=instruction.instruction
    )
    result = instruction_templates_collection.insert_one(new_template.dict(exclude={"id"}, by_alias=True))
    new_template.id = result.inserted_id
    return new_template

def analyze_email(sent_text: str, reply_text: str) -> ClassificationResult:
    categories = get_categories()
    user_needs = get_user_needs()
    
    categories_str = "\n".join([f"- {cat.name}" for cat in categories])
    user_needs_str = "\n".join([f"- {need.name}" for need in user_needs])
    
    prompt = f"""
    Analyze the following email exchange and provide:

    1. The most appropriate generic category for the reply. If no existing category fits, suggest a new one.
    2. The specific generic user need or intent behind the reply. If no existing user need fits, suggest a new one.

    Do not include any personally identifiable information in your classification.

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
        print(f"Error processing LLM response: {str(e)}")
        return None

def generate_instruction(category: str, user_need: str) -> InstructionResult:
    prompt = f"""
    Generate a generic instruction for creating an appropriate follow-up email based on the following category and user need:

    Category: {category}
    User Need: {user_need}

    Provide a generic and contextual instruction for crafting a follow-up email.
    Do not include any specific details or personally identifiable information.
    The instruction should be applicable to any email within this category and user need.
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates email response instructions."},
                {"role": "user", "content": prompt}
            ],
            response_format=InstructionResult
        )

        result = completion.choices[0].message.parsed
        return result
    except Exception as e:
        print(f"Error generating instruction: {str(e)}")
        return None

def convert_to_serializable(data: Any) -> Any:
    """
    Recursively converts non-serializable data types to JSON-compliant formats.
    """
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, float):
        if data == float('inf') or data == float('-inf') or data != data:  # NaN or Infinity
            return str(data)
        return data
    return data

def get_case_by_email_id(email_id):
    try:
        email_object_id = ObjectId(email_id)
        case = cases_collection.find_one({"email_id": email_object_id})
        
        if not case:
            print(f"No case found for email_id: {email_id}")
            return None
        
        print("Case found:")
        print(case)
        return case
    except Exception as e:
        print(f"Error retrieving case: {str(e)}")
        return None

def get_case_details(case):
    try:
        category = categories_collection.find_one({"_id": case["category_id"]})
        user_need = user_needs_collection.find_one({"_id": case["user_need_id"]})
        instruction = instruction_templates_collection.find_one({"_id": case["instruction_template_id"]})
        
        if not all([category, user_need, instruction]):
            print("Some case details are missing")
            return None
        
        case_details = CaseDetails(
            case=Case(**case),
            category=Category(**category),
            user_need=UserNeed(**user_need),
            instruction=InstructionTemplate(**instruction)
        )
        
        print("Case details found:")
        print(case_details)
        return case_details
    except Exception as e:
        print(f"Error retrieving case details: {str(e)}")
        return None

@app.get("/case-details-by-email/{email_id}", response_model=CaseDetails)
async def get_case_details_by_email(email_id: str):
    case = get_case_by_email_id(email_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case found for email_id: {email_id}")
    
    case_details = get_case_details(case)
    if not case_details:
        raise HTTPException(status_code=404, detail=f"Case details not found for case_id: {case['_id']}")
    
    return case_details

@app.post("/classify-email/", response_model=Case)
async def classify_email_endpoint(email_data: EmailData):
    classification = analyze_email(email_data.sent_message_text, email_data.reply_message_text)
    if not classification:
        raise HTTPException(status_code=500, detail="Failed to classify email")
    
    category = get_or_create_category(classification.category)
    user_need = get_or_create_user_need(category.id, classification.user_need)
    
    if not category or not user_need:
        raise HTTPException(status_code=500, detail="Failed to create category or user need")
    
    instruction_template = get_or_create_instruction_template(category.id, user_need.id)
    if not instruction_template:
        raise HTTPException(status_code=500, detail="Failed to create instruction template")
    
    new_case = Case(
        email_id=ObjectId(),  # Generate a new ObjectId for the email
        category_id=category.id,
        user_need_id=user_need.id,
        instruction_template_id=instruction_template.id,
        confidence_score=classification.confidence_score
    )
    
    result = cases_collection.insert_one(new_case.dict(exclude={"id"}, by_alias=True))
    new_case.id = result.inserted_id
    
    return new_case


@app.get("/fetch-emails", tags=["Emails"])
async def fetch_emails(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Number of emails per page"),
    sort_by: str = Query("event_timestamp", description="Field to sort by"),
    order: str = Query("desc", description="Sort order (asc or desc)")
):
    """
    Fetch emails from the email data collection with pagination and sorting.
    """
    skip = (page - 1) * per_page
    sort_direction = DESCENDING if order.lower() == "desc" else ASCENDING

    try:
        total_emails = email_collection.count_documents({})
        emails_cursor = email_collection.find({}) \
            .sort(sort_by, sort_direction) \
            .skip(skip) \
            .limit(per_page)

        emails = [convert_to_serializable(email) for email in emails_cursor]

        return JSONResponse(
            status_code=200,
            content={
                "total": total_emails,
                "page": page,
                "per_page": per_page,
                "emails": emails
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching emails: {str(e)}")



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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")