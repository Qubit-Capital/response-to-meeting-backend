import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["response-to-meeting"]  # Use the existing database name

# Existing collection
email_collection = db["emails"]

# New collections to be created
categories_collection = db["categories"]
user_needs_collection = db["user_needs"]
instruction_templates_collection = db["instruction_templates"]
cases_collection = db["cases"]

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

class Category(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_custom: bool = False

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True

class UserNeed(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    category_id: PyObjectId
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_custom: bool = False

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True
        

class InstructionTemplate(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    category_id: PyObjectId
    user_need_id: PyObjectId
    template: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True

class Case(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    email_id: PyObjectId
    category_id: PyObjectId
    user_need_id: PyObjectId
    instruction_template_id: PyObjectId
    confidence_score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True


class ClassificationResult(BaseModel):
    category: str
    user_need: str
    confidence_score: float

class InstructionResult(BaseModel):
    instruction: str

def get_categories() -> List[Category]:
    return [Category(**{**cat, '_id': str(cat['_id'])}) for cat in categories_collection.find()]

def get_user_needs() -> List[UserNeed]:
    return [UserNeed(**{**need, '_id': str(need['_id'])}) for need in user_needs_collection.find()]

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
    3. A confidence score between 0 and 1 for your classification.

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

def process_email(email_data):
    sent_text = email_data['sent_message_text']
    reply_text = email_data['reply_message_text']
    
    classification = analyze_email(sent_text, reply_text)
    if not classification:
        print(f"Failed to classify email {email_data['_id']}")
        return
    
    category = get_or_create_category(classification.category)
    user_need = get_or_create_user_need(category.id, classification.user_need)
    print(category)
    print(user_need)
    if not category or not user_need:
        print(f"Failed to create category or user need for email {email_data['_id']}")
        return
    
    instruction_template = get_or_create_instruction_template(category.id, user_need.id)
    if not instruction_template:
        print(f"Failed to create instruction template for email {email_data['_id']}")
        return
    
    new_case = Case(
        email_id=email_data['_id'],
        category_id=category.id,
        user_need_id=user_need.id,
        instruction_template_id=instruction_template.id,
        confidence_score=classification.confidence_score
    )
    print(new_case)
    result = cases_collection.insert_one(new_case.dict(exclude={"id"}, by_alias=True))
    print(f"Processed email {email_data['_id']}, created case with _id: {result.inserted_id}")

def init_db():
    # Check if new collections exist, create them if they don't
    if "categories" not in db.list_collection_names():
        db.create_collection("categories")
    if "user_needs" not in db.list_collection_names():
        db.create_collection("user_needs")
    if "cases" not in db.list_collection_names():
        db.create_collection("cases")

    # Initialize categories if the collection is empty
    if categories_collection.count_documents({}) == 0:
        initial_categories = [
            Category(name="Not interested", description="Not interested in any type of service/collaboration"),
            Category(name="Out of office", description="Out of office, please contact later"),
            Category(name="Schedule call", description="Would like to schedule a call"),
            Category(name="Request information", description="Request information about services or company"),
            Category(name="Open to explore", description="Open to explore potential partnerships/services"),
        ]
        categories_collection.insert_many([cat.dict(by_alias=True, exclude={"id"}) for cat in initial_categories])
        print("Initialized categories collection")

    # Initialize user needs if the collection is empty
    if user_needs_collection.count_documents({}) == 0:
        initial_user_needs = [
            UserNeed(category_id=categories_collection.find_one({"name": "Not interested"})["_id"], name="Polite rejection", description="Politely reject the offer"),
            UserNeed(category_id=categories_collection.find_one({"name": "Out of office"})["_id"], name="Provide alternative contact", description="Provide alternative contact information"),
            UserNeed(category_id=categories_collection.find_one({"name": "Schedule call"})["_id"], name="Propose time slots", description="Propose available time slots for a call"),
            UserNeed(category_id=categories_collection.find_one({"name": "Request information"})["_id"], name="Specific service details", description="Request details about specific services"),
            UserNeed(category_id=categories_collection.find_one({"name": "Open to explore"})["_id"], name="Further discussion", description="Express interest in further discussion"),
        ]
        user_needs_collection.insert_many([need.dict(by_alias=True, exclude={"id"}) for need in initial_user_needs])
        print("Initialized user needs collection")

    # Print collection statistics
    print(f"Total categories: {categories_collection.count_documents({})}")
    print(f"Total user needs: {user_needs_collection.count_documents({})}")
    print(f"Total cases: {cases_collection.count_documents({})}")

    # Process only the first 5 emails in the existing collection
    total_emails = min(100, email_collection.count_documents({}))
    processed_emails = 0

    for email in email_collection.find().limit(100):
        process_email(email)
        processed_emails += 1
        print(f"Processed {processed_emails}/{total_emails} emails")

    print(f"Total email exchanges processed: {processed_emails}")

if __name__ == "__main__":
    init_db()