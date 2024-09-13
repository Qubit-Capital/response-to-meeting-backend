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
    from typing import List, Dict
    from fastapi import APIRouter, HTTPException, Query
    from pymongo import ASCENDING, DESCENDING
    from datetime import datetime
    from typing import Any, Dict, List
    from pydantic import BaseModel, Field
    from datetime import datetime
    from typing import List



    load_dotenv()


    app = FastAPI()


    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # MongoDB connection
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))
    db = mongo_client["response-to-meeting"]


    # Collections
    email_collection = db["emails"]
    cases_collection = db["cases"]
    categories_collection = db["categories"]
    user_needs_collection = db["user_needs"]
    instruction_templates_collection = db["instruction_templates"]



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
            # Handle special float values and out-of-range issues
            if data == float('inf') or data == float('-inf') or data != data:  # NaN or Infinity
                return str(data)
            return data
        return data

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

            emails = []
            for email in emails_cursor:  # Synchronous iteration
                email = convert_to_serializable(email)
                emails.append(email)

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
        


    @app.get("/fetch-from-email-id/{email_id}", tags=["Emails"])
    async def fetch_email_by_id(email_id: str):
        """
        Fetch an email from the email data collection by its ID (from_email field).
        """
        try:
            email = email_collection.find_one({"from_email": email_id})
            
            if email is None:
                raise HTTPException(status_code=404, detail=f"Email with ID {email_id} not found")
            
            serialized_email = convert_to_serializable(email)
            
            return JSONResponse(
                status_code=200,
                content=serialized_email
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the email: {str(e)}")
        
        
    @app.get("/fetch-to-email-id/{to_email}", tags=["Emails"])
    async def fetch_email_by_to_email(to_email: str):
        """
        Fetch an email from the email data collection by its to_email field.
        """
        try:
            email = email_collection.find_one({"to_email": to_email})
            
            if email is None:
                raise HTTPException(status_code=404, detail=f"Email with to_email {to_email} not found")
            
            serialized_email = convert_to_serializable(email)
            
            return JSONResponse(
                status_code=200,
                content=serialized_email
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the email: {str(e)}")
        

    @app.patch("/update-email-status/{doc_id}", tags=["Emails"])
    async def update_email_status(doc_id: str, status: str = Query(..., description="New status for the email")):
        """
        Update the status of an email in the email data collection by its ID (_id field of MongoDB document).
        The new status is provided as a query parameter.
        """
        try:

            # Convert string ID to ObjectId
            object_id = ObjectId(doc_id)

            # Update the document
            result = email_collection.update_one(
                {"_id": object_id},
                {"$set": {"status": status}}
            )

            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail=f"Email with ID {doc_id} not found")

            if result.modified_count == 0:
                return JSONResponse(
                    status_code=200,
                    content={"message": "Email status unchanged (already set to the requested status)"}
                )

            return JSONResponse(
                status_code=200,
                content={"message": f"Email status updated to '{status}'"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while updating the email status: {str(e)}")
        


    @app.get("/fetch-email-by-docId/{doc_id}", tags=["Emails"])
    async def fetch_email_by_doc_id(doc_id: str):
        """
        Fetch an email from the email data collection by its document ID (_id field of MongoDB document).
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(doc_id)
            
            # Find the document
            email = email_collection.find_one({"_id": object_id})
            
            if email is None:
                raise HTTPException(status_code=404, detail=f"Email with ID {doc_id} not found")
            
            serialized_email = convert_to_serializable(email)
            
            return JSONResponse(
                status_code=200,
                content=serialized_email
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the email: {str(e)}")



    class UserNeedResponse(BaseModel):
        id: str
        name: str
        description: str

    class CategoryResponse(BaseModel):
        id: str
        name: str
        description: str


    class CaseInstructionResponse(BaseModel):
        template: str

    class EmailClassificationResponse(BaseModel):
        id: str
        email_id: str
        user_need: UserNeedResponse
        category: CategoryResponse
        instruction: CaseInstructionResponse
        confidence_score: float
        created_at: str


    class CaseInstructionResult(BaseModel):
        id: str
        instruction: str
        

    @app.get("/cases", response_model=List[EmailClassificationResponse], tags=["Email Classifications"])
    async def fetch_email_classifications(
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
        sort_by: str = Query("created_at", regex="^(created_at|confidence_score)$"),
        sort_order: str = Query("desc", regex="^(asc|desc)$")
    ):
        """
        Fetch email classifications(cases) from the cases collection.
        """
        try:
            # Determine sort direction
            sort_direction = DESCENDING if sort_order == "desc" else ASCENDING

            # Fetch email classifications
            cursor = db["cases"].find().sort(sort_by, sort_direction).skip(skip).limit(limit)
            
            # Convert cursor to list
            documents = list(cursor)
            
            classifications = []
            for doc in documents:
                try:
                    user_need = db["user_needs"].find_one({"_id": ObjectId(doc["user_need_id"])})
                    category = db["categories"].find_one({"_id": ObjectId(doc["category_id"])})
                    instruction = db["instruction_templates"].find_one({"_id": ObjectId(doc["instruction_template_id"])})
                    
                    # Handle potential null values in the documents
                    user_need_response = UserNeedResponse(
                        id=str(user_need["_id"]) if user_need else "", 
                        name=user_need["name"] if user_need else "",
                        description=user_need["description"] if user_need else ""
                    )
                    category_response = CategoryResponse(
                        id=str(category["_id"]) if category else "",
                        name=category["name"] if category else "",
                        description=category["description"] if category else ""
                    )


                    instruction_template_response = CaseInstructionResponse(
                        template=instruction["template"] if instruction else ""
                    )

                    # Create the email classification response
                    classification = EmailClassificationResponse(
                        id = str(doc["_id"]),
                        email_id= str(doc["email_id"]),
                        user_need=user_need_response,
                        category=category_response,
                        instruction=instruction_template_response,
                        confidence_score=doc["confidence_score"],
                        created_at=doc["created_at"].isoformat(),
                    )
                    classifications.append(classification)
                except Exception as doc_error:
                    # Log the error and continue with the next document
                    print(f"Error processing document: {str(doc_error)}")
                    continue
            
            return classifications
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching email classifications: {str(e)}")
        


    @app.get("/cases/{case_id}", response_model=EmailClassificationResponse, tags=["Email Classifications"])
    async def fetch_single_case(case_id: str):
        """
        Fetch a single email classification(case) from the cases collection by its ID.
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(case_id)

            # Fetch the case document from the email_data_db
            doc = db["cases"].find_one({"_id": object_id})

            if doc is None:
                raise HTTPException(status_code=404, detail="Case not found")

            user_need = db["user_needs"].find_one({"_id": ObjectId(doc["user_need_id"])})
            category = db["categories"].find_one({"_id": ObjectId(doc["category_id"])})
            instruction = db["instruction_templates"].find_one({"_id": ObjectId(doc["instruction_template_id"])})

            # Handle potential null values in the documents
            user_need_response = UserNeedResponse(
                id=str(user_need["_id"]) if user_need else "",
                name=user_need["name"] if user_need else "",
                description=user_need["description"] if user_need else ""
            )
            category_response = CategoryResponse(
                id=str(category["_id"]) if category else "",
                name=category["name"] if category else "",
                description=category["description"] if category else ""
            )

            instruction_template_response = CaseInstructionResponse(
                template=instruction["template"] if instruction else ""
            )

            # Create the email classification response
            classification = EmailClassificationResponse(
                id = str(doc["_id"]),
                email_id= str(doc["email_id"]),
                user_need=user_need_response,
                category=category_response,
                instruction=instruction_template_response,
                confidence_score=doc["confidence_score"],
                created_at=doc["created_at"].isoformat(),
            )

            return classification
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the case: {str(e)}")





    @app.get("/cases/{case_id}/instruction", response_model=CaseInstructionResult)
    async def get_case_instruction(case_id: str):
        """
        Retrieve the instruction for a specific case by its ID.
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(case_id)

            # Fetch the case document from the email_data_db
            case = db["cases"].find_one({"_id": object_id})

            if case is None:
                raise HTTPException(status_code=404, detail="Case not found")

            # Match instruction_template_id from the case with the _id in instruction_templates
            instruction_template = db["instruction_templates"].find_one({
                "_id": ObjectId(case["instruction_template_id"])
            })

            if instruction_template is None:
                raise HTTPException(status_code=404, detail="Instruction template not found")

            # Return the instruction template
            return CaseInstructionResult(id=str(instruction_template["_id"]),instruction=instruction_template["template"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the case instruction: {str(e)}")


    @app.patch("/cases/{case_id}/new_instruction", response_model=CaseInstructionResult)
    async def update_case_instruction(case_id: str, instruction: str = Query(..., description="The new instruction for the case")):
        """
        Update the instruction for a specific case by its ID.
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(case_id)

            # Fetch the case document from the email_data_db
            case = db["cases"].find_one({"_id": object_id})

            if case is None:
                raise HTTPException(status_code=404, detail="Case not found")

            # Update the instruction template in the instruction_templates collection
            result = db["instruction_templates"].update_one(
                {"_id": ObjectId(case["instruction_template_id"])},
                {"$set": {"template": instruction}}
            )

            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Instruction template not found")

            if result.modified_count == 0:
                raise HTTPException(status_code=304, detail="Instruction not modified")

            # Return the updated instruction
            return CaseInstructionResult(id=str(case["instruction_template_id"]),instruction=instruction)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while updating the case instruction: {str(e)}")
        

        

    @app.patch("/cases/{case_id}/update_email_id", response_model=dict)
    async def update_case_email_id(case_id: str, email_id: str = Query(..., description="The new email ID for the case")):
        """
        Update the email_id for a specific case by its ID.
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(case_id)

            # Fetch the case document from the email_data_db
            case = db["cases"].find_one({"_id": object_id})

            if case is None:
                raise HTTPException(status_code=404, detail="Case not found")

            # Update the email_id in the cases collection
            result = db["cases"].update_one(
                {"_id": object_id},
                {"$set": {"email_id": ObjectId(email_id)}}
            )

            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Case not found")

            if result.modified_count == 0:
                raise HTTPException(status_code=304, detail="Email ID not modified")

            # Return a message indicating the email ID was updated
            return {"message": f"Email ID updated to {email_id} for case {case_id}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while updating the case email ID: {str(e)}")


    class FetchCategory(BaseModel):
        id: str = Field(..., alias="_id")
        name: str

    @app.get("/categories", response_model=List[FetchCategory])
    async def get_categories():
        """
        Fetch all categories from the database.
        """
        try:
            categories = list(db["categories"].find())
            
            # Convert ObjectId to string for each category
            for category in categories:
                category["_id"] = str(category["_id"])
            
            return categories
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching categories: {str(e)}")

    @app.get("/categories/{category_id}", response_model=FetchCategory)
    async def get_category_by_id(category_id: str):
        """
        Fetch a specific category by its ID from the database.
        """
        try:
            category = db["categories"].find_one({"_id": ObjectId(category_id)})
            
            if category is None:
                raise HTTPException(status_code=404, detail="Category not found")
            
            # Convert ObjectId to string
            category["_id"] = str(category["_id"])
            
            return category
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the category: {str(e)}")

    class UserNeedWithCategory(BaseModel):
        id: str = Field(..., alias="_id")
        name: str
        description: str
        created_at: datetime
        updated_at: datetime
        is_custom: bool
        category: FetchCategory

    @app.get("/user-needs", response_model=List[UserNeedWithCategory])
    async def get_user_needs():
        """
        Fetch all user needs from the database, including their associated categories.
        """
        try:
            user_needs = list(db["user_needs"].find())
            
            result = []
            for user_need in user_needs:
                # Fetch the associated category
                category = db["categories"].find_one({"_id": user_need["category_id"]})
                
                # Convert ObjectId to string
                user_need["_id"] = str(user_need["_id"])
                
                # Create UserNeedWithCategory object
                user_need_with_category = UserNeedWithCategory(
                    _id=user_need["_id"],  # Use '_id' instead of 'id'
                    name=user_need["name"],
                    description=user_need["description"],
                    created_at=user_need["created_at"],
                    updated_at=user_need["updated_at"],
                    is_custom=user_need["is_custom"],
                    category=FetchCategory(_id="not found", name="not found") if not category else FetchCategory(_id=str(category["_id"]), name=category["name"])
                )
                result.append(user_need_with_category)
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching user needs: {str(e)}")

    @app.get("/user-needs/{user_need_id}", response_model=UserNeedWithCategory)
    async def get_user_need_by_id(user_need_id: str):
        """
        Fetch a specific user need by its ID from the database, including its associated category.
        """
        try:
            user_need = db["user_needs"].find_one({"_id": ObjectId(user_need_id)})
            
            if user_need is None:
                raise HTTPException(status_code=404, detail="User need not found")
            
            # Fetch the associated category
            category = db["categories"].find_one({"_id": user_need["category_id"]})
            
            # Convert ObjectId to string
            user_need["_id"] = str(user_need["_id"])
            
            # Create UserNeedWithCategory object
            user_need_with_category = UserNeedWithCategory(
                _id=user_need["_id"],
                name=user_need["name"],
                description=user_need["description"],
                created_at=user_need["created_at"],
                updated_at=user_need["updated_at"],
                is_custom=user_need["is_custom"],
                category=FetchCategory(_id="not found", name="not found") if not category else FetchCategory(_id=str(category["_id"]), name=category["name"])
            )
            
            return user_need_with_category
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching the user need: {str(e)}")




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
                    },
                    {
                        "path": "/fetch-emails",
                        "method": "GET",
                        "description": "Fetch emails with pagination and sorting"
                    },
                    {
                        "path": "/categories",
                        "method": "GET",
                        "description": "Fetch all categories"
                    },
                    {
                        "path": "/categories/{category_id}",
                        "method": "GET",
                        "description": "Fetch a specific category by ID"
                    },
                    {
                        "path": "/user-needs",
                        "method": "GET",
                        "description": "Fetch all user needs with their associated categories"
                    },
                    {
                        "path": "/user-needs/{user_need_id}",
                        "method": "GET",
                        "description": "Fetch a specific user need by ID with its associated category"
                    },
                    {
                        "path": "/cases",
                        "method": "GET",
                        "description": "Fetch all cases with pagination and sorting"
                    },
                    {
                        "path": "/cases/{case_id}",
                        "method": "GET",
                        "description": "Fetch a specific case by ID"
                    },
                    {
                        "path": "/cases",
                        "method": "POST",
                        "description": "Create a new case"
                    },
                    {
                        "path": "/cases/{case_id}",
                        "method": "PUT",
                        "description": "Update an existing case"
                    }
                ],
                "documentation": "/docs",
                "openapi_schema": "/openapi.json"
            }
        )



    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0")
