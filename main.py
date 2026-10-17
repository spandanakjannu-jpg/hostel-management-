
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError


# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017/"
)
DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "hostel_management"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# MongoDB connection
client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

db = client[DATABASE_NAME]

students_collection = db["students"]
rooms_collection = db["rooms"]
allocations_collection = db["allocations"]
fees_collection = db["fees"]
complaints_collection = db["complaints"]


# Create database indexes when the application starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client.admin.command("ping")

        students_collection.create_index(
            [("email", ASCENDING)],
            unique=True
        )

        rooms_collection.create_index(
            [("room_number", ASCENDING)],
            unique=True
        )

        allocations_collection.create_index(
            [("student_id", ASCENDING)],
            unique=True
        )

    except PyMongoError as error:
        print(f"MongoDB startup error: {error}")
        raise

    yield
    client.close()


app = FastAPI(
    title="Hostel Management System API",
    description="API for managing hostel students, rooms, fees, complaints, and AI chat.",
    version="1.0.0",
    lifespan=lifespan
)


# Allow the React frontend to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Convert MongoDB ObjectIds and datetimes into JSON-compatible values
def serialize(document):
    if isinstance(document, list):
        return [serialize(item) for item in document]

    if isinstance(document, dict):
        result = {}

        for key, value in document.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = serialize(value)

        return result

    return document


def get_object_id(value: str):
    if not ObjectId.is_valid(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid MongoDB ID"
        )

    return ObjectId(value)


def handle_database_error(error):
    print(f"Database error: {error}")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="A database operation failed."
    )


# -------------------- PYDANTIC MODELS --------------------

class Student(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(gt=0, le=120)
    course: str = Field(min_length=1, max_length=100)


class Room(BaseModel):
    room_number: str = Field(min_length=1, max_length=20)
    capacity: int = Field(gt=0, le=100)
    room_type: str = Field(default="Standard", max_length=50)


class Allocation(BaseModel):
    student_id: str
    room_id: str


class Fee(BaseModel):
    student_id: str
    amount: float = Field(gt=0)
    description: str = Field(min_length=1, max_length=200)
    status: str = Field(default="Pending", pattern="^(Pending|Paid)$")


class Complaint(BaseModel):
    student_id: str
    subject: str = Field(min_length=1, max_length=150)
    description: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


# -------------------- HOME --------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to the Hostel Management System API!"
    }


# -------------------- STUDENTS --------------------

@app.post("/students", status_code=status.HTTP_201_CREATED)
def add_student(student: Student):
    student_data = student.model_dump()
    student_data["email"] = str(student.email).lower()
    student_data["created_at"] = datetime.now(timezone.utc)

    try:
        result = students_collection.insert_one(student_data)

        return {
            "message": "Student added successfully!",
            "student": serialize(
                students_collection.find_one(
                    {"_id": result.inserted_id}
                )
            )
        }

    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A student with this email already exists."
        )
    except PyMongoError as error:
        handle_database_error(error)


@app.get("/students")
def get_students():
    try:
        return serialize(list(students_collection.find()))
    except PyMongoError as error:
        handle_database_error(error)


@app.get("/students/{student_id}")
def get_student(student_id: str):
    object_id = get_object_id(student_id)

    try:
        student = students_collection.find_one({"_id": object_id})

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found."
            )

        return serialize(student)
    except PyMongoError as error:
        handle_database_error(error)


@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    object_id = get_object_id(student_id)

    try:
        if allocations_collection.find_one(
            {"student_id": object_id}
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Remove the student's room allocation before deleting."
            )

        result = students_collection.delete_one({"_id": object_id})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found."
            )

        return {"message": "Student deleted successfully."}

    except PyMongoError as error:
        handle_database_error(error)


# -------------------- ROOMS --------------------

@app.post("/rooms", status_code=status.HTTP_201_CREATED)
def add_room(room: Room):
    room_data = room.model_dump()
    room_data["room_number"] = room_data["room_number"].strip()
    room_data["created_at"] = datetime.now(timezone.utc)

    try:
        result = rooms_collection.insert_one(room_data)

        return {
            "message": "Room added successfully!",
            "room": serialize(
                rooms_collection.find_one(
                    {"_id": result.inserted_id}
                )
            )
        }

    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This room number already exists."
        )
    except PyMongoError as error:
        handle_database_error(error)


@app.get("/rooms")
def get_rooms():
    try:
        result = []

        for room in rooms_collection.find():
            allocated = allocations_collection.count_documents(
                {"room_id": room["_id"]}
            )

            room["occupied"] = allocated
            room["available_beds"] = max(
                0, room["capacity"] - allocated
            )

            result.append(room)

        return serialize(result)

    except PyMongoError as error:
        handle_database_error(error)


# -------------------- ROOM ALLOCATIONS --------------------

@app.post("/allocations", status_code=status.HTTP_201_CREATED)
def allocate_room(allocation: Allocation):
    student_object_id = get_object_id(allocation.student_id)
    room_object_id = get_object_id(allocation.room_id)

    try:
        student = students_collection.find_one(
            {"_id": student_object_id}
        )

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found."
            )

        room = rooms_collection.find_one(
            {"_id": room_object_id}
        )

        if not room:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Room not found."
            )

        existing = allocations_collection.find_one(
            {"student_id": student_object_id}
        )

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This student already has a room allocation."
            )

        occupied = allocations_collection.count_documents(
            {"room_id": room_object_id}
        )

        if occupied >= room["capacity"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This room is full."
            )

        allocation_data = {
            "student_id": student_object_id,
            "room_id": room_object_id,
            "allocated_at": datetime.now(timezone.utc)
        }

        result = allocations_collection.insert_one(allocation_data)

        return {
            "message": "Room allocated successfully!",
            "allocation": serialize(
                allocations_collection.find_one(
                    {"_id": result.inserted_id}
                )
            )
        }

    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This student already has a room allocation."
        )
    except PyMongoError as error:
        handle_database_error(error)


@app.get("/allocations")
def get_allocations():
    try:
        return serialize(list(allocations_collection.find()))
    except PyMongoError as error:
        handle_database_error(error)


@app.delete("/allocations/{allocation_id}")
def delete_allocation(allocation_id: str):
    object_id = get_object_id(allocation_id)

    try:
        result = allocations_collection.delete_one(
            {"_id": object_id}
        )

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Allocation not found."
            )

        return {"message": "Room allocation removed successfully."}

    except PyMongoError as error:
        handle_database_error(error)


# -------------------- HOSTEL FEES --------------------

@app.post("/fees", status_code=status.HTTP_201_CREATED)
def add_fee(fee: Fee):
    student_object_id = get_object_id(fee.student_id)

    try:
        if not students_collection.find_one(
            {"_id": student_object_id}
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found."
            )

        fee_data = fee.model_dump()
        fee_data["student_id"] = student_object_id
        fee_data["created_at"] = datetime.now(timezone.utc)

        result = fees_collection.insert_one(fee_data)

        return {
            "message": "Fee record added successfully!",
            "fee": serialize(
                fees_collection.find_one(
                    {"_id": result.inserted_id}
                )
            )
        }

    except PyMongoError as error:
        handle_database_error(error)


@app.get("/fees")
def get_fees():
    try:
        return serialize(list(fees_collection.find()))
    except PyMongoError as error:
        handle_database_error(error)


# -------------------- COMPLAINTS --------------------

@app.post("/complaints", status_code=status.HTTP_201_CREATED)
def add_complaint(complaint: Complaint):
    student_object_id = get_object_id(complaint.student_id)

    try:
        if not students_collection.find_one(
            {"_id": student_object_id}
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found."
            )

        complaint_data = complaint.model_dump()
        complaint_data["student_id"] = student_object_id
        complaint_data["status"] = "Open"
        complaint_data["created_at"] = datetime.now(timezone.utc)

        result = complaints_collection.insert_one(complaint_data)

        return {
            "message": "Complaint submitted successfully!",
            "complaint": serialize(
                complaints_collection.find_one(
                    {"_id": result.inserted_id}
                )
            )
        }

    except PyMongoError as error:
        handle_database_error(error)


@app.get("/complaints")
def get_complaints():
    try:
        return serialize(list(complaints_collection.find()))
    except PyMongoError as error:
        handle_database_error(error)


# -------------------- AI HOSTEL ASSISTANT --------------------

@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini is not configured. Add GEMINI_API_KEY to .env."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request.message,
            config={
                "system_instruction": (
                    "You are a helpful AI assistant for a hostel "
                    "management system. Answer questions about "
                    "student registration, rooms, room allocation, "
                    "fees, and complaints. Do not invent hostel "
                    "rules, fees, availability, or student records. "
                    "You do not have direct access to live hostel "
                    "records. Keep answers clear and beginner-friendly."
                )
            }
        )

        return {"reply": response.text or "No response was generated."}

    except Exception as error:
        print(f"Gemini AI error: {error}")

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini could not respond. Check the API key, model, and connection."
        )
