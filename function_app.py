import azure.functions as func
import logging
import json
import os
from openai import AzureOpenAI, OpenAIError
import pyodbc
from datetime import datetime
import uuid
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure SQL configuration
SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")

# Azure Blob Storage configuration
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
STORAGE_CONTAINER_NAME = os.getenv("STORAGE_CONTAINER_NAME")  # Ensure this container exists

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT]):
    logging.error("Azure OpenAI configuration is missing.")
    raise ValueError("Azure OpenAI configuration is incomplete.")

if not SQL_CONNECTION_STRING:
    logging.error("SQL connection string is missing.")
    raise ValueError("SQL configuration is incomplete.")

if not STORAGE_CONNECTION_STRING:
    logging.error("Storage connection string is missing.")
    raise ValueError("Storage configuration is incomplete.")

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

@app.route(route="processComplaint", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def process_complaint(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing complaint request.")

    if req.method != "POST":
        return func.HttpResponse("Method not allowed. Use POST.", status_code=405)

    try:
        req_body = req.get_json()
        complaint = req_body.get("complaint")
        findings = req_body.get("findings", "")

        if not complaint:
            return func.HttpResponse(
                json.dumps({"error": "Complaint text is required."}),
                mimetype="application/json",
                status_code=400,
                headers={"Access-Control-Allow-Origin": "*"}
            )

        response_prompt = (
            "You are a professional customer service agent. Draft a polite, empathetic, "
            "and concise response to the customer's complaint based on the provided text "
            "and investigation findings."
        )
        category_prompt = (
            "Classify the following complaint into one of these categories: Credit Cards, Channels, Staff, Banking & Savings. "
            "Return only the category name."
        )

        response_messages = [
            {"role": "system", "content": response_prompt},
            {"role": "user", "content": f"Complaint: {complaint}\nFindings: {findings}"}
        ]
        response_result = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=response_messages,
            max_tokens=500,
            temperature=0.7,
            timeout=10
        )
        generated_response = response_result.choices[0].message.content.strip()
        logging.info(f"Generated response length: {len(generated_response)} characters")
        logging.info(f"Generated response: {generated_response}")

        category_messages = [
            {"role": "system", "content": category_prompt},
            {"role": "user", "content": f"Complaint: {complaint}\nFindings: {findings}"}
        ]
        category_result = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=category_messages,
            max_tokens=20,
            temperature=0.3,
            timeout=10
        )
        category = category_result.choices[0].message.content.strip()
        valid_categories = ["Credit Cards", "Channels", "Staff", "Banking & Savings"]
        if category not in valid_categories:
            category = "Banking & Savings"

        result = {"category": category, "response": generated_response}
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200,
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON payload."}),
            mimetype="application/json",
            status_code=400,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except OpenAIError as oai_err:
        logging.error(f"Azure OpenAI error: {str(oai_err)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to generate response due to OpenAI service issue."}),
            mimetype="application/json",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "An error occurred while processing your request."}),
            mimetype="application/json",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.route(route="saveResponse", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def save_response(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Saving response and uploading document to database.")

    if req.method != "POST":
        return func.HttpResponse("Method not allowed. Use POST.", status_code=405)

    try:
        complaint = req.form.get("complaint", "Unknown")
        original_response = req.form.get("originalResponse")
        edited_response = req.form.get("editedResponse")
        original_category = req.form.get("originalCategory")
        edited_category = req.form.get("editedCategory")
        file = req.files.get("document")  # Optional file

        if not all([original_response, edited_response, original_category, edited_category]):
            return func.HttpResponse(
                json.dumps({"error": "All response and category fields are required."}),
                mimetype="application/json",
                status_code=400,
                headers={"Access-Control-Allow-Origin": "*"}
            )

        # Compute IsCorrectCategory: True if categories match, False otherwise
        # Handle None (NULL) explicitly to match SQL behavior
        is_correct_category = (original_category == edited_category) if original_category is not None and edited_category is not None else False

        response_id = str(uuid.uuid4())
        document_url = ""

        if file:
            folder_name = response_id  # Use response_id as folder name
            blob_name = f"{folder_name}/{file.filename}"
            blob_client = blob_service_client.get_blob_client(container=STORAGE_CONTAINER_NAME, blob=blob_name)
            blob_client.upload_blob(file.read(), overwrite=True)
            document_url = blob_client.url
            logging.info(f"Uploaded document to: {document_url}")

        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ComplaintResponses (
                ResponseId, Complaint, OriginalResponse, EditedResponse, 
                OriginalCategory, EditedCategory, DocumentUrl, SavedAt, IsCorrectCategory
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id, complaint, original_response, edited_response, 
                original_category, edited_category, document_url, datetime.now(), 
                is_correct_category  # New column value
            )
        )
        conn.commit()
        logging.info(f"Saved response with ID: {response_id}")
        cursor.close()
        conn.close()

        return func.HttpResponse(
            json.dumps({"status": "Response and document saved successfully", "responseId": response_id}),
            mimetype="application/json",
            status_code=200,
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid form data."}),
            mimetype="application/json",
            status_code=400,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except pyodbc.Error as db_err:
        logging.error(f"Database error: {str(db_err)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to save response due to database issue."}),
            mimetype="application/json",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logging.error(f"Error saving response: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "An error occurred while saving the response."}),
            mimetype="application/json",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.route(route="GetRoles", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def get_roles(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"Http function processed request for url \"{req.url}\"")
    
    # Initialize roles as an empty list
    roles = []
    
    # Process request body if present (for POST requests)
    try:
        req_body = req.get_json() if req.method == "POST" else {}
        logging.info(f"Request body: {req_body}")
    except ValueError:
        req_body = {}
        logging.info("No valid JSON body provided.")

    # Helper function to get all roles claims
    def get_roles_claims() -> list:
        user_claims = req_body.get("claims", [])
        logging.info("User Claims: {user_claims}")
        # Extract all claims with typ "roles"
        role_values = [claim.get("val") for claim in user_claims if (claim.get("typ") == "roles" or claim.get("typ") == "http://schemas.microsoft.com/ws/2008/06/identity/claims/role") and claim.get("val")]
        return role_values

    # Get all roles from claims
    claim_roles = get_roles_claims()

    # Push specific values into roles array based on claim values
    for role in claim_roles:
        if role == "complaintsysadmin":  # Example: Map "admin" to a specific role
            roles.append("complaintsysadmin")
            logging.info("append complaintsysadmin")
        elif role == "complaintsysuser":  # Example: Include "user" as-is
            roles.append("complaintsysuser")
            logging.info("append complaintsysuser")
        elif role == "guest":  # Example: Include "guest" with a prefix
            roles.append("temp_guest")
            logging.info("append temp_guest")
        # Add more conditions as needed, or include all roles unconditionally
        # Uncomment the next line to include all roles as-is without filtering
        # roles.append(role)

    return func.HttpResponse(
        json.dumps({"roles": roles}),
        mimetype="application/json",
        status_code=200,
        headers={"Access-Control-Allow-Origin": "*"}
    )



@app.route(route="GetSavedResponses", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def get_saved_responses(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"Http function processed request for url \"{req.url}\"")
    
    if req.method != "GET":
        return func.HttpResponse("Method not allowed. Use GET.", status_code=405)

    try:
        # Connect to Azure SQL Database and execute query
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT [Id], [ResponseId], [Complaint], [OriginalResponse], [EditedResponse], 
                   [OriginalCategory], [EditedCategory], [DocumentUrl], [SavedAt]
            FROM [dbo].[ComplaintResponses]
        """)
        
        # Fetch all rows and format as a list of dictionaries
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Convert datetime objects to ISO strings for JSON compatibility
        for result in results:
            if "SavedAt" in result and isinstance(result["SavedAt"], datetime):
                result["SavedAt"] = result["SavedAt"].isoformat()

        cursor.close()
        conn.close()

        return func.HttpResponse(
            json.dumps({"responses": results}),
            mimetype="application/json",
            status_code=200,
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except pyodbc.Error as db_err:
        logging.error(f"Database error: {str(db_err)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to retrieve saved responses due to database issue."}),
            mimetype="application/json",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logging.error(f"Error retrieving saved responses: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "An error occurred while retrieving saved responses."}),
            mimetype="application/json",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )