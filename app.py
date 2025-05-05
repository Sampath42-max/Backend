import os
import PyPDF2
from flask import Flask, jsonify, request, make_response, session
from flask_cors import CORS
from weasyprint import HTML
from dotenv import load_dotenv
import nltk
import re
import pytesseract
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer, util
import json
import spacy
import textstat 
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"NLTK download error: {str(e)}")
    raise Exception(f"Failed to download NLTK data: {str(e)}")

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Session configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secure-secret-key')  # Replace with secure key in .env
app.config['SESSION_PERMANENT'] = False  # Session expires on browser close
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Required for CORS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS

# MongoDB configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/test'
mongo = PyMongo(app)
users_collection = mongo.db.Proj_Resume

# Enable CORS with credentials support
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Configuration for file uploads
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        raise FileNotFoundError("Tesseract executable not found at the specified path")
except Exception as e:
    print(f"Tesseract configuration error: {str(e)}")
    raise Exception(f"Failed to configure Tesseract: {str(e)}")

# Configure Poppler path
try:
    poppler_path = r"C:\Program Files\Poppler\bin"
    os.environ["PATH"] += os.pathsep + poppler_path
except Exception as e:
    print(f"Poppler configuration error: {str(e)}")
    raise Exception(f"Failed to configure Poppler: {str(e)}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"spaCy loading error: {str(e)}")
    raise Exception(f"Failed to load spaCy model: {str(e)}")

# Load BERT model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"SentenceTransformer loading error: {str(e)}")
    raise Exception(f"Failed to load SentenceTransformer model: {str(e)}")

# Load companies data with validation
try:
    with open('server/companies.json', 'r') as f:
        companies = json.load(f)
    if not isinstance(companies, list):
        raise ValueError("companies.json must contain a list of company objects")
    # Validate that each company has an 'id' key
    valid_companies = []
    for company in companies:
        if not isinstance(company, dict) or 'id' not in company:
            print(f"Invalid company entry (missing 'id'): {company}")
            continue
        valid_companies.append(company)
    companies = valid_companies
    if not companies:
        raise ValueError("No valid companies found in companies.json")
    print(f"Loaded {len(companies)} valid companies from companies.json")
except Exception as e:
    print(f"Failed to load companies.json: {str(e)}")
    raise Exception(f"Failed to load companies.json: {str(e)}")

# Minimal resume data
resume_data = {
    '1': {'name': 'Template 1', 'content': 'Default content for Template 1'},
    '2': {'name': 'Template 2', 'content': 'Default content for Template 2'},
    '3': {'name': 'Template 3', 'content': 'Default content for Template 3'},
    '4': {'name': 'Template 4', 'content': 'Default content for Template 4'},
    '5': {'name': 'Template 5', 'content': 'Default content for Template 5'},
    '6': {'name': 'Template 6', 'content': 'Default content for Template 6'},
    '7': {'name': 'Template 7', 'content': 'Default content for Template 7'},
    '8': {'name': 'Template 8', 'content': 'Default content for Template 8'},
    '9': {'name': 'Template 9', 'content': 'Default content for Template 9'},
}

# Helper functions
def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])

def convert_keys_to_camel_case(data):
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_keys_to_camel_case(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_camel_case(item) for item in data]
    else:
        return data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    try:
        print("Converting PDF to images...")
        images = convert_from_bytes(pdf_file.read())
        if not images:
            raise ValueError("No images extracted from the PDF")
        
        print(f"Extracted {len(images)} images from the PDF")
        text = ''
        for i, image in enumerate(images):
            print(f"Extracting text from image {i + 1}...")
            page_text = pytesseract.image_to_string(image)
            if page_text:
                text += page_text + '\n'
            else:
                print(f"No text extracted from image {i + 1}")
        
        if not text.strip():
            raise ValueError("No text extracted from the PDF after processing all images")
        
        print("Text extraction successful")
        return text
    except Exception as e:
        print(f"Text extraction failed: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

# ATS Scoring Functions (unchanged)
def calculate_keyword_matching_score(resume_text):
    print("Calculating keyword matching score...")
    general_keywords = [
        "communication", "leadership", "teamwork", "problem-solving", "management",
        "analysis", "development", "design", "project", "research",
        "marketing", "sales", "customer service", "data", "software",
        "programming", "networking", "cloud", "security", "automation",
        "graphic", "typography", "illustration", "ui", "ux", "visual", "creative"
    ]

    doc = nlp(resume_text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    keyword_matches = sum(1 for token in tokens if token in general_keywords)
    detected_keywords = list(set(token for token in tokens if token in general_keywords))

    score = min(100, (keyword_matches / 5) * 100)
    print(f"Keyword matching score: {score}, Detected keywords: {detected_keywords}")
    return round(score, 2), detected_keywords

def calculate_structure_formatting_score(resume_text):
    print("Calculating structure and formatting score...")
    score = 0
    max_points = 100
    found_sections = []

    sections = {
        "contact": r'(?i)(contact|phone|email)',
        "experience": r'(?i)(experience|work history|employment)',
        "education": r'(?i)(education|academic background)',
        "skills": r'(?i)(skills|technical skills|core competencies)',
        "about": r'(?i)(about me|summary|profile)'
    }
    for section, pattern in sections.items():
        if re.search(pattern, resume_text):
            found_sections.append(section)
            score += 15

    bullet_points = len(re.findall(r'[•\-\*]\s', resume_text))
    bullet_score = min(25, (bullet_points / 3) * 25)
    score += bullet_score

    words = resume_text.split()
    all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 3)
    caps_ratio = all_caps_words / len(words) if words else 0
    caps_penalty = 5 if caps_ratio > 0.3 else 0
    spacing_issues = len(re.findall(r'\s{3,}', resume_text))
    spacing_penalty = 5 if spacing_issues > 10 else 0
    formatting_score = 25 - caps_penalty - spacing_penalty
    score += max(0, formatting_score)

    print(f"Structure and formatting score: {score}, Sections found: {found_sections}")
    return round(score, 2), found_sections

def calculate_experience_skills_score(resume_text):
    print("Calculating experience and skills score...")
    score = 0
    max_points = 100

    action_verbs = [
        "managed", "developed", "optimized", "led", "designed",
        "implemented", "improved", "analyzed", "created", "delivered",
        "worked", "collaborated", "contributed"
    ]
    experience_section = re.split(r'(?i)(experience|work history|employment)', resume_text)[-1]
    verb_count = sum(1 for verb in action_verbs if re.search(rf'\b{verb}\b', experience_section, re.IGNORECASE))
    action_verb_score = min(60, (verb_count / 2) * 60)
    score += action_verb_score

    skills_section_pattern = r'(?i)(skills|technical skills|core competencies)'
    has_skills_section = bool(re.search(skills_section_pattern, resume_text))
    skills_section_score = 20 if has_skills_section else 0
    score += skills_section_score

    detected_skills = []
    if has_skills_section:
        skills_section = re.split(skills_section_pattern, resume_text, flags=re.IGNORECASE)[-1]
        skills_list = re.findall(r'(?:[•\-\*]\s*|\b)([A-Za-z\s]+)(?:,|\n|$)', skills_section)
        detected_skills = [skill.strip() for skill in skills_list if len(skill.strip()) > 2]
    skills_count_score = min(20, (len(detected_skills) / 2) * 20)
    score += skills_count_score

    print(f"Experience and skills score: {score}, Detected skills: {detected_skills}")
    return round(score, 2), detected_skills

def calculate_grammar_readability_score(resume_text):
    print("Calculating grammar and readability score...")
    score = 0
    max_points = 100

    try:
        fk_score = textstat.flesch_kincaid_grade(resume_text)
        if fk_score <= 6:
            readability_score = 60
        elif fk_score <= 10:
            readability_score = 50
        elif fk_score <= 14:
            readability_score = 40
        else:
            readability_score = 30
    except:
        readability_score = 40
    score += readability_score

    sentences = re.split(r'[.!?]\s', resume_text)
    long_sentences = sum(1 for sentence in sentences if len(sentence.split()) > 30)
    repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', resume_text, re.IGNORECASE))
    grammar_penalty = (long_sentences * 3) + (repeated_words * 3)
    grammar_score = max(0, 40 - grammar_penalty)
    score += grammar_score

    print(f"Grammar and readability score: {score}")
    return round(score, 2)

def calculate_contact_length_score(resume_text):
    print("Calculating contact info and length score...")
    score = 0
    max_points = 100

    contact_points = 0
    if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', resume_text):
        contact_points += 15
    if re.search(r'\b\d{3}-\d{3}-\d{4}\b', resume_text):
        contact_points += 15
    if re.search(r'(?i)linkedin\.com', resume_text):
        contact_points += 10
    if re.search(r'\b\d+\s+\w+\s+(st|ave|rd|blvd|dr)\b', resume_text, re.IGNORECASE):
        contact_points += 10
    score += contact_points

    word_count = len(resume_text.split())
    if 300 <= word_count <= 700:
        length_score = 50
    elif 200 <= word_count < 300 or 700 < word_count <= 900:
        length_score = 40
    else:
        length_score = 30
    score += length_score

    print(f"Contact info and length score: {score}")
    return round(score, 2)

def calculate_industry_match_score(resume_text, company_industry, max_points=35):
    try:
        print("Calculating industry match score...")
        industry_keywords = {
            "Technology": ["software", "development", "programming", "cloud", "networking", "security", "data", "automation"],
            "Finance": ["finance", "accounting", "banking", "investment", "audit", "tax", "budget", "financial analysis"],
            "Healthcare": ["healthcare", "medical", "nursing", "pharmacy", "clinical", "patient care", "research", "diagnostics"],
            "Marketing": ["marketing", "advertising", "branding", "social media", "campaign", "SEO", "content", "analytics"],
            "Design": ["graphic", "design", "visual", "typography", "illustration", "ui", "ux", "creative"]
        }.get(company_industry, [])

        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        keyword_embeddings = model.encode(industry_keywords, convert_to_tensor=True)
        scores = util.cos_sim(resume_embedding, keyword_embeddings)[0]
        matched_industries = [industry_keywords[i] for i, score in enumerate(scores) if score > 0.4]
        industry_score = min(len(matched_industries) * 10, max_points)
        print(f"Industry match score: {industry_score}, Matched industries: {matched_industries}")
        return industry_score, matched_industries
    except Exception as e:
        print(f"Industry match score calculation failed: {str(e)}")
        return 0, []

def calculate_company_reputation_score(resume_text, company_tier, max_points=25):
    try:
        print("Calculating company reputation score...")
        tier_keywords = {
            "Tier 1": ["Google", "Microsoft", "Amazon", "Apple", "Facebook", "CISCO"],
            "Tier 2": ["Tech Mahindra", "TCL", "Infosys", "Wipro"],
            "Tier 3": ["Startup", "Small Business", "Aldenaire", "Thynk", "Warderie"]
        }.get(company_tier, [])

        tiered_companies = [company for company in tier_keywords if company.lower() in resume_text.lower()]
        reputation_score = min(len(tiered_companies) * 10, max_points)
        print(f"Company reputation score: {reputation_score}, Tiered companies: {tiered_companies}")
        return reputation_score, tiered_companies
    except Exception as e:
        print(f"Company reputation score calculation failed: {str(e)}")
        return 0, []

def calculate_job_role_match_score(resume_text, target_role, max_points=25):
    try:
        print("Calculating job role match score...")
        role_keywords = {
            "Software Engineer": ["software", "development", "programming", "coding", "engineer"],
            "Network Engineer": ["network", "infrastructure", "routing", "switching", "engineer"],
            "Hardware Engineer": ["hardware", "electronics", "circuit", "design", "engineer"],
            "Software Developer": ["software", "development", "coding", "developer"],
            "Graphic Designer": ["graphic", "design", "visual", "typography", "illustration", "ui", "ux"]
        }.get(target_role, [])

        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        keyword_embeddings = model.encode(role_keywords, convert_to_tensor=True)
        scores = util.cos_sim(resume_embedding, keyword_embeddings)[0]
        matched_roles = [role_keywords[i] for i, score in enumerate(scores) if score > 0.4]
        role_score = min(len(matched_roles) * 5, max_points)
        print(f"Job role match score: {role_score}, Matched roles: {matched_roles}")
        return role_score, matched_roles
    except Exception as e:
        print(f"Job role match score calculation failed: {str(e)}")
        return 0, []

def calculate_tenure_score(resume_text, max_points=15):
    try:
        print("Calculating tenure score...")
        experience_matches = re.findall(r'(\d+)\s*(?:year|yr)s?', resume_text, re.IGNORECASE)
        tenures = [int(years) for years in experience_matches]
        average_tenure = sum(tenures) / len(tenures) if tenures else 0
        if not tenures:
            year_matches = re.findall(r'(\d{4})\s*–\s*(?:(\d{4})|Now)', resume_text, re.IGNORECASE)
            tenures = []
            for start, end in year_matches:
                end_year = 2025 if end.lower() == "now" else int(end)
                tenure = end_year - int(start)
                tenures.append(tenure)
            average_tenure = sum(tenures) / len(tenures) if tenures else 0
        tenure_score = min(average_tenure * 5, max_points) if average_tenure > 0 else 5
        print(f"Tenure score: {tenure_score}, Average tenure: {average_tenure} years")
        return tenure_score
    except Exception as e:
        print(f"Tenure score calculation failed: {str(e)}")
        return 5

def calculate_location_match_score(resume_text, company_location, max_points=5):
    try:
        print("Calculating location match score...")
        matched_locations = [company_location] if company_location.lower() in resume_text.lower() else []
        location_score = max_points if matched_locations else 0
        print(f"Location match score: {location_score}, Matched locations: {matched_locations}")
        return location_score, matched_locations
    except Exception as e:
        print(f"Location match score calculation failed: {str(e)}")
        return 0, []

# Routes
@app.route('/', methods=['GET'])
def home():
    print("Received GET request for /")
    return jsonify({"message": "Flask server is running!"}), 200

@app.route('/api/resume/<template_id>', methods=['GET'])
def get_resume(template_id):
    print(f"Received GET request for /api/resume/{template_id}")
    data = resume_data.get(template_id)
    if data:
        return jsonify(data)
    return jsonify({"error": "Template not found"}), 404

@app.route('/api/resume/check', methods=['POST'])
def check_resume():
    try:
        print("Received request for /api/resume/check")
        if 'resume' not in request.files:
            print("No resume file provided in the request")
            return jsonify({"error": "No resume file provided"}), 400

        resume_file = request.files['resume']
        if not resume_file:
            print("Resume file is empty")
            return jsonify({"error": "Resume file is empty"}), 400

        print(f"Received file: {resume_file.filename}")
        if not allowed_file(resume_file.filename):
            print("File is not a PDF")
            return jsonify({"error": "Only PDF files are supported"}), 400

        resume_file.seek(0)
        resume_text = extract_text_from_pdf(resume_file)

        keyword_score, detected_keywords = calculate_keyword_matching_score(resume_text)
        structure_score, sections_found = calculate_structure_formatting_score(resume_text)
        experience_skills_score, detected_skills = calculate_experience_skills_score(resume_text)
        grammar_readability_score = calculate_grammar_readability_score(resume_text)
        contact_length_score = calculate_contact_length_score(resume_text)

        final_score = (
            (keyword_score * 0.4) +
            (structure_score * 0.2) +
            (experience_skills_score * 0.25) +
            (grammar_readability_score * 0.1) +
            (contact_length_score * 0.05)
        )

        response = {
            "resume_text": resume_text,
            "normal_score": round(final_score, 2),
            "normal_score_details": {
                "keyword_matching_score": keyword_score,
                "structure_formatting_score": structure_score,
                "experience_skills_score": experience_skills_score,
                "grammar_readability_score": grammar_readability_score,
                "contact_length_score": contact_length_score,
                "detected_keywords": detected_keywords,
                "detected_skills": detected_skills,
                "sections_found": sections_found
            }
        }
        print("Response prepared successfully for /api/resume/check:", response)
        return jsonify(convert_keys_to_camel_case(response))

    except Exception as e:
        print(f"Error in /api/resume/check: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/resume/check-with-company', methods=['POST'])
def check_resume_with_company():
    try:
        print("Received request for /api/resume/check-with-company")
        print("Form data:", request.form)
        print("Files:", request.files)

        # Check for required fields
        if 'resume' not in request.files or 'companyId' not in request.form:
            print("Missing resume file or companyId")
            return jsonify({"error": "Missing resume file or companyId"}), 400

        resume_file = request.files['resume']
        try:
            company_id = request.form['companyId']
        except KeyError:
            print("companyId not found in form data")
            return jsonify({"error": "companyId is required"}), 400

        # Validate resume file
        if not resume_file or not allowed_file(resume_file.filename):
            print("Invalid or empty resume file")
            return jsonify({"error": "Invalid or empty resume file (PDF only)"}), 400

        print(f"Received file: {resume_file.filename}, Company ID: {company_id}")
        print(f"Available company IDs: {[c['id'] for c in companies]}")

        # Find company with flexible ID comparison
        company = next((c for c in companies if str(c['id']) == str(company_id) or c['id'] == company_id), None)
        if not company:
            print(f"Company with ID {company_id} not found")
            return jsonify({"error": f"Company with ID {company_id} not found"}), 404
        print(f"Company found: {company['name']}")

        resume_file.seek(0)
        resume_text = extract_text_from_pdf(resume_file)

        keyword_score_normal, matched_keywords_normal = calculate_keyword_matching_score(resume_text)
        structure_score, sections_found = calculate_structure_formatting_score(resume_text)
        experience_skills_score, matched_skills_normal = calculate_experience_skills_score(resume_text)
        grammar_readability_score = calculate_grammar_readability_score(resume_text)
        contact_length_score = calculate_contact_length_score(resume_text)
        normal_score = (
            (keyword_score_normal * 0.4) +
            (structure_score * 0.2) +
            (experience_skills_score * 0.25) +
            (grammar_readability_score * 0.1) +
            (contact_length_score * 0.05)
        )

        industry_match_score, matched_industry_companies = calculate_industry_match_score(resume_text, company.get('industry', ''))
        company_reputation_score, tiered_companies = calculate_company_reputation_score(resume_text, company.get('tier', ''))
        job_role_match_score, matched_roles = calculate_job_role_match_score(resume_text, company.get('target_role', ''))
        tenure_score = calculate_tenure_score(resume_text)
        location_match_score, matched_locations = calculate_location_match_score(resume_text, company.get('location', ''))

        company_score = (
            (industry_match_score * 0.35) +
            (company_reputation_score * 0.25) +
            (job_role_match_score * 0.25) +
            (tenure_score * 0.15)
        )

        combined_ats_score = (normal_score + company_score) / 2

        response = {
            "resume_text": resume_text,
            "combined_ats_score": round(combined_ats_score, 2),
            "normal_score": round(normal_score, 2),
            "normal_score_details": {
                "keyword_matching_score": keyword_score_normal,
                "structure_formatting_score": structure_score,
                "experience_skills_score": experience_skills_score,
                "grammar_readability_score": grammar_readability_score,
                "contact_length_score": contact_length_score,
                "matched_keywords": matched_keywords_normal,
                "matched_skills": matched_skills_normal,
                "sections_found": sections_found
            },
            "company_name": company['name'],
            "company_score": round(company_score, 2),
            "company_score_details": {
                "industry_match_score": round(industry_match_score, 2),
                "company_reputation_score": round(company_reputation_score, 2),
                "job_role_match_score": round(job_role_match_score, 2),
                "tenure_score": round(tenure_score, 2),
                "location_match_score": round(location_match_score, 2),
                "matched_industry_companies": matched_industry_companies,
                "tiered_companies": tiered_companies,
                "matched_roles": matched_roles,
                "matched_locations": matched_locations
            }
        }
        print("Final company score breakdown:", response["company_score_details"])
        return jsonify(convert_keys_to_camel_case(response))

    except Exception as e:
        print(f"Error in /api/resume/check-with-company: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        print("Received signup data:", data)
        if not data:
            print("No data provided")
            return jsonify({"error": "No data provided"}), 400

        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        if not all([name, email, password]):
            print("Missing required fields")
            return jsonify({"error": "Name, email, and password are required"}), 400

        if users_collection.find_one({'email': email}):
            print("Email already exists")
            return jsonify({"error": "Email already exists"}), 400

        print("Hashing password and preparing insertion")
        hashed_password = generate_password_hash(password)
        result = users_collection.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password
        })
        print("Inserted user ID:", result.inserted_id)
        session['user_id'] = str(result.inserted_id)  # Set session
        return jsonify({"message": "User created successfully"}), 201
    except Exception as e:
        print(f"Error in signup: {str(e)}")
        return jsonify({"error": "An error occurred. Please try again."}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print("Received login data:", data)
        if not data:
            print("No data provided")
            return jsonify({"error": "No data provided"}), 400

        email = data.get('email')
        password = data.get('password')

        if not all([email, password]):
            print("Missing required fields")
            return jsonify({"error": "Email and password are required"}), 400

        user = users_collection.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])  # Set session
            print(f"Login successful, session user_id: {session['user_id']}")
            return jsonify({"message": "User logged in successfully", "username": user['name']}), 200
        else:
            print("Invalid email or password")
            return jsonify({"error": "Invalid email or password"}), 401
    except Exception as e:
        print(f"Error in login: {str(e)}")
        return jsonify({"error": "An error occurred. Please try again."}), 500

@app.route('/api/profile', methods=['GET'])
def profile():
    try:
        print("Received profile request")
        user_id = session.get('user_id')
        print(f"Session user_id: {user_id}")
        if not user_id:
            print("No user_id in session")
            return jsonify({}), 401
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if user:
            print(f"User found: {user['name']}")
            return jsonify({"username": user['name'], "email": user['email']}), 200
        print("User not found in database")
        return jsonify({}), 401
    except Exception as e:
        print(f"Error in profile: {str(e)}")
        return jsonify({"error": "An error occurred. Please try again."}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        print("Received logout request")
        session.pop('user_id', None)
        print("Session cleared")
        return jsonify({"message": "Logged out successfully"}), 200
    except Exception as e:
        print(f"Error in logout: {str(e)}")
        return jsonify({"error": "An error occurred. Please try again."}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    try:
        if 'resume' in request.files:
            file = request.files['resume']
            company = request.form.get('company', '')
            if not company:
                return jsonify({'error': 'No company provided'}), 400
            pdf = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
            score = len(text) % 100
            return jsonify({
                'normalScore': score,
                'companyScore': score + 10,
                'generalFeedback': 'Good resume structure.',
                'companyFeedback': f'Tailored for {company}.'
            })
        else:
            data = request.get_json()
            resume_id = data.get('resumeId')
            company = data.get('company')
            if not resume_id or not company:
                return jsonify({'error': 'Missing resumeId or company'}), 400
            return jsonify({
                'normalScore': 85,
                'companyScore': 90,
                'generalFeedback': 'Good resume structure.',
                'companyFeedback': f'Tailored for {company}.'
            })
    except Exception as e:
        print(f'Error: {str(e)}')
        return jsonify({'error': f'Failed to analyze resume: {str(e)}'}), 500
        
if __name__ == '__main__':
    app.run(debug=True, port=5001) 