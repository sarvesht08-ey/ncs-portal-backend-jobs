import os
import json
import re
import asyncio
import pickle
from typing import List, Union, Any, Optional, Dict
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Any, Optional, Dict, Literal, Tuple
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import asyncpg
from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss


from fastapi import File, UploadFile
from typing import Literal
import fitz  # PyMuPDF - you'll need: pip install PyMuPDF
import docx

from cv import CVProcessor, CVProfile
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive embedding operations
embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

# Azure OpenAI client for text generation only
try:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint and not endpoint.endswith('/'):
        endpoint = endpoint + '/'
    
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        azure_endpoint=endpoint
    )
    
    gpt_deployment = os.getenv("AZURE_GPT_DEPLOYMENT")
    if not gpt_deployment:
        raise ValueError("Missing AZURE_GPT_DEPLOYMENT in environment variables")
        
    logger.info(f"Initialized Azure client for GPT: {gpt_deployment}")
    
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    raise

# PostgreSQL connection string
DB_URL = os.getenv("DATABASE_URL")


class ChatRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    response: str
    message_type: str = "text"
    chat_phase: Optional[str] = None
    profile_data: Optional[Dict] = None
    jobs: Optional[List[Dict]] = None
    suggestions: Optional[List[str]] = []
    location_searched: Optional[str] = None
    location_matches: Optional[Dict] = None
    total_found: Optional[int] = None
    filters_applied: Optional[Dict] = None
    search_context: Optional[Dict] = None

class CVUploadResponse(BaseModel):
    response: str
    profile_data: Optional[Dict] = None
    success: bool = True

class CVUploadRequest(BaseModel):
    """CV upload request model"""
    pass

class ChatWithCVRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []
    cv_profile_data: Optional[Dict] = None

class CVAnalysisResponse(BaseModel):
    """Enhanced CV analysis response"""
    success: bool
    message: str
    profile: Optional[Dict[str, Any]] = None
    jobs: Optional[List[Dict]] = None
    total_jobs_found: int = 0
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    recommendations: Optional[List[str]] = None

class LocationJobRequest(BaseModel):
    location: str = Field(..., min_length=2, max_length=100, description="City, district, or state name")
    job_type: Optional[str] = Field(default=None, description="Optional job type filter")
    skills: Optional[List[str]] = Field(default=None, description="Optional skills filter")
    experience_range: Optional[Tuple[float, float]] = Field(default=None, description="Min, Max experience in years")
    salary_range: Optional[Tuple[float, float]] = Field(default=None, description="Min, Max salary range")
    limit: Optional[int] = Field(default=50, ge=1, le=200, description="Maximum jobs to return")
    sort_by: Optional[Literal["relevance", "salary", "experience", "date"]] = "relevance"

class LocationJobResponse(BaseModel):
    location_searched: str
    location_matches: Dict[str, List[str]]
    jobs: List[Dict[str, Any]]
    total_found: int
    returned_count: int
    processing_time_ms: int
    filters_applied: Dict[str, Any]
    search_context: Dict[str, Any]


class LocationMappingService:
    """Service to handle location name mapping and normalization"""
    
    def __init__(self):
        # Comprehensive location mappings
        self.city_to_state = {
            # Major Cities to States
            'mumbai': 'Maharashtra', 'bombay': 'Maharashtra',
            'pune': 'Maharashtra', 'nagpur': 'Maharashtra', 'nashik': 'Maharashtra',
            'aurangabad': 'Maharashtra', 'solapur': 'Maharashtra', 'kolhapur': 'Maharashtra',
            
            'delhi': 'Delhi', 'new delhi': 'Delhi', 'gurgaon': 'Haryana', 'gurugram': 'Haryana',
            'noida': 'Uttar Pradesh', 'ghaziabad': 'Uttar Pradesh', 'faridabad': 'Haryana',
            
            'bangalore': 'Karnataka', 'bengaluru': 'Karnataka', 'mysore': 'Karnataka',
            'hubli': 'Karnataka', 'mangalore': 'Karnataka', 'belgaum': 'Karnataka',
            
            'chennai': 'Tamil Nadu', 'madras': 'Tamil Nadu', 'coimbatore': 'Tamil Nadu',
            'madurai': 'Tamil Nadu', 'salem': 'Tamil Nadu', 'tiruchirapalli': 'Tamil Nadu',
            'trichy': 'Tamil Nadu', 'vellore': 'Tamil Nadu',
            
            'hyderabad': 'Telangana', 'secunderabad': 'Telangana', 'warangal': 'Telangana',
            
            'kolkata': 'West Bengal', 'calcutta': 'West Bengal', 'durgapur': 'West Bengal',
            'siliguri': 'West Bengal', 'howrah': 'West Bengal',
            
            'ahmedabad': 'Gujarat', 'surat': 'Gujarat', 'vadodara': 'Gujarat',
            'rajkot': 'Gujarat', 'bhavnagar': 'Gujarat', 'jamnagar': 'Gujarat',
            
            'jaipur': 'Rajasthan', 'udaipur': 'Rajasthan', 'jodhpur': 'Rajasthan',
            'kota': 'Rajasthan', 'bikaner': 'Rajasthan', 'ajmer': 'Rajasthan',
            
            'lucknow': 'Uttar Pradesh', 'kanpur': 'Uttar Pradesh', 'agra': 'Uttar Pradesh',
            'varanasi': 'Uttar Pradesh', 'meerut': 'Uttar Pradesh', 'allahabad': 'Uttar Pradesh',
            'prayagraj': 'Uttar Pradesh', 'bareilly': 'Uttar Pradesh',
            
            'bhopal': 'Madhya Pradesh', 'indore': 'Madhya Pradesh', 'gwalior': 'Madhya Pradesh',
            'jabalpur': 'Madhya Pradesh', 'ujjain': 'Madhya Pradesh',
            
            'patna': 'Bihar', 'gaya': 'Bihar', 'bhagalpur': 'Bihar', 'muzaffarpur': 'Bihar',
            
            'bhubaneswar': 'Odisha', 'cuttack': 'Odisha', 'rourkela': 'Odisha',
            
            'chandigarh': 'Chandigarh', 'amritsar': 'Punjab', 'ludhiana': 'Punjab',
            'jalandhar': 'Punjab', 'patiala': 'Punjab',
            
            'kochi': 'Kerala', 'cochin': 'Kerala', 'thiruvananthapuram': 'Kerala',
            'trivandrum': 'Kerala', 'kozhikode': 'Kerala', 'calicut': 'Kerala',
            'thrissur': 'Kerala', 'kollam': 'Kerala',
            
            'visakhapatnam': 'Andhra Pradesh', 'vijayawada': 'Andhra Pradesh',
            'guntur': 'Andhra Pradesh', 'nellore': 'Andhra Pradesh', 'tirupati': 'Andhra Pradesh',
            
            'guwahati': 'Assam', 'dibrugarh': 'Assam', 'jorhat': 'Assam',
            
            'ranchi': 'Jharkhand', 'jamshedpur': 'Jharkhand', 'dhanbad': 'Jharkhand',
            
            'raipur': 'Chhattisgarh', 'bilaspur': 'Chhattisgarh',
            
            'panaji': 'Goa', 'margao': 'Goa',
            
            'dehradun': 'Uttarakhand', 'haridwar': 'Uttarakhand'
        }
        
        # State name normalization
        self.state_aliases = {
            'mh': 'Maharashtra', 'maharashtra': 'Maharashtra',
            'ka': 'Karnataka', 'karnataka': 'Karnataka',
            'tn': 'Tamil Nadu', 'tamil nadu': 'Tamil Nadu', 'tamilnadu': 'Tamil Nadu',
            'ts': 'Telangana', 'telangana': 'Telangana',
            'ap': 'Andhra Pradesh', 'andhra pradesh': 'Andhra Pradesh',
            'wb': 'West Bengal', 'west bengal': 'West Bengal',
            'gj': 'Gujarat', 'gujarat': 'Gujarat',
            'rj': 'Rajasthan', 'rajasthan': 'Rajasthan',
            'up': 'Uttar Pradesh', 'uttar pradesh': 'Uttar Pradesh',
            'mp': 'Madhya Pradesh', 'madhya pradesh': 'Madhya Pradesh',
            'dl': 'Delhi', 'delhi': 'Delhi',
            'hr': 'Haryana', 'haryana': 'Haryana',
            'pb': 'Punjab', 'punjab': 'Punjab',
            'or': 'Odisha', 'odisha': 'Odisha', 'orissa': 'Odisha',
            'jh': 'Jharkhand', 'jharkhand': 'Jharkhand',
            'cg': 'Chhattisgarh', 'chhattisgarh': 'Chhattisgarh',
            'uk': 'Uttarakhand', 'uttarakhand': 'Uttarakhand',
            'br': 'Bihar', 'bihar': 'Bihar',
            'as': 'Assam', 'assam': 'Assam',
            'kl': 'Kerala', 'kerala': 'Kerala',
            'ga': 'Goa', 'goa': 'Goa'
        }
        
        # Regional keywords
        self.regional_keywords = {
            'north india': ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Uttarakhand', 'Rajasthan'],
            'south india': ['Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Kerala'],
            'west india': ['Maharashtra', 'Gujarat', 'Rajasthan', 'Goa'],
            'east india': ['West Bengal', 'Odisha', 'Jharkhand', 'Bihar'],
            'central india': ['Madhya Pradesh', 'Chhattisgarh'],
            'metro cities': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata'],
            'tier 1 cities': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad'],
        }
        
    def normalize_location(self, location_input: str) -> Dict[str, List[str]]:
        """
        Normalize location input and return potential states and districts
        Returns: {"states": [list], "districts": [list]}
        """
        location_lower = location_input.lower().strip()
        location_lower = re.sub(r'\b(city|district|state|region|area)\b', '', location_lower).strip()
        
        states = []
        districts = []
        
        # Check regional keywords first
        for region, region_states in self.regional_keywords.items():
            if region in location_lower:
                if region in ['metro cities', 'tier 1 cities']:
                    districts.extend(region_states)
                else:
                    states.extend(region_states)
                return {"states": states, "districts": districts}
        
        # Check if it's a direct state match
        if location_lower in self.state_aliases:
            states.append(self.state_aliases[location_lower])
            return {"states": states, "districts": districts}
        
        # Check if it's a city/district
        if location_lower in self.city_to_state:
            districts.append(location_input.title())
            corresponding_state = self.city_to_state[location_lower]
            if corresponding_state not in states:
                states.append(corresponding_state)
            return {"states": states, "districts": districts}
        
        # Fuzzy matching for typos or variations
        for city, state in self.city_to_state.items():
            if city in location_lower or location_lower in city:
                districts.append(city.title())
                if state not in states:
                    states.append(state)
        
        for alias, state in self.state_aliases.items():
            if alias in location_lower or location_lower in alias:
                if state not in states:
                    states.append(state)
        
        # If no matches found, treat as potential district name
        if not states and not districts:
            districts.append(location_input.title())
        
        return {"states": list(set(states)), "districts": list(set(districts))}

class LocationJobSearchService:
    """Enhanced service for location-based job searching"""
    
    def __init__(self):
        self.location_mapper = LocationMappingService()
        
    async def search_jobs_by_location(self, request: LocationJobRequest) -> LocationJobResponse:
        """Search jobs by location with comprehensive filtering"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Normalize location input
            location_matches = self.location_mapper.normalize_location(request.location)
            logger.info(f"Extracted Location location_matches : {location_matches}")

            
            if not location_matches["states"] and not location_matches["districts"]:
                return LocationJobResponse(
                    location_searched=request.location,
                    location_matches=location_matches,
                    jobs=[],
                    total_found=0,
                    returned_count=0,
                    processing_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
                    filters_applied={},
                    search_context={"error": "Location not recognized"}
                )
            
            # Build dynamic SQL query
            query, params = self._build_location_query(request, location_matches)
            
            # Execute query
            jobs = await self._execute_location_query(query, params)
            
            # Apply additional filtering if needed
            if request.skills:
                jobs = await self._filter_by_skills(jobs, request.skills)
            
            # Apply sorting
            jobs = self._apply_sorting(jobs, request.sort_by)
            
            # Limit results
            limited_jobs = jobs[:request.limit]
            
            processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            return LocationJobResponse(
                location_searched=request.location,
                location_matches=location_matches,
                jobs=limited_jobs,
                total_found=len(jobs),
                returned_count=len(limited_jobs),
                processing_time_ms=processing_time_ms,
                filters_applied=self._get_applied_filters(request),
                search_context={
                    "query_executed": True,
                    "location_normalized": location_matches,
                    "total_before_limit": len(jobs)
                }
            )
            
        except Exception as e:
            logger.error(f"Location job search failed: {e}")
            processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            return LocationJobResponse(
                location_searched=request.location,
                location_matches={},
                jobs=[],
                total_found=0,
                returned_count=0,
                processing_time_ms=processing_time_ms,
                filters_applied={},
                search_context={"error": str(e)}
            )
    
    def _build_location_query(self, request: LocationJobRequest, location_matches: Dict) -> Tuple[str, List]:
        """Build SQL query for location-based search"""
        
        base_query = """
            SELECT ncspjobid, title, keywords, description, 
                   organization_name, statename, districtname,
                   industryname, sectorname, functionalareaname, functionalrolename,
                   aveexp, avewage, numberofopenings, highestqualification,
                   gendercode, date
            FROM vacancies_summary WHERE
        """
        
        conditions = []
        params = []
        param_index = 1
        
        # Location conditions
        location_conditions = []
        
        if location_matches["states"]:
            state_placeholders = []
            for state in location_matches["states"]:
                state_placeholders.append(f'${param_index}')
                params.append(state)
                param_index += 1
            location_conditions.append(f"statename = ANY(ARRAY[{','.join(state_placeholders)}])")
        
        if location_matches["districts"]:
            district_placeholders = []
            for district in location_matches["districts"]:
                district_placeholders.append(f'${param_index}')
                params.append(district)
                param_index += 1
            location_conditions.append(f"districtname = ANY(ARRAY[{','.join(district_placeholders)}])")
        
        if location_conditions:
            conditions.append(f"({' OR '.join(location_conditions)})")
        
        # Experience filter
        if request.experience_range:
            min_exp, max_exp = request.experience_range
            if min_exp is not None:
                conditions.append(f"aveexp >= ${param_index}")
                params.append(min_exp)
                param_index += 1
            if max_exp is not None:
                conditions.append(f"aveexp <= ${param_index}")
                params.append(max_exp)
                param_index += 1
        
        # Salary filter
        if request.salary_range:
            min_salary, max_salary = request.salary_range
            if min_salary is not None:
                conditions.append(f"avewage >= ${param_index}")
                params.append(min_salary)
                param_index += 1
            if max_salary is not None:
                conditions.append(f"avewage <= ${param_index}")
                params.append(max_salary)
                param_index += 1
        
        
        if request.job_type:
            job_type_condition = f"(LOWER(title) LIKE ${param_index} OR LOWER(keywords) LIKE ${param_index} OR LOWER(functionalrolename) LIKE ${param_index})"
            conditions.append(job_type_condition)
            params.extend([f"%{request.job_type.lower()}%"] * 3)
            param_index += 3
        
        if conditions:
            base_query +=  "".join(conditions)
        
        base_query += " ORDER BY ncspjobid DESC LIMIT 2000"
        logger.info(f"Query : {base_query}")
        return base_query, params
    
    async def _execute_location_query(self, query: str, params: List) -> List[Dict]:
        """Execute the location query and return formatted results"""
        conn = None
        try:
            conn = await asyncpg.connect(DB_URL)
            rows = await conn.fetch(query, *params)
            logger.info(f"Query :{query}")
            jobs = []
            for row in rows:
                job_dict = {
                    'ncspjobid': row['ncspjobid'],
                    'title': row['title'],
                    'keywords': row['keywords'],
                    'description': row['description'],
                    'organization_name': row['organization_name'],
                    'statename': row['statename'],
                    'districtname': row['districtname'],
                    'industryname': row['industryname'],
                    'sectorname': row['sectorname'],
                    'functionalareaname': row['functionalareaname'],
                    'functionalrolename': row['functionalrolename'],
                    'aveexp': float(row['aveexp']) if row['aveexp'] else 0,
                    'avewage': float(row['avewage']) if row['avewage'] else 0,
                    'numberofopenings': int(row['numberofopenings']) if row['numberofopenings'] else 1,
                    'highestqualification': row['highestqualification'],
                    'gendercode': row['gendercode'],
                    'date': row['date'].isoformat() if row['date'] else None,
                    'match_percentage': 75  # Default match for location-based search
                }
                # print(job_dict)
                jobs.append(job_dict)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Database query execution failed: {e}")
            return []
        finally:
            if conn:
                await conn.close()
    
    async def _filter_by_skills(self, jobs: List[Dict], skills: List[str]) -> List[Dict]:
        """Filter jobs by skills with enhanced matching and update match percentage"""
        if not skills:
            return jobs

        filtered_jobs = []
        skills_lower = [skill.lower().strip() for skill in skills]

        # Create skill variations for better matching
        skill_variations = {}
        for skill in skills_lower:
            variations = [skill]
            # Add common variations
            if 'data entry' in skill:
                variations.extend(['data processing', 'typing', 'data operator', 'data entry operator'])
            elif 'python' in skill:
                variations.extend(['python developer', 'python programming', 'django', 'flask'])
            elif 'javascript' in skill or 'js' in skill:
                variations.extend(['javascript developer', 'js developer', 'frontend developer'])
            elif 'customer service' in skill:
                variations.extend(['call center', 'voice process', 'customer support', 'telecalling'])

            skill_variations[skill] = variations

        for job in jobs:
            # Get job text for matching
            job_text = f"{job.get('title', '')} {job.get('keywords', '')} {job.get('description', '')} {job.get('functionalrolename', '')}".lower()

            skill_matches = 0
            matched_skills = []
            partial_matches = []

            for skill in skills_lower:
                # Check exact match first
                if skill in job_text:
                    skill_matches += 1
                    matched_skills.append(skill)
                # Check skill variations
                elif any(var in job_text for var in skill_variations.get(skill, [])):
                    skill_matches += 0.8
                    matched_skills.append(skill)
                # Check partial word matches
                elif any(word in job_text for word in skill.split() if len(word) > 2):
                    skill_matches += 0.3
                    partial_matches.append(skill)

            # More lenient filtering - include jobs with any skill match
            if skill_matches >= 0.3:  # Lower threshold for inclusion
                skill_match_ratio = skill_matches / len(skills)
                # Calculate match percentage based on skill matching
                base_score = 60 if matched_skills else 45
                job['match_percentage'] = min(95, base_score + (skill_match_ratio * 35))
                job['skills_matched'] = matched_skills if matched_skills else partial_matches
                filtered_jobs.append(job)

        # If filtering results in too few jobs, return original list with updated scores
        if len(filtered_jobs) < 5 and len(jobs) > 0:
            logger.info(f"Skill filtering too restrictive, returning all jobs with updated scores")
            for job in jobs[:50]:  # Return top 50 from location search
                job_text = f"{job.get('title', '')} {job.get('keywords', '')}".lower()
                # Calculate a basic match score
                matches = sum(1 for skill in skills_lower if skill in job_text)
                job['match_percentage'] = min(75, 40 + (matches / len(skills)) * 35)
                job['skills_matched'] = [s for s in skills_lower if s in job_text]
            return jobs[:50]

        return sorted(filtered_jobs, key=lambda x: x.get('match_percentage', 0), reverse=True)
    
    def _apply_sorting(self, jobs: List[Dict], sort_by: str) -> List[Dict]:
        """Apply sorting to job results"""
        if sort_by == "salary":
            return sorted(jobs, key=lambda x: x.get('avewage', 0), reverse=True)
        elif sort_by == "experience":
            return sorted(jobs, key=lambda x: x.get('aveexp', 0), reverse=True)
        elif sort_by == "date":
            return sorted(jobs, key=lambda x: x.get('date', ''), reverse=True)
        else:  # relevance (default)
            return sorted(jobs, key=lambda x: x.get('match_percentage', 0), reverse=True)
    
    def _get_applied_filters(self, request: LocationJobRequest) -> Dict[str, Any]:
        """Get summary of applied filters"""
        filters = {"location": request.location}
        
        if request.job_type:
            filters["job_type"] = request.job_type
        if request.skills:
            filters["skills"] = request.skills
        if request.experience_range:
            filters["experience_range"] = request.experience_range
        if request.salary_range:
            filters["salary_range"] = request.salary_range
        
        filters["sort_by"] = request.sort_by
        return filters
    
class EnhancedChatService:
    """Complete enhanced chat service that integrates location search with existing chat functionality"""
    
    def __init__(self):
        self.location_job_service = LocationJobSearchService()
        self.location_mapper = LocationMappingService()
        self.fallback_count = {}  # Track fallback responses to prevent loops
        self.conversation_state = {}  # Track conversation state per user

        # Complete skill keywords dictionary
        self.skill_keywords = {
            # Technical/IT Skills
            'python': ['python', 'py', 'django', 'flask', 'fastapi'],
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'node.js'],
            'react': ['react', 'reactjs', 'react.js', 'nextjs', 'next.js'],
            'angular': ['angular', 'angularjs'],
            'vue': ['vue', 'vuejs', 'vue.js', 'nuxt'],
            'java': ['java', 'spring', 'springboot', 'spring boot'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
            'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite'],
            'mongodb': ['mongodb', 'mongo'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3', 'sass', 'scss', 'tailwind'],
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'git': ['git', 'github', 'gitlab'],
            'machine learning': ['ml', 'machine learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras'],
            'data science': ['data science', 'pandas', 'numpy', 'matplotlib', 'jupyter'],
            'typescript': ['typescript', 'ts'],
            'php': ['php', 'laravel', 'symfony'],
            'ruby': ['ruby', 'rails', 'ruby on rails'],
            
            # Business Process/BPO Skills
            'data entry': ['data entry', 'data processing', 'typing', 'keyboarding'],
            'voice process': ['voice process', 'call center', 'customer service', 'telecalling', 'telesales'],
            'chat process': ['chat process', 'chat support', 'live chat', 'online support'],
            'email support': ['email support', 'email handling', 'email management'],
            'back office': ['back office', 'administrative', 'admin work'],
            'content writing': ['content writing', 'copywriting', 'blogging', 'article writing'],
            'virtual assistant': ['virtual assistant', 'va', 'personal assistant'],
            
            # Finance & Accounting
            'accounting': ['accounting', 'bookkeeping', 'accounts', 'financial'],
            'tally': ['tally', 'tally erp'],
            'excel': ['excel', 'microsoft excel', 'spreadsheet', 'vlookup', 'pivot tables'],
            'sap': ['sap', 'sap fico', 'sap mm', 'sap hr'],
            'quickbooks': ['quickbooks', 'quick books'],
            'gst': ['gst', 'goods and services tax', 'taxation'],
            'payroll': ['payroll', 'salary processing', 'hr payroll'],
            
            # Sales & Marketing
            'sales': ['sales', 'selling', 'business development', 'lead generation'],
            'digital marketing': ['digital marketing', 'online marketing', 'internet marketing'],
            'seo': ['seo', 'search engine optimization'],
            'sem': ['sem', 'search engine marketing', 'google ads', 'ppc'],
            'social media': ['social media', 'facebook marketing', 'instagram marketing', 'linkedin'],
            'email marketing': ['email marketing', 'mailchimp', 'newsletter'],
            'content marketing': ['content marketing', 'inbound marketing'],
            
            # Healthcare & Medical
            'nursing': ['nursing', 'nurse', 'rn', 'lpn'],
            'medical': ['medical', 'healthcare', 'clinical'],
            'pharmacy': ['pharmacy', 'pharmacist', 'pharma'],
            
            # Education & Training
            'teaching': ['teaching', 'teacher', 'education', 'tutor'],
            'training': ['training', 'corporate training', 'soft skills'],
            
            # Manufacturing & Operations
            'manufacturing': ['manufacturing', 'production', 'operations'],
            'quality control': ['quality control', 'qc', 'quality assurance', 'qa'],
            'logistics': ['logistics', 'supply chain', 'warehouse'],
            
            # Human Resources
            'hr': ['hr', 'human resources', 'recruitment', 'talent acquisition'],
            'payroll': ['payroll', 'hr payroll', 'compensation'],
            
            # Design & Creative
            'graphic design': ['graphic design', 'design', 'photoshop', 'illustrator'],
            'ui/ux': ['ui', 'ux', 'user interface', 'user experience'],
            'video editing': ['video editing', 'premiere', 'after effects'],
            
            # Legal & Compliance
            'legal': ['legal', 'law', 'compliance', 'contracts'],
            'paralegal': ['paralegal', 'legal assistant'],
            
            # Project Management
            'project management': ['project management', 'pmp', 'agile', 'scrum'],
            'business analyst': ['business analyst', 'ba', 'requirements'],
        }
    
    async def handle_chat_message(self, request: ChatRequest) -> ChatResponse:
        """Main chat handler that routes queries based on intent (skill-only, location-only, or skill+location)"""
        message = request.message.lower().strip()
        user_id = id(request)  # Simple user tracking

        try:
            # Reset fallback count if message has meaningful content
            if len(message) > 3 and not message in ['okay', 'ok', 'yes', 'no']:
                self.fallback_count[user_id] = 0

            # Parse query intent - detect skill-only, location-only, or combined location + skill queries
            query_intent = await self._parse_query_intent(message)
            query_type = query_intent.get('query_type', 'general')

            logger.info(f"=== Routing query with type: {query_type} ===")

            # Route based on query type
            if query_type in ['location_only', 'location_skill', 'skill_location']:
                # Location-based search (with or without skills)
                logger.info(f"Routing to location handler with skills: {query_intent.get('skills', [])}")
                return await self._handle_location_job_query(request, query_intent)

            elif query_type == 'skill_only':
                # Skill-based search (no location)
                logger.info(f"Routing to skill handler with skills: {query_intent.get('skills', [])}")
                return await self._handle_skill_job_query(request, query_intent)

            else:
                # General conversation or unclear intent - use regular chat
                logger.info("Routing to regular chat handler")
                return await self._handle_regular_chat(request)

        except Exception as e:
            logger.error(f"Enhanced chat failed: {e}")
            # Track fallback to prevent loops
            self.fallback_count[user_id] = self.fallback_count.get(user_id, 0) + 1

            if self.fallback_count.get(user_id, 0) > 2:
                # After 2 fallbacks, provide more specific guidance
                return ChatResponse(
                    response="Let me help you get started! Here are some specific examples:\n\n1. 'Show me Python jobs in Mumbai'\n2. 'Data Entry positions in Delhi'\n3. 'I know Python and React'\n4. Upload your CV using the file upload option",
                    message_type="text",
                    chat_phase="intro",
                    suggestions=["Python jobs in Mumbai", "Data Entry in Delhi", "Upload CV", "I know Python"]
                )

            return ChatResponse(
                response="I can help you find jobs! You can ask about specific locations like 'Jobs in Mumbai' or tell me about your skills.",
                message_type="text",
                chat_phase="profile_building",
                suggestions=["Jobs in Mumbai", "My skills are...", "Remote work", "Entry level jobs"]
            )

    async def _parse_query_intent(self, message: str) -> Dict[str, Any]:
        """Parse user query to extract location, skills, and intent - handles combined queries"""
        intent = {
            'has_location': False,
            'has_skills': False,
            'location': None,
            'skills': [],
            'job_type': None,
            'query_type': 'general'
        }

        # STEP 1: Try Azure OpenAI-based intelligent classification first
        try:
            logger.info("=== Using Azure OpenAI Query Classification ===")
            classification = await query_classifier.classify_and_extract(message)

            # If classification has high confidence and is not general, use it
            if classification['confidence'] >= 0.7 and classification['query_type'] != 'general':
                logger.info(f"✓ Using AI classification: {classification}")

                # Map query_type from AI to internal format
                ai_query_type = classification['query_type']
                if ai_query_type == 'skill_only':
                    intent['query_type'] = 'skill_only'
                    intent['has_skills'] = True
                    intent['skills'] = classification['skills']
                elif ai_query_type == 'location_only':
                    intent['query_type'] = 'location_only'
                    intent['has_location'] = True
                    intent['location'] = classification['location']
                elif ai_query_type == 'skill_location':
                    intent['query_type'] = 'location_skill'
                    intent['has_skills'] = True
                    intent['has_location'] = True
                    intent['skills'] = classification['skills']
                    intent['location'] = classification['location']

                logger.info(f"✓ AI-based intent - Location: {intent['location']}, Skills: {intent['skills']}, Type: {intent['query_type']}")
                return intent
            else:
                logger.info(f"AI classification confidence too low ({classification['confidence']}) or general query, falling back to regex")

        except Exception as e:
            logger.warning(f"AI classification failed, falling back to regex patterns: {e}")

        # STEP 2: Fallback to regex-based pattern matching
        logger.info("=== Using Regex-based Pattern Matching ===")

        # Known cities and states for better detection
        known_locations = [
            'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad',
            'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna',
            'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut',
            'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad',
            'amritsar', 'navi mumbai', 'allahabad', 'ranchi', 'howrah', 'coimbatore',
            'maharashtra', 'karnataka', 'tamil nadu', 'delhi', 'telangana', 'andhra pradesh',
            'west bengal', 'gujarat', 'rajasthan', 'uttar pradesh', 'madhya pradesh'
        ]

        # Enhanced patterns to detect combined queries
        combined_patterns = [
            # Pattern: "jobs in [location] on/for [skill]"
            r'(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)\s+(?:on|for|in|with|of)\s+([a-zA-Z\s]+)',
            # Pattern: "[skill] jobs in [location]"
            r'([a-zA-Z\s]+)\s+(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)',
            # Pattern: "show me [skill] in [location]"
            r'(?:show|find|get|search)\s+(?:me\s+)?([a-zA-Z\s]+)\s+(?:jobs?|positions?)?\s+in\s+([a-zA-Z\s]+)',
            # Pattern: "show/find/get/give me jobs in [location]" (location only)
            r'(?:show|find|get|give|search)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|positions?|openings?|vacancies?)\s+in\s+([a-zA-Z\s]+)$',
            # Pattern: "jobs/positions in [location]" (location only - start of message)
            r'^(?:jobs?|positions?|openings?|vacancies?)\s+in\s+([a-zA-Z\s]+)$',
        ]

        # Check combined patterns first
        for pattern_idx, pattern in enumerate(combined_patterns):
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # Handle location-only patterns (indices 3 and 4)
                if pattern_idx in [3, 4]:
                    # These patterns only extract location
                    part1 = match.group(1).strip()
                    part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()

                    intent['location'] = part1_clean
                    intent['has_location'] = True
                    intent['query_type'] = 'location_only'
                    logger.info(f"✓ Location-only query detected - Location: {intent['location']}")
                    return intent

                # Extract both parts for combined patterns
                part1 = match.group(1).strip()
                part2 = match.group(2).strip() if match.lastindex >= 2 else None

                if not part2:
                    continue

                # Clean up common words
                part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()
                part2_clean = re.sub(r'\b(the|all|any|some)\b', '', part2, flags=re.IGNORECASE).strip()

                # Smart detection: Check which part is a known location
                part1_is_location = part1_clean.lower() in known_locations
                part2_is_location = part2_clean.lower() in known_locations

                # Check if parts contain location-related words
                part1_has_location_words = any(loc in part1_clean.lower() for loc in known_locations)
                part2_has_location_words = any(loc in part2_clean.lower() for loc in known_locations)

                logger.info(f"Pattern {pattern_idx}: part1='{part1_clean}' (is_loc={part1_is_location}), part2='{part2_clean}' (is_loc={part2_is_location})")

                # Determine location and skill based on detection
                if part2_is_location or part2_has_location_words:
                    # part2 is location, part1 is skill
                    intent['location'] = part2_clean
                    intent['skills'] = self._extract_skills_from_text(part1_clean)
                    logger.info(f"Detected: Location={part2_clean}, Skills from '{part1_clean}'")
                elif part1_is_location or part1_has_location_words:
                    # part1 is location, part2 is skill
                    intent['location'] = part1_clean
                    intent['skills'] = self._extract_skills_from_text(part2_clean)
                    logger.info(f"Detected: Location={part1_clean}, Skills from '{part2_clean}'")
                else:
                    # Use pattern-specific logic
                    if pattern_idx == 0:  # "jobs in [location] on [skill]"
                        intent['location'] = part1_clean
                        intent['skills'] = self._extract_skills_from_text(part2_clean)
                    else:  # "[skill] jobs in [location]" or "show me [skill] in [location]"
                        intent['skills'] = self._extract_skills_from_text(part1_clean)
                        intent['location'] = part2_clean

                # Validate we have both location and skills
                if intent['location'] or intent['skills']:
                    intent['has_location'] = bool(intent['location'])
                    intent['has_skills'] = bool(intent['skills'])
                    intent['query_type'] = 'combined' if (intent['has_location'] and intent['has_skills']) else ('location_only' if intent['has_location'] else 'skill_only')
                    logger.info(f"✓ Query detected - Location: {intent['location']}, Skills: {intent['skills']}, Type: {intent['query_type']}")
                    return intent

        # If no combined pattern matched, extract separately
        logger.info("No combined pattern matched, extracting separately")
        intent['location'] = self._extract_location_from_message(message)
        intent['skills'] = self._extract_skills_from_text(message)
        intent['job_type'] = self._extract_job_type(message)

        intent['has_location'] = bool(intent['location'])
        intent['has_skills'] = bool(intent['skills'])

        if intent['has_location'] and intent['has_skills']:
            intent['query_type'] = 'location_skill'
        elif intent['has_location']:
            intent['query_type'] = 'location_only'
        elif intent['has_skills']:
            intent['query_type'] = 'skill_only'

        logger.info(f"Separate extraction - Location: {intent['location']}, Skills: {intent['skills']}, Type: {intent['query_type']}")
        return intent

    def _is_location_query(self, message: str) -> bool:
        """Detect if message is asking for location-based job search (including skill+location combos)"""
        message_lower = message.lower()

        # Comprehensive location patterns
        location_patterns = [
            r'\b(?:jobs?|openings?|vacancies?|positions?)\s+(?:in|at|for|near|from)\s+\w+',
            r'\b(?:show|find|get|give|search)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|openings?|vacancies?)\s+(?:in|for|at)\s+\w+',
            r'\w+\s+(?:jobs?|openings?|positions?|vacancies?)\s+(?:in|at|for|near)\s+\w+',  # "Data Entry jobs in Mumbai"
            r'\ball\s+(?:the\s+)?(?:jobs?|openings?|vacancies?)\s+(?:in|for|at)\s+\w+',
            r'\b(?:jobs?|positions?)\s+(?:in|at|for)\s+[a-zA-Z\s]+\s+(?:on|for|in)\s+\w+',  # "jobs in Mumbai on Data Entry"
        ]

        for pattern in location_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.info(f"Location query detected by pattern: {pattern}")
                return True

        # Check for known locations in message
        known_locations = [
            'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad',
            'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'patna', 'vadodara', 'ghaziabad',
            'noida', 'gurgaon', 'gurugram', 'faridabad', 'coimbatore', 'kochi',
            'visakhapatnam', 'ludhiana', 'agra', 'nashik', 'meerut', 'rajkot',
            'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'gujarat',
            'rajasthan', 'uttar pradesh', 'madhya pradesh', 'west bengal', 'kerala'
        ]

        # Check if message contains job keywords + location
        job_keywords = ['job', 'jobs', 'opening', 'openings', 'vacancy', 'vacancies',
                       'position', 'positions', 'work', 'employment', 'career', 'opportunity']

        has_job_keyword = any(keyword in message_lower for keyword in job_keywords)
        has_location = any(location in message_lower for location in known_locations)

        if has_job_keyword and has_location:
            logger.info(f"Location query detected: has job keyword and location")
            return True

        # Check for "in" + location patterns
        location_indicators = [' in ', ' at ', ' for ', ' near ', ' from ', ' around ']
        if has_job_keyword and any(indicator in message_lower for indicator in location_indicators):
            logger.info(f"Location query detected: has job keyword and location indicator")
            return True

        return False
    
    async def _handle_location_job_query(self, request: ChatRequest, query_intent: Dict[str, Any] = None) -> ChatResponse:
        """Handle location-based job queries with enhanced skill detection"""
        message = request.message.lower().strip()

        try:
            # Use parsed intent if available, otherwise extract
            if query_intent:
                location = query_intent.get('location')
                skills = query_intent.get('skills', [])
                job_type = query_intent.get('job_type')
            else:
                location = self._extract_location_from_message(message)
                skills = self._extract_skills_from_text(message)
                job_type = self._extract_job_type(message)

            logger.info(f"Extracted Location: {location}, Skills: {skills}, Job Type: {job_type}")

            if not location:
                return ChatResponse(
                    response="I'd be happy to help you find jobs by location! Please specify a city or state. For example:\n\n• 'Jobs in Mumbai'\n• 'Show openings in Karnataka'\n• 'IT positions in Delhi'\n• 'All jobs in Bangalore'",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Jobs in Mumbai", "IT jobs in Bangalore", "Remote positions", "Entry level jobs in Delhi"]
                )

            # Extract additional parameters if not already in query_intent
            experience_range = self._extract_experience_range(message)
            salary_range = self._extract_salary_range(message)
            
            limit = 50
            if any(word in message for word in ['all', 'every', 'complete', 'full']):
                limit = 100
            elif any(word in message for word in ['few', 'some', 'top']):
                limit = 20
            
            location_request = LocationJobRequest(
                location=location,
                job_type=job_type,
                skills=skills,
                experience_range=experience_range,
                salary_range=salary_range,
                limit=limit,
                sort_by="relevance"
            )
            
            search_response = await self.location_job_service.search_jobs_by_location(location_request)

            if search_response.jobs:
                response_text = self._format_location_success_response(search_response, skills)

                return ChatResponse(
                    response=response_text,
                    message_type="job_results",
                    chat_phase="job_results",
                    jobs=self._convert_to_chat_job_format(search_response.jobs[:10]),
                    location_searched=search_response.location_searched,
                    location_matches=search_response.location_matches,
                    total_found=search_response.total_found,
                    filters_applied=search_response.filters_applied,
                    search_context=search_response.search_context,
                    suggestions=self._get_location_followup_suggestions(search_response)
                )
            else:
                response_text = self._format_location_no_results_response(search_response)
                
                return ChatResponse(
                    response=response_text,
                    message_type="text",
                    chat_phase="job_searching",
                    location_searched=search_response.location_searched,
                    location_matches=search_response.location_matches,
                    suggestions=self._get_location_alternative_suggestions(location)
                )
                
        except Exception as e:
            logger.error(f"Location job query failed: {e}")
            return ChatResponse(
                response=f"I had trouble searching for jobs in that location. Could you try rephrasing? For example: 'Show me jobs in Mumbai' or 'IT positions in Bangalore'",
                message_type="text",
                chat_phase="job_searching",
                suggestions=["Jobs in Mumbai", "IT jobs in Bangalore", "Remote work", "Entry level positions"]
            )

    async def _handle_skill_job_query(self, request: ChatRequest, query_intent: Dict[str, Any] = None) -> ChatResponse:
        """Handle skill-only job queries using vector embeddings and GPT ranking"""
        message = request.message.lower().strip()
        user_id = id(request)

        try:
            # Use parsed intent if available, otherwise extract
            if query_intent and query_intent.get('skills'):
                skills = query_intent.get('skills', [])
            else:
                skills = self._extract_skills_from_text(message)

            logger.info(f"Skill-only search - Extracted Skills: {skills}")

            if not skills:
                return ChatResponse(
                    response="I'd be happy to help you find jobs by skill! Please specify the skills you're looking for. For example:\n\n• 'Show me Python jobs'\n• 'Find React developer positions'\n• 'Data Entry jobs'\n• 'Customer Service openings'",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Python jobs", "React developer", "Data Entry", "Customer Service"]
                )

            # Extract additional parameters
            experience_range = self._extract_experience_range(message)
            salary_range = self._extract_salary_range(message)

            # Determine result limit
            limit = 50
            if any(word in message for word in ['all', 'every', 'complete', 'full']):
                limit = 100
            elif any(word in message for word in ['few', 'some', 'top']):
                limit = 20

            # Use vector embedding search for skill matching
            self.fallback_count[user_id] = 0  # Reset on valid skill extraction

            try:
                skills_text = " ".join(skills)
                logger.info(f"Creating embedding for skills: {skills_text}")
                skills_embedding = await embedding_service.get_embedding(skills_text)

                # Search for similar jobs using vector similarity
                logger.info(f"Searching for similar jobs (top_k={limit})...")
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=limit)

                if similar_jobs:
                    logger.info(f"Found {len(similar_jobs)} similar jobs, re-ranking...")
                    # Re-rank jobs using GPT for better relevance
                    ranked_jobs = await gpt_service.rerank_jobs(skills, similar_jobs)

                    # Get top results (show top 10, but keep more for follow-up)
                    top_jobs = ranked_jobs[:10]
                    job_ids = [job["ncspjobid"] for job in top_jobs]

                    # Fetch complete job details
                    complete_jobs = await get_complete_job_details(job_ids)

                    # Format results
                    job_results = []
                    for job_data in top_jobs:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data.get("match_percentage", 0),
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0),
                                "keywords": complete_job.get("keywords", ""),
                                "description": complete_job.get("description", "")[:200] + "..."
                            })

                    skills_formatted = ', '.join(skills)
                    response_text = f"🎯 Found {len(ranked_jobs)} jobs matching your skills: **{skills_formatted}**\n\n"
                    response_text += f"Showing top {len(job_results)} results sorted by relevance.\n\n"

                    if experience_range:
                        response_text += f"💼 Experience filter: {experience_range[0]}-{experience_range[1]} years\n"
                    if salary_range:
                        response_text += f"💰 Salary filter: ₹{salary_range[0]:,.0f} - ₹{salary_range[1]:,.0f}\n"

                    # Provide location distribution summary
                    locations = {}
                    for job in job_results:
                        loc = job.get('statename', 'Unknown')
                        locations[loc] = locations.get(loc, 0) + 1

                    if locations:
                        top_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:3]
                        response_text += f"\n📍 Top locations: {', '.join([f'{loc} ({count})' for loc, count in top_locations])}"

                    logger.info(f"✓ Returning {len(job_results)} skill-matched jobs")

                    return ChatResponse(
                        response=response_text,
                        message_type="job_results",
                        chat_phase="job_results",
                        profile_data={"skills": skills},
                        jobs=job_results,
                        total_found=len(ranked_jobs),
                        filters_applied={
                            "skills": skills,
                            "experience_range": experience_range,
                            "salary_range": salary_range
                        },
                        suggestions=[
                            f"{skills[0]} jobs in Mumbai",
                            f"{skills[0]} jobs in Bangalore",
                            "Show more jobs",
                            "Filter by location"
                        ]
                    )
                else:
                    logger.warning(f"No jobs found for skills: {skills}")
                    return ChatResponse(
                        response=f"I couldn't find jobs matching: **{', '.join(skills)}**.\n\nTry:\n• Broader skill terms (e.g., 'Python' instead of 'Python 3.11')\n• Different skills\n• Adding a location: '{skills[0]} jobs in Mumbai'\n• Checking spelling",
                        message_type="text",
                        chat_phase="job_searching",
                        profile_data={"skills": skills},
                        suggestions=[
                            "Try different skills",
                            f"{skills[0]} jobs in Mumbai",
                            "Data Entry jobs",
                            "All jobs in Bangalore"
                        ]
                    )

            except Exception as e:
                logger.error(f"Skill-based job search failed: {e}")
                return ChatResponse(
                    response=f"I noted your skills: **{', '.join(skills)}**. Let me try a different approach. Would you like to:\n• Add a location: '{skills[0]} jobs in Mumbai'\n• Try different skills\n• Upload your CV for better matching",
                    message_type="text",
                    chat_phase="profile_building",
                    profile_data={"skills": skills},
                    suggestions=[
                        f"{skills[0]} jobs in Mumbai",
                        "Try different skills",
                        "Upload CV",
                        "Jobs in Bangalore"
                    ]
                )

        except Exception as e:
            logger.error(f"Skill job query handler failed: {e}")
            return ChatResponse(
                response="I had trouble searching for jobs by skill. Could you try rephrasing? For example:\n• 'Show me Python jobs'\n• 'Find Data Entry positions'\n• 'JavaScript developer openings'",
                message_type="text",
                chat_phase="job_searching",
                suggestions=["Python jobs", "Data Entry", "JavaScript developer", "Jobs in Mumbai"]
            )

    async def _handle_regular_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle regular chat using existing logic with improved fallback handling"""
        message = request.message.lower().strip()
        chat_phase = request.chat_phase
        user_profile = request.user_profile or {}
        user_id = id(request)

        try:
            if chat_phase == "intro":
                if any(word in message for word in ["upload", "cv", "resume", "file"]):
                    self.fallback_count[user_id] = 0  # Reset on valid interaction
                    return ChatResponse(
                        response="Great! Please click the paperclip icon to upload your CV. I support PDF, DOC, and DOCX files.",
                        message_type="text",
                        chat_phase="intro"
                    )
                elif any(word in message for word in ["chat", "talk", "build", "skills", "hello", "hi", "hey"]):
                    self.fallback_count[user_id] = 0  # Reset on valid interaction
                    return ChatResponse(
                        response="Perfect! Let's build your profile together. What are your main skills? (e.g., Python, React, Data Entry, Customer Service, etc.)\n\nYou can also ask about jobs in specific locations like 'Jobs in Mumbai'.",
                        message_type="text",
                        chat_phase="profile_building"
                    )
                else:
                    return ChatResponse(
                        response="I can help you find jobs in multiple ways:\n\n1. 📄 Upload your CV - I'll analyze it automatically\n2. 💬 Chat with me - I'll ask about your skills\n3. 📍 Ask about specific locations - 'Jobs in Mumbai'\n4. 🔍 Combined search - 'Data Entry jobs in Mumbai'\n\nWhich would you prefer?",
                        message_type="text",
                        chat_phase="intro",
                        suggestions=["Upload CV", "Tell me your skills", "Jobs in Mumbai", "Data Entry jobs in Mumbai"]
                    )
            
            elif chat_phase == "profile_building":
                skills = self._extract_skills_from_text(message)

                if skills:
                    self.fallback_count[user_id] = 0  # Reset on valid skill extraction
                    try:
                        skills_text = " ".join(skills)
                        skills_embedding = await embedding_service.get_embedding(skills_text)
                        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=20)

                        if similar_jobs:
                            ranked_jobs = await gpt_service.rerank_jobs(skills, similar_jobs)
                            job_ids = [job["ncspjobid"] for job in ranked_jobs[:5]]
                            complete_jobs = await get_complete_job_details(job_ids)
                            
                            job_results = []
                            for job_data in ranked_jobs[:5]:
                                complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                                if complete_job:
                                    job_results.append({
                                        "ncspjobid": job_data["ncspjobid"],
                                        "title": job_data["title"],
                                        "organization_name": complete_job.get("organization_name", ""),
                                        "match_percentage": job_data["match_percentage"],
                                        "statename": complete_job.get("statename", ""),
                                        "districtname": complete_job.get("districtname", ""),
                                        "avewage": complete_job.get("avewage", 0),
                                        "aveexp": complete_job.get("aveexp", 0)
                                    })
                            
                            return ChatResponse(
                                response=f"Great! I found {len(job_results)} jobs matching your skills: {', '.join(skills)}. Here are the top matches:",
                                message_type="job_results",
                                chat_phase="job_searching",
                                profile_data={"skills": skills},
                                jobs=job_results,
                                suggestions=["Show more jobs"]
                            )
                        else:
                            return ChatResponse(
                                response=f"I understand your skills: {', '.join(skills)}. Would you like to search in a specific location? You can ask 'Show me {skills[0]} jobs in Mumbai' for example.",
                                message_type="text",
                                chat_phase="profile_building",
                                profile_data={"skills": skills},
                                suggestions=[f"{skills[0]} jobs in Mumbai", "Jobs in Bangalore",]
                            )
                    except Exception as e:
                        logger.error(f"Job search failed in chat: {e}")
                        return ChatResponse(
                            response=f"I noted your skills: {', '.join(skills)}. What other skills do you have? Or ask me about jobs in specific cities!",
                            message_type="text",
                            chat_phase="profile_building",
                            profile_data={"skills": skills},
                            suggestions=["Add more skills", "Jobs in Mumbai", "Remote positions", "Experience level"]
                        )
                else:
                    return ChatResponse(
                        response="I'd like to help you find jobs. Please tell me your skills. For example: 'I know Python and React' or 'I can do Data Entry and Customer Service'\n\nOr ask about jobs in specific locations like 'Jobs in Mumbai'.",
                        message_type="text",
                        chat_phase="profile_building",
                        suggestions=["I know Python", "Data Entry skills", "Jobs in Mumbai", "Customer Service"]
                    )
            
            else:
                return ChatResponse(
                    response="I can help you find more jobs or search in specific locations. What would you like to do?",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Show more jobs", "Jobs in Mumbai", "Different skills", "Remote work"]
                )
                
        except Exception as e:
            logger.error(f"Regular chat error: {e}")
            # Track fallback to prevent loops
            self.fallback_count[user_id] = self.fallback_count.get(user_id, 0) + 1

            if self.fallback_count.get(user_id, 0) > 2:
                return ChatResponse(
                    response="Let's try a different approach! Here are specific examples:\n\n• 'Show me Python jobs in Mumbai'\n• 'I have Data Entry skills'\n• 'Customer Service positions in Delhi'\n• Or upload your CV for automatic matching",
                    message_type="text",
                    chat_phase="intro",
                    suggestions=["Python jobs in Mumbai", "Data Entry skills", "Upload CV", "Customer Service in Delhi"]
                )

            return ChatResponse(
                response="Let me help you find jobs. What skills do you have? Or ask me about jobs in specific locations like 'Mumbai jobs'.",
                message_type="text",
                chat_phase="profile_building",
                suggestions=["My skills are...", "Jobs in Mumbai", "Remote work", "Entry level"]
            )

    # =========================================================================
    # HELPER METHODS FOR LOCATION PROCESSING
    # =========================================================================
    
    def _extract_location_from_message(self, message: str) -> Optional[str]:
        """Extract location from chat message"""
        location_patterns = [
            r'\b(?:jobs?\s+in|openings?\s+in|vacancies?\s+in|positions?\s+in)\s+([a-zA-Z\s]+)',
            r'\b(?:show|find|get|give)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|openings?|vacancies?|positions?)\s+(?:in|for|at)\s+([a-zA-Z\s]+)',
            r'\b([a-zA-Z\s]+)\s+(?:jobs?|openings?|vacancies?|positions?)',
            r'\blocation[:\s]+([a-zA-Z\s]+)',
            r'\bin\s+([a-zA-Z\s]+)(?:\s+city|\s+state|\s+region)?'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                location = matches[0].strip()
                location = re.sub(r'\b(for|all|any|the|in|at|city|state|region|area|jobs?|openings?|vacancies?|positions?)\b', '', location, flags=re.IGNORECASE).strip()
                if location and len(location) > 1:
                    return location
        
        major_cities = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad', 'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur']
        message_words = message.split()
        
        for word in message_words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in major_cities:
                return clean_word
        
        return None
    
    def _extract_skills_from_text(self, message: str) -> List[str]:
        """Extract skills using comprehensive keyword matching with smart conflict resolution"""
        skills = []
        message_lower = message.lower()
        matched_skills = set()  # Track already matched skills

        # Sort skills by variation length (longest first) to avoid false positives
        # e.g., match "javascript" before "java"
        sorted_skills = sorted(
            self.skill_keywords.items(),
            key=lambda x: max(len(v) for v in x[1]),
            reverse=True
        )

        for main_skill, variations in sorted_skills:
            if main_skill in matched_skills:
                continue

            for variation in sorted(variations, key=len, reverse=True):  # Longest variation first
                # Use word boundary matching for single words to avoid substring matches
                if ' ' not in variation:  # Single word
                    # Check with word boundaries
                    pattern = r'\b' + re.escape(variation) + r'\b'
                    if re.search(pattern, message_lower):
                        # Special formatting for specific skills
                        if main_skill == 'c++':
                            skills.append('C++')
                        elif main_skill == 'c#':
                            skills.append('C#')
                        elif main_skill == 'javascript':
                            skills.append('JavaScript')
                            matched_skills.add('java')  # Prevent "Java" from matching
                        elif main_skill == 'typescript':
                            skills.append('TypeScript')
                        elif main_skill == 'machine learning':
                            skills.append('Machine Learning')
                        elif main_skill == 'data science':
                            skills.append('Data Science')
                        elif main_skill == 'data entry':
                            skills.append('Data Entry')
                        elif main_skill == 'voice process':
                            skills.append('Voice Process')
                        elif main_skill == 'ui/ux':
                            skills.append('UI/UX')
                        else:
                            skills.append(main_skill.title())
                        matched_skills.add(main_skill)
                        break
                else:  # Multi-word skill
                    if variation in message_lower:
                        if main_skill == 'machine learning':
                            skills.append('Machine Learning')
                        elif main_skill == 'data science':
                            skills.append('Data Science')
                        elif main_skill == 'data entry':
                            skills.append('Data Entry')
                        elif main_skill == 'voice process':
                            skills.append('Voice Process')
                        elif main_skill == 'customer service':
                            skills.append('Customer Service')
                        else:
                            skills.append(main_skill.title())
                        matched_skills.add(main_skill)
                        break
        
        # Experience pattern matching
        experience_patterns = {
            'years': r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            'months': r'(\d+)\s*(?:months?|mos?)\s*(?:of\s*)?(?:experience|exp)',
            'fresher': r'\b(?:fresher|fresh|new|entry\s*level|no\s*experience)\b'
        }
        
        for pattern_name, pattern in experience_patterns.items():
            matches = re.findall(pattern, message_lower)
            if matches:
                if pattern_name == 'years' and matches:
                    skills.append(f"{matches[0]} Years Experience")
                elif pattern_name == 'months' and matches:
                    skills.append(f"{matches[0]} Months Experience")
                elif pattern_name == 'fresher':
                    skills.append("Fresher")
                break
        
        return list(dict.fromkeys(skills))  # Remove duplicates while preserving order
    
    def _extract_job_type(self, message: str) -> Optional[str]:
        """Extract job type from message"""
        job_type_patterns = {
            'software': ['software', 'developer', 'programming', 'coding'],
            'it': ['it', 'information technology', 'tech'],
            'sales': ['sales', 'selling'],
            'marketing': ['marketing', 'digital marketing'],
            'data entry': ['data entry', 'typing'],
            'customer service': ['customer service', 'call center', 'support'],
            'finance': ['finance', 'financial', 'accounting'],
            'hr': ['hr', 'human resources', 'recruitment'],
            'healthcare': ['healthcare', 'medical', 'nursing'],
            'education': ['education', 'teaching', 'training']
        }
        
        for job_type, keywords in job_type_patterns.items():
            if any(keyword in message for keyword in keywords):
                return job_type
        
        return None
    
    def _extract_experience_range(self, message: str) -> Optional[Tuple[float, float]]:
        """Extract experience range from message"""
        # Fresher/Entry level
        if re.search(r'\bfresh(?:er)?|entry\s*level|no\s*experience\b', message, re.IGNORECASE):
            return (0, 2)
        
        # Range like "2-5 years"
        range_match = re.search(r'\b(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)\b', message, re.IGNORECASE)
        if range_match:
            return (float(range_match.group(1)), float(range_match.group(2)))
        
        # Minimum experience
        min_match = re.search(r'\b(?:minimum|min|at least)\s*(\d+)\s*(?:years?|yrs?)\b', message, re.IGNORECASE)
        if min_match:
            return (float(min_match.group(1)), 50)
        
        # Maximum experience
        max_match = re.search(r'\b(?:maximum|max|up to)\s*(\d+)\s*(?:years?|yrs?)\b', message, re.IGNORECASE)
        if max_match:
            return (0, float(max_match.group(1)))
        
        # Specific years with +
        exp_match = re.search(r'\b(\d+)(?:\+|\s*plus)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b', message, re.IGNORECASE)
        if exp_match:
            years = float(exp_match.group(1))
            if '+' in exp_match.group(0) or 'plus' in exp_match.group(0).lower():
                return (years, 50)
            else:
                return (max(0, years-1), years+2)
        
        return None
    
    def _extract_salary_range(self, message: str) -> Optional[Tuple[float, float]]:
        """Extract salary range from message"""
        def convert_salary(amount_str: str) -> float:
            """Convert salary string to number"""
            amount = float(re.sub(r'[^\d.]', '', amount_str))
            if 'lakh' in message or 'l' in amount_str.lower():
                return amount * 100000
            elif 'k' in amount_str.lower():
                return amount * 1000
            return amount
        
        # Salary range
        range_match = re.search(r'\b(?:rs\.?|₹)\s*(\d+(?:k|lakh|l)?)\s*(?:to|-)\s*(?:rs\.?|₹)?\s*(\d+(?:k|lakh|l)?)\b', message, re.IGNORECASE)
        if range_match:
            min_sal = convert_salary(range_match.group(1))
            max_sal = convert_salary(range_match.group(2))
            return (min_sal, max_sal)
        
        # Minimum salary
        min_match = re.search(r'\b(?:salary|pay|wage)\s*(?:above|over|more than|>)\s*(?:rs\.?|₹)?\s*(\d+(?:k|lakh|l)?)\b', message, re.IGNORECASE)
        if min_match:
            min_sal = convert_salary(min_match.group(1))
            return (min_sal, 10000000)  # 1 crore as max
        
        # Maximum salary
        max_match = re.search(r'\b(?:salary|pay|wage)\s*(?:below|under|less than|<)\s*(?:rs\.?|₹)?\s*(\d+(?:k|lakh|l)?)\b', message, re.IGNORECASE)
        if max_match:
            max_sal = convert_salary(max_match.group(1))
            return (0, max_sal)
        
        return None
    
    # =========================================================================
    # RESPONSE FORMATTING METHODS
    # =========================================================================
    
    def _format_location_success_response(self, search_response: LocationJobResponse, skills: List[str] = None) -> str:
        """Format successful location search response"""
        location = search_response.location_searched
        total = search_response.total_found
        returned = search_response.returned_count

        # Enhanced response with skill information
        if skills:
            response_parts = [f"🎯 Found {total} {', '.join(skills)} job openings in {location}!\n"]
        else:
            response_parts = [f"🎯 Found {total} job openings in {location}!\n"]
        
        if search_response.location_matches:
            locations = []
            if search_response.location_matches.get("states"):
                locations.extend([f"State: {s}" for s in search_response.location_matches["states"]])
            if search_response.location_matches.get("districts"):
                locations.extend([f"City: {d}" for d in search_response.location_matches["districts"]])
            
            if locations:
                response_parts.append(f"📍 Locations: {' | '.join(locations)}\n")
        
        filters = search_response.filters_applied
        filter_info = []
        
        if filters.get("job_type"):
            filter_info.append(f"Type: {filters['job_type'].title()}")
        if filters.get("skills"):
            filter_info.append(f"Skills: {', '.join(filters['skills'])}")
        if filters.get("experience_range"):
            min_exp, max_exp = filters["experience_range"]
            if max_exp == 50:
                filter_info.append(f"Experience: {min_exp}+ years")
            else:
                filter_info.append(f"Experience: {min_exp}-{max_exp} years")
        if filters.get("salary_range"):
            min_sal, max_sal = filters["salary_range"]
            if max_sal >= 10000000:
                filter_info.append(f"Salary: ₹{self._format_salary(min_sal)}+")
            else:
                filter_info.append(f"Salary: ₹{self._format_salary(min_sal)}-₹{self._format_salary(max_sal)}")
        
        if filter_info:
            response_parts.append(f"🔍 Filters: {' | '.join(filter_info)}\n")
        
        response_parts.append(f"📊 Results: Showing top {returned} opportunities")
        
        if search_response.processing_time_ms:
            response_parts.append(f" (processed in {search_response.processing_time_ms}ms)")
        
        return "\n".join(response_parts)
    
    def _format_location_no_results_response(self, search_response: LocationJobResponse) -> str:
        """Format no results response"""
        location = search_response.location_searched
        
        response_parts = [f"😔 No jobs found for {location}\n"]
        
        # Check if location was recognized
        if search_response.location_matches:
            matched_locations = []
            if search_response.location_matches.get("states"):
                matched_locations.extend(search_response.location_matches["states"])
            if search_response.location_matches.get("districts"):
                matched_locations.extend(search_response.location_matches["districts"])
            
            if matched_locations:
                response_parts.append(f"📍 I searched in: {', '.join(matched_locations)}\n")
            else:
                response_parts.append(f"⚠️ Location '{location}' might not be in our database.\n")
        
        response_parts.extend([
            "💡 Try these alternatives:",
            "• Search in nearby cities or states",
            "• Remove specific skill requirements",
            "• Try broader job categories",
            "• Check for remote work opportunities"
        ])
        
        return "\n".join(response_parts)
    
    def _convert_to_chat_job_format(self, location_jobs: List[Dict]) -> List[Dict]:
        """Convert location job results to chat job format"""
        chat_jobs = []
        
        for job in location_jobs:
            chat_job = {
                "ncspjobid": job["ncspjobid"],
                "title": job["title"],
                "organization_name": job["organization_name"],
                "match_percentage": job.get("match_percentage", 75),
                "statename": job["statename"],
                "districtname": job["districtname"],
                "avewage": job["avewage"],
                "aveexp": job["aveexp"],
                "functionalrolename": job.get("functionalrolename"),
                "industryname": job.get("industryname"),
                "keywords": job.get("keywords"),
                "skills_matched": job.get("skills_matched", [])
            }
            chat_jobs.append(chat_job)
        
        return chat_jobs
    
    def _get_location_followup_suggestions(self, search_response: LocationJobResponse) -> List[str]:
        """Get follow-up suggestions for location searches"""
        suggestions = []
        
        if search_response.total_found > search_response.returned_count:
            suggestions.append("Show more jobs")
        
        # Location-based suggestions
        if search_response.location_matches.get("states"):
            for state in search_response.location_matches["states"][:1]:
                if state != search_response.location_searched:
                    suggestions.append(f"Jobs in other cities of {state}")
        
        # Add filter suggestions
        suggestions.extend([
            "Filter by salary range",
            "Filter by experience level", 
            "Show remote jobs"
        ])
        
        return suggestions[:4]  # Return max 4 suggestions
    
    def _get_location_alternative_suggestions(self, location: str) -> List[str]:
        """Get alternative suggestions when no results found"""
        suggestions = [
            f"Remote jobs (work from {location})",
            "Jobs in nearby cities",
            "Entry level positions",
            "Browse all locations"
        ]
        
        # Add major city suggestions
        major_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
        if location.title() not in major_cities:
            suggestions.append(f"Jobs in {major_cities[0]}")
        
        return suggestions[:4]
    
    def _format_salary(self, amount: float) -> str:
        """Format salary amount for display"""
        if amount >= 100000:
            return f"{amount/100000:.1f}L"
        elif amount >= 1000:
            return f"{int(amount/1000)}K"
        else:
            return str(int(amount))
    
    # =========================================================================
    # ENHANCED FOLLOW-UP CHAT METHODS 
    # =========================================================================
    
    async def handle_cv_followup_chat(self, request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle follow-up conversations after CV analysis with location awareness"""
        message = request.message.lower().strip()
        
        try:
            # Check if this is a location-based follow-up
            if self._is_location_query(message):
                return await self._handle_cv_location_followup(request, cv_profile)
            
            # Handle more jobs request
            if any(word in message for word in ["show more jobs", "show more", "additional", "other jobs"]):
                return await self._handle_cv_more_jobs(cv_profile)
            
            # Handle skill addition
            elif any(word in message for word in ["add skill", "more skill", "also know", "i can", "i have experience"]):
                return await self._handle_cv_skill_addition(request, cv_profile)
            
            # Default response
            else:
                return ChatResponse(
                    response="I can help you with:\n• Show more job opportunities\n• Add skills to your profile\n• Search by location\n• Start a new search\n\nWhat would you like to do?",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Show more jobs", "Add skills", "Search by location", "Jobs in Mumbai"]
                )
                
        except Exception as e:
            logger.error(f"CV followup chat failed: {e}")
            return ChatResponse(
                response="I can help you find more jobs or refine your search. What would you like to do?",
                message_type="text",
                chat_phase="job_results"
            )
    
    async def _handle_cv_location_followup(self, request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle location-based queries after CV analysis"""
        message = request.message.lower().strip()
        location = self._extract_location_from_message(message)
        
        if not location:
            return ChatResponse(
                response="Which location would you like me to search in? For example: 'Jobs in Mumbai' or 'Show me positions in Bangalore'",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Jobs in Mumbai", "Positions in Delhi", "Bangalore opportunities", "Remote work"]
            )
        
        try:
            # Use CV skills for location-based search
            location_request = LocationJobRequest(
                location=location,
                skills=cv_profile.skills[:10] if hasattr(cv_profile, 'skills') else [],
                limit=20,
                sort_by="relevance"
            )
            
            search_response = await self.location_job_service.search_jobs_by_location(location_request)
            
            if search_response.jobs:
                response_text = f"🎯 Found {search_response.total_found} jobs in {location} matching your CV skills!\n\n📍 Location: {location}\n🔧 Skills from CV: {', '.join(cv_profile.skills[:5]) if hasattr(cv_profile, 'skills') else 'Various skills'}\n📊Results: Showing top {search_response.returned_count} opportunities"
                
                return ChatResponse(
                    response=response_text,
                    message_type="job_results",
                    chat_phase="job_results",
                    jobs=self._convert_to_chat_job_format(search_response.jobs[:8]),
                    location_searched=search_response.location_searched,
                    location_matches=search_response.location_matches,
                    total_found=search_response.total_found,
                    suggestions=["Show more jobs", f"Other cities in {location}", "Different location", "Salary filter"]
                )
            else:
                return ChatResponse(
                    response=f"No jobs found in {location} with your current skills. Try:\n• Different location nearby\n• Broader skill categories\n• Remote opportunities",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Remote jobs", "Jobs in Mumbai", "Nearby cities", "Different skills"]
                )
                
        except Exception as e:
            logger.error(f"CV location followup failed: {e}")
            return ChatResponse(
                response="I had trouble searching in that location. Try asking about jobs in major cities like Mumbai, Delhi, or Bangalore.",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Jobs in Mumbai", "Delhi positions", "Bangalore jobs", "Remote work"]
            )
    
    async def _handle_cv_more_jobs(self, cv_profile) -> ChatResponse:
        """Handle request for more jobs based on CV"""
        try:
            if hasattr(cv_profile, 'skills') and cv_profile.skills:
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=30)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[5:15]]  # Skip first 5, get next 10
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    job_results = []
                    for job_data in ranked_jobs[5:15]:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0),
                                "keywords": complete_job.get("keywords", ""),
                                "functionalrolename": complete_job.get("functionalrolename", ""),
                                "skills_matched": job_data.get("keywords_matched", [])
                            })
                    
                    return ChatResponse(
                        response=f"Here are {len(job_results)} more job opportunities based on your CV analysis:",
                        message_type="job_results",
                        chat_phase="job_results",
                        jobs=job_results,
                        suggestions=["Even more jobs", "Filter by location", "Salary range", "Experience level"]
                    )
            
            return ChatResponse(
                response="I'll search for more opportunities. What specific type of jobs are you most interested in?",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Software jobs", "Remote work", "Entry level", "Senior positions"]
            )
            
        except Exception as e:
            logger.error(f"More jobs request failed: {e}")
            return ChatResponse(
                response="I can help you find more jobs! What type of positions interest you most?",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Software development", "Data entry", "Sales", "Customer service"]
            )
    
    async def _handle_cv_skill_addition(self, request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle adding skills to CV profile"""
        message = request.message
        additional_skills = self._extract_skills_from_text(message)
        
        if not additional_skills:
            return ChatResponse(
                response="What additional skills would you like to add to your profile? For example: 'I also know Data Entry and Voice Process' or 'I have experience in Customer Service'",
                message_type="text",
                chat_phase="profile_refinement",
                suggestions=["I also know Python", "Customer service experience", "Data entry skills", "Sales experience"]
            )
        
        try:
            # Combine existing CV skills with new skills
            existing_skills = cv_profile.skills if hasattr(cv_profile, 'skills') else []
            new_skills = [skill for skill in additional_skills if skill not in existing_skills]
            
            if new_skills:
                combined_skills = existing_skills + new_skills
                skills_text = " ".join(combined_skills)
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=25)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(combined_skills, similar_jobs)
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[:8]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    job_results = []
                    for job_data in ranked_jobs[:8]:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0),
                                "skills_matched": job_data.get("keywords_matched", [])
                            })
                    
                    return ChatResponse(
                        response=f"Great! I've added {', '.join(new_skills)} to your profile. Here are updated job matches with your enhanced skillset:",
                        message_type="job_results",
                        chat_phase="job_results",
                        profile_data={"skills": combined_skills},
                        jobs=job_results,
                        suggestions=["More jobs", "Filter by location", "Different skills", "Experience level"]
                    )
            
            return ChatResponse(
                response="Those skills are already in your profile! Any other skills you'd like to add?",
                message_type="text",
                chat_phase="profile_refinement",
                suggestions=["Add different skills", "Search by location", "Show current profile", "Find more jobs"]
            )
            
        except Exception as e:
            logger.error(f"Skill addition failed: {e}")
            return ChatResponse(
                response="I noted your additional skills. Let me search for jobs with your updated profile.",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Show jobs", "Search by location", "Add more skills", "Different search"]
            )
    
    # =========================================================================
    # ADVANCED QUERY HANDLING
    # =========================================================================
    
    def _handle_advanced_location_queries(self, message: str) -> Optional[Dict[str, Any]]:
        """Handle advanced location queries like salary ranges, experience levels"""
        advanced_patterns = {
            'salary_location': r'(?:jobs?|positions?)\s+(?:in|at)\s+([a-zA-Z\s]+)\s+(?:with\s+)?(?:salary|pay|wage)\s+(?:above|over|more than)\s+(?:rs\.?|₹)?\s*(\d+)(?:k|000|lakh|l)?',
            'experience_location': r'(?:jobs?|positions?)\s+(?:in|at)\s+([a-zA-Z\s]+)\s+(?:with\s+|for\s+)?(\d+)(?:\+|\s*plus)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            'skill_location_exp': r'([a-zA-Z\s]+)\s+(?:jobs?|positions?)\s+(?:in|at)\s+([a-zA-Z\s]+)\s+(?:with\s+|for\s+)?(\d+)(?:\+|\s*plus)?\s*(?:years?|yrs?)',
            'remote_location': r'(?:remote|work\s+from\s+home|wfh)\s+(?:jobs?|positions?)\s+(?:in|for|from)\s+([a-zA-Z\s]+)',
        }
        
        for pattern_type, pattern in advanced_patterns.items():
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if pattern_type == 'salary_location':
                    location = match.group(1).strip()
                    salary_amount = float(match.group(2))
                    if 'lakh' in message or 'l' in message:
                        salary_amount *= 100000
                    elif 'k' in message:
                        salary_amount *= 1000
                    
                    return {
                        'type': 'salary_location',
                        'location': location,
                        'min_salary': salary_amount
                    }
                
                elif pattern_type == 'experience_location':
                    location = match.group(1).strip()
                    experience = float(match.group(2))
                    has_plus = '+' in match.group(0) or 'plus' in match.group(0)
                    
                    return {
                        'type': 'experience_location',
                        'location': location,
                        'min_experience': experience,
                        'max_experience': 50 if has_plus else experience + 2
                    }
                
                elif pattern_type == 'skill_location_exp':
                    skill = match.group(1).strip()
                    location = match.group(2).strip()
                    experience = float(match.group(3))
                    has_plus = '+' in match.group(0) or 'plus' in match.group(0)
                    
                    return {
                        'type': 'skill_location_exp',
                        'skill': skill,
                        'location': location,
                        'min_experience': experience,
                        'max_experience': 50 if has_plus else experience + 2
                    }
                
                elif pattern_type == 'remote_location':
                    location = match.group(1).strip()
                    
                    return {
                        'type': 'remote_location',
                        'location': location,
                        'remote': True
                    }
        
        return None
    
    def _is_follow_up_query(self, message: str, conversation_history: List[Dict]) -> bool:
        """Check if this is a follow-up query based on conversation history"""
        follow_up_indicators = [
            'show more', 'more jobs', 'additional', 'other', 'different',
            'also', 'too', 'as well', 'similar', 'like that', 'those',
            'in that location', 'same city', 'same place'
        ]
        
        return any(indicator in message.lower() for indicator in follow_up_indicators)
    
    def _extract_context_from_history(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract context from conversation history for follow-up queries"""
        context = {
            'previous_location': None,
            'previous_skills': [],
            'previous_job_type': None,
            'previous_filters': {}
        }
        
        # Look through recent conversation history
        for msg in reversed(conversation_history[-5:]):  # Last 5 messages
            if msg.get('type') == 'bot' and msg.get('metadata', {}).get('location_searched'):
                context['previous_location'] = msg['metadata']['location_searched']
            
            if msg.get('type') == 'bot' and msg.get('metadata', {}).get('filters_applied'):
                filters = msg['metadata']['filters_applied']
                if filters.get('skills'):
                    context['previous_skills'] = filters['skills']
                if filters.get('job_type'):
                    context['previous_job_type'] = filters['job_type']
                context['previous_filters'] = filters
                break
        
        return context

# Add enhanced service specifically for CV-based interactions
class CVChatService:
    """Enhanced chat service specifically for CV upload interactions"""
    
    @staticmethod
    async def handle_cv_upload_chat(cv_profile: CVProfile) -> ChatResponse:
        """Handle chat after CV upload with enhanced profile data"""
        try:
            if cv_profile.skills and len(cv_profile.skills) >= 3:
                # Search for jobs using extracted skills
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=15)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[:5]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    # Format jobs for chat
                    job_results = []
                    for job_data in ranked_jobs[:5]:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0)
                            })
                    
                    profile_summary = {
                        "name": cv_profile.name,
                        "skills": cv_profile.skills[:10],
                        "experience_count": len(cv_profile.experience),
                        "confidence": cv_profile.confidence_score
                    }
                    
                    return ChatResponse(
                        response=f"🎉 Perfect! I've analyzed your CV and found your skills: {', '.join(cv_profile.skills[:5])}{'...' if len(cv_profile.skills) > 5 else ''}. Here are {len(job_results)} matching jobs:",
                        message_type="cv_results",
                        chat_phase="job_results",
                        profile_data=profile_summary,
                        jobs=job_results
                    )
                else:
                    return ChatResponse(
                        response=f"I've extracted your skills: {', '.join(cv_profile.skills[:5])}. Let me search for more opportunities or we can refine your profile.",
                        message_type="text",
                        chat_phase="profile_refinement",
                        profile_data={"skills": cv_profile.skills}
                    )
            else:
                return ChatResponse(
                    response="I've processed your CV but found limited technical skills. Let's chat to build a complete profile for better job matching.",
                    message_type="text", 
                    chat_phase="profile_building"
                )
                
        except Exception as e:
            logger.error(f"CV chat integration failed: {e}")
            return ChatResponse(
                response="I've processed your CV! Let's discuss your skills to find the best job matches.",
                message_type="text",
                chat_phase="profile_building"
            )
    
    @staticmethod
    async def handle_cv_followup_chat(request: ChatRequest, cv_profile: CVProfile) -> ChatResponse:
        """Handle follow-up chat after CV analysis"""
        message = request.message.lower().strip()
        
        try:
            # Handle requests for more jobs
            if any(word in message for word in ["more jobs", "show more", "additional", "other jobs"]):
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=25)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    # Skip first 5 jobs (already shown) and get next 5
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[5:10]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    job_results = []
                    for job_data in ranked_jobs[5:10]:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0)
                            })
                    
                    return ChatResponse(
                        response=f"Here are {len(job_results)} more job opportunities based on your CV:",
                        message_type="job_results",
                        chat_phase="job_results",
                        jobs=job_results
                    )
            
            # Handle skill refinement requests
            elif any(word in message for word in ["add skill", "more skill", "also know", "i can"]):
                # Extract additional skills from message
                additional_skills = []
                skill_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'html', 'css', 'node', 'angular', 'vue', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'aws', 'docker', 'kubernetes']
                
                for skill in skill_keywords:
                    if skill in message and skill.title() not in cv_profile.skills:
                        additional_skills.append(skill.title())
                
                if additional_skills:
                    combined_skills = cv_profile.skills + additional_skills
                    skills_text = " ".join(combined_skills)
                    skills_embedding = await embedding_service.get_embedding(skills_text)
                    similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=20)
                    
                    if similar_jobs:
                        ranked_jobs = await gpt_service.rerank_jobs(combined_skills, similar_jobs)
                        job_ids = [job["ncspjobid"] for job in ranked_jobs[:5]]
                        complete_jobs = await get_complete_job_details(job_ids)
                        
                        job_results = []
                        for job_data in ranked_jobs[:5]:
                            complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                            if complete_job:
                                job_results.append({
                                    "ncspjobid": job_data["ncspjobid"],
                                    "title": job_data["title"],
                                    "organization_name": complete_job.get("organization_name", ""),
                                    "match_percentage": job_data["match_percentage"],
                                    "statename": complete_job.get("statename", ""),
                                    "districtname": complete_job.get("districtname", ""),
                                    "avewage": complete_job.get("avewage", 0),
                                    "aveexp": complete_job.get("aveexp", 0)
                                })
                        
                        return ChatResponse(
                            response=f"Great! I've added {', '.join(additional_skills)} to your profile. Here are updated job matches:",
                            message_type="job_results",
                            chat_phase="job_results",
                            profile_data={"skills": combined_skills},
                            jobs=job_results
                        )
                else:
                    return ChatResponse(
                        response="What additional skills would you like to add to your profile? For example: 'I also know Docker and AWS'",
                        message_type="text",
                        chat_phase="profile_refinement"
                    )
            
            # Default response
            else:
                return ChatResponse(
                    response="I can help you with:\n• Show more job opportunities\n• Add skills to your profile\n• Search by location\n• Start a new search\n\nWhat would you like to do?",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Show more jobs", "Add skills", "Search by location", "Start over"]
                )
                
        except Exception as e:
            logger.error(f"CV followup chat failed: {e}")
            return ChatResponse(
                response="I can help you find more jobs or refine your search. What would you like to do?",
                message_type="text",
                chat_phase="job_results"
            )


# Initialize the chat service
#chat_service = SimpleChatService()
enhanced_chat_service = EnhancedChatService()
cv_chat_service = CVChatService()


# Request/Response models
class JobSearchRequest(BaseModel):
    skills: List[str] = Field(..., min_items=1, max_items=50, description="List of skills to search for")
    limit: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of jobs to return")
    
    @validator('skills', pre=True)
    def validate_and_clean_skills(cls, v):
        if not v:
            raise ValueError('Skills list cannot be empty')
        
        # Handle different input types
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]
        
        # Filter and clean skills
        cleaned_skills = []
        for skill in v:
            if isinstance(skill, str) and skill.strip():
                # Remove special characters but keep programming symbols
                cleaned = re.sub(r'[^\w\s+#.-]', '', skill.strip())
                if cleaned and len(cleaned) > 1:
                    cleaned_skills.append(cleaned)
        
        if not cleaned_skills:
            raise ValueError('No valid skills found after cleaning')
        
        if len(cleaned_skills) > 50:
            cleaned_skills = cleaned_skills[:50]
        
        return cleaned_skills
    
class CourseRecommendationRequest(BaseModel):
    keywords_unmatched: List[str] = Field(..., min_items=1, max_items=20, description="List of unmatched keywords to get course recommendations for")
    
    @validator('keywords_unmatched', pre=True)
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        
        # Handle different input types
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]
        
        # Clean keywords
        cleaned_keywords = []
        for keyword in v:
            if isinstance(keyword, str) and keyword.strip():
                cleaned = keyword.strip()
                if len(cleaned) > 1:
                    cleaned_keywords.append(cleaned)
        
        if not cleaned_keywords:
            raise ValueError('No valid keywords found after cleaning')
        
        if len(cleaned_keywords) > 20:
            cleaned_keywords = cleaned_keywords[:20]
        
        return cleaned_keywords
    
class CourseRecommendation(BaseModel):
    course_name: str
    platform: str
    duration: str
    link: str
    educator: str
    skill_covered: str
    difficulty_level: Optional[str] = None
    rating: Optional[str] = None

class CourseRecommendationResponse(BaseModel):
    recommendations: List[CourseRecommendation]
    keywords_processed: List[str]
    total_recommendations: int
    processing_time_ms: int

class JobResult(BaseModel):
    ncspjobid: str
    title: str
    match_percentage: float = Field(..., ge=0, le=100)
    similarity_score: Optional[float] = None
    keywords: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    organizationid: Optional[int] = None
    organization_name: Optional[str] = None
    numberofopenings: Optional[int] = None
    industryname: Optional[str] = None
    sectorname: Optional[str] = None
    functionalareaname: Optional[str] = None
    functionalrolename: Optional[str] = None
    aveexp: Optional[float] = None
    avewage: Optional[float] = None
    gendercode: Optional[str] = None
    highestqualification: Optional[str] = None
    statename: Optional[str] = None
    districtname: Optional[str] = None
    keywords_matched: Optional[List[str]] = None
    keywords_unmatched: Optional[List[str]] = None
    user_skills_matched: Optional[List[str]] = None
    keyword_match_score: Optional[float] = None

class JobSearchResponse(BaseModel):
    jobs: List[JobResult]
    query_skills: List[str]
    total_found: int
    processing_time_ms: int

class LocalEmbeddingService:
    """Local embedding service using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            test_embedding = self.model.encode("test input", convert_to_tensor=False)
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation for thread pool execution"""
        try:
            if not self.model:
                raise ValueError("Model not initialized")
            
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            raise
    
    async def get_embedding(self, text: Union[str, List[str], Any]) -> List[float]:
        """Generate embedding using local Sentence Transformer model"""
        
        # Normalize input to string
        try:
            if isinstance(text, list):
                processed_text = " ".join(str(item) for item in text if item)
            elif isinstance(text, (int, float)):
                processed_text = str(text)
            elif text is None:
                raise ValueError("Embedding input cannot be None")
            else:
                processed_text = str(text)
            
            processed_text = re.sub(r'\s+', ' ', processed_text.strip())
            
            if not processed_text or len(processed_text) == 0:
                raise ValueError("Embedding input must be non-empty after processing")
            
            if len(processed_text) > 2000:
                processed_text = processed_text[:2000]
            
        except Exception as e:
            logger.error(f"Input processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                embedding_executor,
                self._generate_embedding_sync,
                processed_text
            )
            
            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding generated")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

class FAISSVectorStore:
    """FAISS-based vector store for job similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.job_metadata = []
        self.is_loaded = False
        self._lock = threading.Lock()
        self.index_file = "faiss_job_index.bin"
        self.metadata_file = "job_metadata.pkl"
    
    async def load_jobs_from_db(self, force_reload: bool = False):
        """Load jobs from PostgreSQL and build/load FAISS index"""
        if self.is_loaded and not force_reload:
            logger.info("FAISS index already loaded")
            return
        
        with self._lock:
            try:
                # Try to load existing index first
                if not force_reload and os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                    logger.info("Loading existing FAISS index from disk...")
                    self.index = faiss.read_index(self.index_file)
                    
                    with open(self.metadata_file, 'rb') as f:
                        self.job_metadata = pickle.load(f)
                    
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} jobs")
                    self.is_loaded = True
                    return
                
                logger.info("Building new FAISS index from database...")
                
                # Connect to database and fetch jobs
                conn = await asyncpg.connect(DB_URL)
                try:
                    rows = await conn.fetch("""
                        SELECT ncspjobid, title, keywords, description
                        FROM vacancies_summary
                        WHERE (keywords IS NOT NULL AND keywords != '') 
                           OR (description IS NOT NULL AND description != '')
                        ORDER BY ncspjobid;
                    """)
                    
                    if not rows:
                        logger.warning("No jobs found in database")
                        return
                    
                    logger.info(f"Found {len(rows)} jobs in database")
                    
                    # Prepare job texts and metadata
                    job_texts = []
                    self.job_metadata = []
                    
                    for row in rows:
                        # Combine title, keywords, and description for embedding
                        text_parts = []
                        if row['title']:
                            text_parts.append(row['title'])
                        if row['keywords']:
                            text_parts.append(row['keywords'])
                        if row['description']:
                            desc = row['description'][:500] if row['description'] else ""
                            if desc:
                                text_parts.append(desc)
                        
                        job_text = " ".join(text_parts)
                        job_texts.append(job_text)
                        
                        self.job_metadata.append({
                            'ncspjobid': row['ncspjobid'],
                            'title': row['title'],
                            'keywords': row['keywords'],
                            'description': row['description']
                        })
                    
                    # Generate embeddings for all jobs
                    logger.info("Generating embeddings for all jobs...")
                    embeddings = await self._generate_job_embeddings(job_texts)
                    
                    # Create FAISS index
                    self.index = faiss.IndexFlatIP(self.dimension)
                    
                    # Add embeddings to index
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings_array)
                    self.index.add(embeddings_array)
                    
                    # Save index and metadata to disk
                    faiss.write_index(self.index, self.index_file)
                    with open(self.metadata_file, 'wb') as f:
                        pickle.dump(self.job_metadata, f)
                    
                    logger.info(f"Built FAISS index with {self.index.ntotal} jobs and saved to disk")
                    self.is_loaded = True
                    
                finally:
                    await conn.close()
                    
            except Exception as e:
                logger.error(f"Failed to load jobs into FAISS: {e}")
                raise HTTPException(status_code=503, detail="Failed to initialize job search index")
    
    async def _generate_job_embeddings(self, job_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for job texts using the embedding service"""
        embeddings = []
        batch_size = 10
        
        for i in range(0, len(job_texts), batch_size):
            batch = job_texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(job_texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for text in batch:
                try:
                    embedding = await embedding_service.get_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for job text: {e}")
                    batch_embeddings.append([0.0] * self.dimension)
            
            embeddings.extend(batch_embeddings)
            await asyncio.sleep(0.1)
        
        return embeddings
    
    async def search_similar_jobs(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        """Search for similar jobs using FAISS"""
        if not self.is_loaded or self.index is None:
            raise HTTPException(status_code=503, detail="Job search index not available")
        
        try:
            # Normalize query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0:
                    job_data = self.job_metadata[idx].copy()
                    job_data['similarity'] = float(similarity)
                    results.append(job_data)
            
            logger.info(f"FAISS search returned {len(results)} similar jobs")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise HTTPException(status_code=500, detail="Job search failed")


class CourseRecommendationService:
    """Service class for course recommendations using Azure OpenAI"""
    
    @staticmethod
    async def get_course_recommendations(keywords: List[str]) -> List[Dict]:
        """Get course recommendations for unmatched keywords using Azure OpenAI GPT"""
        
        if not keywords:
            return []
        
        keywords_str = ', '.join(keywords)
        
        prompt = f"""
    You are an expert career advisor. For each of these technical skills: {keywords_str}

    Provide EXACTLY 5 specific, real courses for each skill from these platforms ONLY:
    - Udemy.com
    - Coursera.org  
    - edX.org
    - Pluralsight.com
    - LinkedIn Learning
    - DataCamp.com

    STRICT REQUIREMENTS:
    1. NO generic course names like "Learn Python" or "Master React"
    2. NO Google search links or google.com URLs
    3. Use ACTUAL course titles from real platforms
    4. Each course must have a realistic platform-specific URL
    5. EXACTLY 5 courses per keyword (total: {len(keywords) * 5} courses)

    For each course provide:
    - course_name: Specific real course title
    - platform: One of the approved platforms above
    - duration: Realistic timeframe
    - link: Actual course URL (not search links)
    - educator: Real instructor/organization name
    - skill_covered: The exact keyword from the list
    - difficulty_level: Beginner/Intermediate/Advanced
    - rating: Realistic rating like "4.5/5"

    Example format:
    {{
        "course_name": "Python for Everybody Specialization",
        "platform": "Coursera", 
        "duration": "8 months",
        "link": "https://www.coursera.org/specializations/python",
        "educator": "University of Michigan",
        "skill_covered": "Python",
        "difficulty_level": "Beginner",
        "rating": "4.8/5"
    }}

    Return ONLY a valid JSON array with {len(keywords) * 5} courses total.
    NO explanatory text. NO markdown formatting.
    """

        try:
            logger.info(f"Getting course recommendations for keywords: {keywords_str}")
            
            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a learning advisor. Return ONLY valid JSON array with exactly {len(keywords) * 5} course recommendations. NO explanation text. NO markdown. NO generic course names. NO Google links."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=4000   # Increased for more courses
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response more thoroughly
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = re.sub(r'^[^[]*', '', content)  # Remove text before JSON array
            content = re.sub(r'[^}]*$', '}]', content)  # Ensure proper JSON ending
            content = content.strip()
            
            try:
                return CourseRecommendationService._get_fallback_recommendations(keywords)
                # course_recommendations = json.loads(content)
                # if isinstance(course_recommendations, list):
                #     # Validate each course has required fields
                #     valid_courses = []
                #     for course in course_recommendations:
                #         if (isinstance(course, dict) and 
                #             'course_name' in course and 
                #             'platform' in course and
                #             'link' in course and
                #             'google.com' not in course.get('link', '').lower() and
                #             len(course.get('course_name', '')) > 10):  # Avoid generic names
                #             valid_courses.append(course)
                    
                #     if len(valid_courses) >= len(keywords) * 3:  # At least 3 per keyword
                #         logger.info(f"Successfully got {len(valid_courses)} valid course recommendations")
                #         return valid_courses[:len(keywords) * 5]  # Cap at 5 per keyword
                #     else:
                #         logger.warning(f"Only got {len(valid_courses)} valid courses, expected {len(keywords) * 5}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT course response: {e}")
                logger.debug(f"Raw response: {content}")
            
            # Fallback to sample recommendations
            logger.info("Using fallback sample course recommendations")
            return CourseRecommendationService._get_fallback_recommendations(keywords)
            
        except Exception as e:
            logger.error(f"Course recommendation failed: {e}")
            return CourseRecommendationService._get_fallback_recommendations(keywords)
    
    @staticmethod
    def _get_fallback_recommendations(keywords: List[str]) -> List[Dict]:
        """Provide curated course recommendations with exactly 5 courses per keyword"""
        
        # Comprehensive course database with 5 courses per skill
        course_database = {
            "Python": [
                {
                    "course_name": "Python for Everybody Specialization",
                    "platform": "Coursera",
                    "duration": "8 months", 
                    "link": "https://www.coursera.org/specializations/python",
                    "educator": "University of Michigan",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.8/5"
                },
                {
                    "course_name": "Complete Python Bootcamp From Zero to Hero",
                    "platform": "Udemy",
                    "duration": "22 hours",
                    "link": "https://www.udemy.com/course/complete-python-bootcamp/", 
                    "educator": "Jose Portilla",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Python Programming Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/python-fundamentals",
                    "educator": "Austin Bingham",
                    "skill_covered": "Python", 
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Computer Science and Programming Using Python",
                    "platform": "edX",
                    "duration": "9 weeks",
                    "link": "https://www.edx.org/course/introduction-to-computer-science-and-programming-7",
                    "educator": "MIT",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner", 
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "4 hours",
                    "link": "https://www.linkedin.com/learning/python-essential-training-2",
                    "educator": "Bill Weinman",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "JavaScript": [
                {
                    "course_name": "JavaScript Algorithms and Data Structures",
                    "platform": "Coursera",
                    "duration": "6 months",
                    "link": "https://www.coursera.org/learn/javascript-algorithms-data-structures",
                    "educator": "University of California San Diego",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "The Complete JavaScript Course 2024: From Zero to Expert!",
                    "platform": "Udemy", 
                    "duration": "69 hours",
                    "link": "https://www.udemy.com/course/the-complete-javascript-course/",
                    "educator": "Jonas Schmedtmann",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "All Levels",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "JavaScript Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/javascript-fundamentals",
                    "educator": "Liam McLennan",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Introduction to JavaScript",
                    "platform": "edX",
                    "duration": "6 weeks", 
                    "link": "https://www.edx.org/course/introduction-to-javascript",
                    "educator": "W3C",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                {
                    "course_name": "JavaScript Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "5 hours",
                    "link": "https://www.linkedin.com/learning/javascript-essential-training",
                    "educator": "Morten Rand-Hendriksen",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "React": [
                {
                    "course_name": "React - The Complete Guide (incl Hooks, React Router, Redux)",
                    "platform": "Udemy", 
                    "duration": "40.5 hours",
                    "link": "https://www.udemy.com/course/react-the-complete-guide-incl-hooks-react-router-redux/",
                    "educator": "Maximilian Schwarzmüller",
                    "skill_covered": "React",
                    "difficulty_level": "All Levels",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Frontend Development using React Specialization",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/react", 
                    "educator": "The Hong Kong University of Science and Technology",
                    "skill_covered": "React",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "React.js Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "2 hours", 
                    "link": "https://www.linkedin.com/learning/react-js-essential-training",
                    "educator": "Eve Porcello",
                    "skill_covered": "React",
                    "difficulty_level": "Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "React Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/react-fundamentals-update",
                    "educator": "Liam McLennan", 
                    "skill_covered": "React",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to ReactJS",
                    "platform": "edX",
                    "duration": "5 weeks",
                    "link": "https://www.edx.org/course/introduction-to-reactjs",
                    "educator": "Microsoft",
                    "skill_covered": "React",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                }
            ],
            "HTML/CSS": [
                {
                    "course_name": "HTML, CSS, and Javascript for Web Developers",
                    "platform": "Coursera",
                    "duration": "5 weeks",
                    "link": "https://www.coursera.org/learn/html-css-javascript-for-web-developers",
                    "educator": "Johns Hopkins University",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Build Responsive Real-World Websites with HTML and CSS",
                    "platform": "Udemy",
                    "duration": "37.5 hours",
                    "link": "https://www.udemy.com/course/design-and-develop-a-killer-website-with-html5-and-css3/",
                    "educator": "Jonas Schmedtmann",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner to Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "HTML5 and CSS3 Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/html-css-fundamentals",
                    "educator": "Matt Milner",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Introduction to Web Development",
                    "platform": "edX",
                    "duration": "5 weeks",
                    "link": "https://www.edx.org/course/introduction-to-web-development",
                    "educator": "University of California Davis",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                {
                    "course_name": "CSS Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "4 hours",
                    "link": "https://www.linkedin.com/learning/css-essential-training-3",
                    "educator": "Christina Truong",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "SQL": [
                {
                    "course_name": "Introduction to Structured Query Language (SQL)",
                    "platform": "Coursera",
                    "duration": "4 weeks",
                    "link": "https://www.coursera.org/learn/intro-sql",
                    "educator": "University of Michigan",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.8/5"
                },
                {
                    "course_name": "The Complete SQL Bootcamp: Go from Zero to Hero",
                    "platform": "Udemy",
                    "duration": "9 hours",
                    "link": "https://www.udemy.com/course/the-complete-sql-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "SQL Server Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/sql-server-fundamentals",
                    "educator": "Pinal Dave",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Databases: Introduction to Databases and SQL Querying",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/databases-introduction-databases-sql",
                    "educator": "IBM",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "SQL Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/sql-essential-training-3",
                    "educator": "Walter Shields",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                }
            ],
            "Django": [
                {
                    "course_name": "Django for Everybody Specialization",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/django",
                    "educator": "University of Michigan", 
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python Django - The Practical Guide",
                    "platform": "Udemy",
                    "duration": "23 hours",
                    "link": "https://www.udemy.com/course/python-django-the-practical-guide/",
                    "educator": "Maximilian Schwarzmüller",
                    "skill_covered": "Django", 
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Django Fundamentals",
                    "platform": "Pluralsight", 
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/django-fundamentals-update",
                    "educator": "Reindert-Jan Ekker",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Django Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/django-essential-training",
                    "educator": "Nick Walter",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate", 
                    "rating": "4.6/5"
                },
                {
                    "course_name": "CS50's Web Programming with Python and JavaScript",
                    "platform": "edX",
                    "duration": "12 weeks",
                    "link": "https://www.edx.org/course/cs50s-web-programming-with-python-and-javascript",
                    "educator": "Harvard University",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.8/5"
                }
            ],
            "Spring Boot": [
                {
                    "course_name": "Spring Boot Microservices and Spring Cloud",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/spring-boot-cloud",
                    "educator": "LearnQuest",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Spring Boot For Beginners",
                    "platform": "Udemy",
                    "duration": "7 hours",
                    "link": "https://www.udemy.com/course/spring-boot-tutorial-for-beginners/",
                    "educator": "in28Minutes Official",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Spring Boot Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "4 hours",
                    "link": "https://www.pluralsight.com/courses/spring-boot-fundamentals",
                    "educator": "Dan Bunker",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Spring Boot Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/spring-boot-essential-training",
                    "educator": "Frank Moley",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Spring Boot",
                    "platform": "edX",
                    "duration": "4 weeks",
                    "link": "https://www.edx.org/course/introduction-to-spring-boot",
                    "educator": "Microsoft",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.3/5"
                }
            ],
            "Data Analysis": [
                {
                    "course_name": "Google Data Analytics Professional Certificate",
                    "platform": "Coursera",
                    "duration": "6 months",
                    "link": "https://www.coursera.org/professional-certificates/google-data-analytics",
                    "educator": "Google",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python for Data Science and Machine Learning Bootcamp",
                    "platform": "Udemy",
                    "duration": "25 hours",
                    "link": "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Data Analysis Fundamentals with Tableau",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/tableau-data-analysis-fundamentals",
                    "educator": "Ben Sullins",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Data Analysis using Excel",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/introduction-to-data-analysis-using-excel",
                    "educator": "Rice University",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Excel Essential Training (Microsoft 365)",
                    "platform": "LinkedIn Learning",
                    "duration": "6 hours",
                    "link": "https://www.linkedin.com/learning/excel-essential-training-microsoft-365",
                    "educator": "Dennis Taylor",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                }
            ],
            "Power BI": [
                {
                    "course_name": "Microsoft Power BI Data Analyst Professional Certificate",
                    "platform": "Coursera", 
                    "duration": "5 months",
                    "link": "https://www.coursera.org/professional-certificates/microsoft-power-bi-data-analyst",
                    "educator": "Microsoft",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Microsoft Power BI Desktop for Business Intelligence",
                    "platform": "Udemy",
                    "duration": "20 hours",
                    "link": "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/",
                    "educator": "Maven Analytics",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Power BI Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "4 hours", 
                    "link": "https://www.pluralsight.com/courses/power-bi-fundamentals",
                    "educator": "Stacia Varga",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Analyzing and Visualizing Data with Power BI",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/analyzing-and-visualizing-data-with-power-bi",
                    "educator": "Microsoft",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Intermediate",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Power BI Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/power-bi-essential-training-3",
                    "educator": "Gini Courter",
                    "skill_covered": "Power BI", 
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                }
            ]
        }
        
        recommendations = []
        for keyword in keywords:
            keyword_normalized = keyword.strip().title()
            if keyword_normalized in course_database:
                recommendations.extend(course_database[keyword_normalized])
            else:
                # Fallback for unknown skills - still no generic names
                skill_lower = keyword.lower().replace(' ', '-')
                recommendations.extend([
                    {
                        "course_name": f"Complete {keyword} Development Masterclass",
                        "platform": "Udemy",
                        "duration": "15 hours",
                        "link": f"https://www.udemy.com/topic/{skill_lower}/",
                        "educator": "Expert Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "All Levels", 
                        "rating": "4.5/5"
                    },
                    {
                        "course_name": f"{keyword} Fundamentals",
                        "platform": "Pluralsight",
                        "duration": "6 hours",
                        "link": f"https://www.pluralsight.com/courses/{skill_lower}-fundamentals",
                        "educator": "Industry Expert",
                        "skill_covered": keyword,
                        "difficulty_level": "Beginner",
                        "rating": "4.4/5"
                    },
                    {
                        "course_name": f"{keyword} Essential Training",
                        "platform": "LinkedIn Learning",
                        "duration": "4 hours",
                        "link": f"https://www.linkedin.com/learning/{skill_lower}-essential-training",
                        "educator": "Professional Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "Intermediate",
                        "rating": "4.6/5"
                    },
                    {
                        "course_name": f"Introduction to {keyword}",
                        "platform": "edX",
                        "duration": "5 weeks",
                        "link": f"https://www.edx.org/course/introduction-to-{skill_lower}",
                        "educator": "University Partner",
                        "skill_covered": keyword,
                        "difficulty_level": "Beginner",
                        "rating": "4.3/5"
                    },
                    {
                        "course_name": f"{keyword} Professional Certificate",
                        "platform": "Coursera",
                        "duration": "4 months",
                        "link": f"https://www.coursera.org/professional-certificates/{skill_lower}",
                        "educator": "Industry Leader",
                        "skill_covered": keyword,
                        "difficulty_level": "Intermediate",
                        "rating": "4.5/5"
                    }
                ])
        
        return recommendations
    

class GPTService:
    """Service class for GPT-based reranking using Azure OpenAI"""
    
    @staticmethod
    async def rerank_jobs(skills: List[str], jobs: List[Dict]) -> List[Dict]:
        """Re-rank jobs using Azure OpenAI GPT"""
        
        if not jobs:
            return []
        
        # Prepare job data for GPT
        processed_jobs = []
        for job in jobs[:25]:  # Limit to top 25 for GPT processing
            processed_job = {
                "ncspjobid": job["ncspjobid"],
                "title": job["title"],
                "keywords": job.get("keywords", "")[:200],
                "description": job.get("description", "")[:300] if job.get("description") else "",
                "similarity": round(job.get("similarity", 0), 3)
            }
            processed_jobs.append(processed_job)
        
        jobs_json = json.dumps(processed_jobs, indent=2)
        skills_str = ', '.join(skills)
        
        prompt = f"""
You are an expert job matcher. Analyze the job seeker's skills and rank the jobs by relevance.

Job Seeker Skills: {skills_str}

Jobs to rank:
{jobs_json}

Instructions:
1. Rank jobs from best to worst match based on skill alignment
2. Assign match_percentage between 100-40 based on how well skills align with job requirements
3. Consider exact skill matches, related skills, and transferable skills
4. Higher percentage for closer skill matches
5. Return ONLY valid JSON array
6. Give me only unique ncspjobid in the json array correctly. 

Required format: [{{"ncspjobid": 123, "title": "Job Title", "match_percentage": 85}}, ...]
"""

        try:
            logger.info("Reranking jobs with Azure GPT...")
            
            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a skilled career advisor. Return only valid JSON array. No explanation text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()
            
            try:
                ranked_jobs = json.loads(content)
                if isinstance(ranked_jobs, list) and len(ranked_jobs) > 0:
                    logger.info(f"Successfully ranked {len(ranked_jobs)} jobs")
                    return ranked_jobs
                else:
                    logger.warning("GPT returned empty or invalid list")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response: {e}")
            
            # Fallback to similarity-based ranking
            logger.info("Using intelligent skill-based fallback ranking")
            fallback_jobs = []

            for job in processed_jobs:
                similarity = job.get("similarity", 0.0)
                job_keywords = job.get("keywords", "").lower()
                job_title = job.get("title", "").lower()
                job_description = job.get("description", "").lower()
                
                # Parse job keywords into list
                job_keyword_list = [kw.strip() for kw in job.get("keywords", "").split(",") if kw.strip()]
                
                # Combine all job text for matching
                job_text = f"{job_keywords} {job_title} {job_description}"
                
                # Track skill matching details
                skill_matches = 0
                partial_matches = 0
                matched_job_keywords = []
                unmatched_job_keywords = []
                user_skills_matched = []
                total_skills = len(skills)
                
                # Check each job keyword against user skills
                for job_kw in job_keyword_list:
                    job_kw_lower = job_kw.lower().strip()
                    kw_matched = False
                    
                    for skill in skills:
                        skill_lower = skill.lower()
                        
                        # Check if user skill matches this job keyword
                        if skill_lower == job_kw_lower or skill_lower in job_kw_lower or job_kw_lower in skill_lower:
                            matched_job_keywords.append(job_kw)
                            if skill not in user_skills_matched:
                                user_skills_matched.append(f"{skill} (keywords)")
                            kw_matched = True
                            break
                    
                    if not kw_matched:
                        unmatched_job_keywords.append(job_kw)
                
                # Now check user skills for matches in title/description if not found in keywords
                for skill in skills:
                    skill_lower = skill.lower()
                    already_matched = any(skill in matched for matched in user_skills_matched)
                    
                    if not already_matched:
                        # Title match (high weight)
                        if skill_lower in job_title:
                            skill_matches += 0.8
                            user_skills_matched.append(f"{skill} (title)")
                        # Description match (medium weight)
                        elif skill_lower in job_description:
                            skill_matches += 0.6
                            user_skills_matched.append(f"{skill} (description)")
                        # Partial match (low weight)
                        elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                            partial_matches += 0.3
                            user_skills_matched.append(f"{skill} (partial)")
                
                # Calculate keyword match score based on matched job keywords
                keyword_matches_count = len(matched_job_keywords)
                if job_keyword_list:
                    keyword_match_score = keyword_matches_count / len(job_keyword_list)
                else:
                    keyword_match_score = 0
                
                # Calculate user skill match score
                user_skill_score = len(user_skills_matched) / total_skills if total_skills > 0 else 0
                
                # Combine both scores (70% user skills, 30% keyword coverage)
                combined_score = (user_skill_score * 0.7) + (keyword_match_score * 0.3)
                
                # Add similarity component (20% weight)
                final_score = (combined_score * 0.8) + (similarity * 0.2)
                
                # Convert to percentage with realistic ranges
                if final_score >= 0.8:
                    match_percentage = 85 + (final_score - 0.8) * 75  # 85-100%
                elif final_score >= 0.6:
                    match_percentage = 70 + (final_score - 0.6) * 75  # 70-85%
                elif final_score >= 0.4:
                    match_percentage = 55 + (final_score - 0.4) * 75  # 55-70%
                elif final_score >= 0.2:
                    match_percentage = 40 + (final_score - 0.2) * 75  # 40-55%
                else:
                    match_percentage = 25 + final_score * 75  # 25-40%
                
                # Cap at reasonable limits
                match_percentage = max(25, min(98, match_percentage))
                
                fallback_jobs.append({
                    "ncspjobid": job["ncspjobid"],
                    "title": job["title"],
                    "match_percentage": round(match_percentage, 1),
                    "keywords_matched": matched_job_keywords,
                    "keywords_unmatched": unmatched_job_keywords,
                    "user_skills_matched": user_skills_matched,
                    "keyword_match_score": round(keyword_match_score, 2),
                    "similarity_used": round(similarity, 3)
                })
            return fallback_jobs
            
        except Exception as e:
            logger.error(f"GPT reranking failed: {e}")
            # Fallback to similarity-based ranking
            logger.info("Using intelligent skill-based fallback ranking")
            fallback_jobs = []

            for job in processed_jobs:
                similarity = job.get("similarity", 0.0)
                job_keywords = job.get("keywords", "").lower()
                job_title = job.get("title", "").lower()
                job_description = job.get("description", "").lower()
                
                # Parse job keywords into list
                job_keyword_list = [kw.strip() for kw in job.get("keywords", "").split(",") if kw.strip()]
                
                # Combine all job text for matching
                job_text = f"{job_keywords} {job_title} {job_description}"
                
                # Track skill matching details
                skill_matches = 0
                partial_matches = 0
                matched_job_keywords = []
                unmatched_job_keywords = []
                user_skills_matched = []
                total_skills = len(skills)
                
                # Check each job keyword against user skills
                for job_kw in job_keyword_list:
                    job_kw_lower = job_kw.lower().strip()
                    kw_matched = False
                    
                    for skill in skills:
                        skill_lower = skill.lower()
                        
                        # Check if user skill matches this job keyword
                        if skill_lower == job_kw_lower or skill_lower in job_kw_lower or job_kw_lower in skill_lower:
                            matched_job_keywords.append(job_kw)
                            if skill not in user_skills_matched:
                                user_skills_matched.append(f"{skill} (keywords)")
                            kw_matched = True
                            break
                    
                    if not kw_matched:
                        unmatched_job_keywords.append(job_kw)
                
                # Now check user skills for matches in title/description if not found in keywords
                for skill in skills:
                    skill_lower = skill.lower()
                    already_matched = any(skill in matched for matched in user_skills_matched)
                    
                    if not already_matched:
                        # Title match (high weight)
                        if skill_lower in job_title:
                            skill_matches += 0.8
                            user_skills_matched.append(f"{skill} (title)")
                        # Description match (medium weight)
                        elif skill_lower in job_description:
                            skill_matches += 0.6
                            user_skills_matched.append(f"{skill} (description)")
                        # Partial match (low weight)
                        elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                            partial_matches += 0.3
                            user_skills_matched.append(f"{skill} (partial)")
                
                # Calculate keyword match score based on matched job keywords
                keyword_matches_count = len(matched_job_keywords)
                if job_keyword_list:
                    keyword_match_score = keyword_matches_count / len(job_keyword_list)
                else:
                    keyword_match_score = 0
                
                # Calculate user skill match score
                user_skill_score = len(user_skills_matched) / total_skills if total_skills > 0 else 0
                
                # Combine both scores (70% user skills, 30% keyword coverage)
                combined_score = (user_skill_score * 0.7) + (keyword_match_score * 0.3)
                
                # Add similarity component (20% weight)
                final_score = (combined_score * 0.8) + (similarity * 0.2)
                
                # Convert to percentage with realistic ranges
                if final_score >= 0.8:
                    match_percentage = 85 + (final_score - 0.8) * 75  # 85-100%
                elif final_score >= 0.6:
                    match_percentage = 70 + (final_score - 0.6) * 75  # 70-85%
                elif final_score >= 0.4:
                    match_percentage = 55 + (final_score - 0.4) * 75  # 55-70%
                elif final_score >= 0.2:
                    match_percentage = 40 + (final_score - 0.2) * 75  # 40-55%
                else:
                    match_percentage = 25 + final_score * 75  # 25-40%
                
                # Cap at reasonable limits
                match_percentage = max(25, min(98, match_percentage))
                
                fallback_jobs.append({
                    "ncspjobid": job["ncspjobid"],
                    "title": job["title"],
                    "match_percentage": round(match_percentage, 1),
                    "keywords_matched": matched_job_keywords,
                    "keywords_unmatched": unmatched_job_keywords,
                    "user_skills_matched": user_skills_matched,
                    "keyword_match_score": round(keyword_match_score, 2),
                    "similarity_used": round(similarity, 3)
                })
            return fallback_jobs


class QueryClassificationService:
    """Service class for intelligent query classification using Azure OpenAI"""

    @staticmethod
    async def classify_and_extract(user_query: str) -> Dict[str, Any]:
        """
        Classify user query and extract skills/locations using Azure OpenAI.

        Returns:
            Dict with keys:
            - query_type: "skill_only", "location_only", "skill_location", or "general"
            - skills: List of extracted skills
            - location: Extracted location string (or None)
            - confidence: Confidence score (0-1)
        """

        if not user_query or len(user_query.strip()) < 3:
            return {
                'query_type': 'general',
                'skills': [],
                'location': None,
                'confidence': 0.0
            }

        prompt = f"""
You are an intelligent job search query analyzer. Analyze the user's query and extract job search intent.

User Query: "{user_query}"

Your task:
1. Determine the query type:
   - "skill_only": User is searching for jobs based on skills/technologies only
   - "location_only": User is searching for jobs in a specific location only
   - "skill_location": User is searching for jobs with both skills AND location
   - "general": General conversation, greetings, or unclear intent

2. Extract skills/technologies mentioned (programming languages, frameworks, tools, job roles, etc.)
   Examples: Java, Python, React, Data Analyst, Machine Learning, etc.

3. Extract location if mentioned (city, state, region)
   Examples: Mumbai, Bangalore, Maharashtra, etc.

4. Provide confidence score (0.0 to 1.0) for your classification

Return ONLY valid JSON in this exact format:
{{
  "query_type": "skill_only" | "location_only" | "skill_location" | "general",
  "skills": ["skill1", "skill2"],
  "location": "location_name" or null,
  "confidence": 0.95
}}

Examples:
- "Hey, I am a Java Developer. Can you find any job openings for me?"
  → {{"query_type": "skill_only", "skills": ["Java"], "location": null, "confidence": 0.95}}

- "Show me jobs in Mumbai"
  → {{"query_type": "location_only", "skills": [], "location": "Mumbai", "confidence": 0.98}}

- "I need Python developer jobs in Bangalore"
  → {{"query_type": "skill_location", "skills": ["Python"], "location": "Bangalore", "confidence": 0.97}}

- "Hello, how are you?"
  → {{"query_type": "general", "skills": [], "location": null, "confidence": 0.99}}
"""

        try:
            logger.info(f"Classifying query with Azure GPT: {user_query[:100]}...")

            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a job search query analyzer. Return ONLY valid JSON. No explanation text. No markdown."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()

            try:
                result = json.loads(content)

                # Validate result structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")

                # Ensure required fields exist
                query_type = result.get('query_type', 'general')
                skills = result.get('skills', [])
                location = result.get('location')
                confidence = result.get('confidence', 0.0)

                # Normalize skills list
                if not isinstance(skills, list):
                    skills = [str(skills)] if skills else []

                # Clean and validate skills
                skills = [s.strip() for s in skills if s and str(s).strip()]

                # Clean location
                if location:
                    location = str(location).strip()
                    if not location or location.lower() in ['null', 'none', 'n/a']:
                        location = None

                logger.info(f"✓ Query classified - Type: {query_type}, Skills: {skills}, Location: {location}, Confidence: {confidence}")

                return {
                    'query_type': query_type,
                    'skills': skills,
                    'location': location,
                    'confidence': float(confidence)
                }

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {e}")
                logger.error(f"Response content: {content}")
                return {
                    'query_type': 'general',
                    'skills': [],
                    'location': None,
                    'confidence': 0.0
                }

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return {
                'query_type': 'general',
                'skills': [],
                'location': None,
                'confidence': 0.0
            }


# Initialize services
embedding_service = LocalEmbeddingService()
vector_store = FAISSVectorStore()
gpt_service = GPTService()
course_service = CourseRecommendationService()
query_classifier = QueryClassificationService()

cv_processor = CVProcessor(
    model_path="all-MiniLM-L6-v2",
    tesseract_path=r"C:\Users\WK929BY\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"  # Update path as needed
)

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Job Search API starting up...")
    
    # Validate environment variables
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_GPT_DEPLOYMENT",
        "DATABASE_URL"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Load FAISS index
    try:
        await vector_store.load_jobs_from_db()
        logger.info("Job search index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load job search index: {e}")
        raise
    
    logger.info("Job Search API started successfully")
    yield
    
    # Shutdown
    logger.info("Job Search API shutting down...")
    embedding_executor.shutdown(wait=True)


async def get_complete_job_details(job_ids: List[str]) -> List[Dict]:
    """Fetch complete job details from database for given job IDs"""
    if not job_ids:
        return []
    
    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            rows = await conn.fetch("""
                SELECT ncspjobid, title, keywords, description, date, organizationid, 
                       organization_name, numberofopenings, industryname, sectorname, 
                       functionalareaname, functionalrolename, aveexp, avewage, 
                       gendercode, highestqualification, statename, districtname
                FROM vacancies_summary
                WHERE ncspjobid = ANY($1)
                ORDER BY ncspjobid;
            """, job_ids)
            
            # Convert to list of dictionaries
            complete_jobs = []
            for row in rows:
                job_dict = {
                    'ncspjobid': row['ncspjobid'],
                    'title': row['title'],
                    'keywords': row['keywords'],
                    'description': row['description'],
                    'date': row['date'].isoformat() if row['date'] else None,
                    'organizationid': row['organizationid'],
                    'organization_name': row['organization_name'],
                    'numberofopenings': row['numberofopenings'],
                    'industryname': row['industryname'],
                    'sectorname': row['sectorname'],
                    'functionalareaname': row['functionalareaname'],
                    'functionalrolename': row['functionalrolename'],
                    'aveexp': float(row['aveexp']) if row['aveexp'] else None,
                    'avewage': float(row['avewage']) if row['avewage'] else None,
                    'gendercode': row['gendercode'],
                    'highestqualification': row['highestqualification'],
                    'statename': row['statename'],
                    'districtname': row['districtname']
                }
                complete_jobs.append(job_dict)
                print(job_dict)
            return complete_jobs
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"Failed to fetch complete job details: {e}")
        return []

app = FastAPI(
    title="Job Search API",
    description="AI-powered job search using skills matching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )


# Main job search endpoint
@app.post("/search_jobs", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """
    Search for relevant job postings based on skills
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Job search request: {len(request.skills)} skills, limit: {request.limit}")
        
        # Combine skills into text for embedding
        skills_text = " ".join(request.skills)
        
        # Generate embedding for skills
        skills_embedding = await embedding_service.get_embedding(skills_text)
        logger.info(f"Generated embedding for skills: {skills_text[:100]}...")
        
        # Search similar jobs using FAISS
        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=50)
        
        if not similar_jobs:
            return JobSearchResponse(
                jobs=[],
                query_skills=request.skills,
                total_found=0,
                processing_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000)
            )
        
        # Re-rank with Azure GPT
        ranked_jobs = await gpt_service.rerank_jobs(request.skills, similar_jobs)

        ranked_jobs.sort(key=lambda job: job.get("match_percentage", 0), reverse=True)
        
        job_ids = [job["ncspjobid"] for job in ranked_jobs[:request.limit]]
        complete_jobs = await get_complete_job_details(job_ids)
        # Convert to response format
        job_results = []
        for job_data in ranked_jobs[:request.limit]:
            # Find complete job data from database
            complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
            
            job_result = JobResult(
                ncspjobid=job_data["ncspjobid"],
                title=job_data["title"],
                match_percentage=job_data["match_percentage"],
                similarity_score=next((j.get("similarity") for j in similar_jobs if j["ncspjobid"] == job_data["ncspjobid"]), None),
                keywords=complete_job.get("keywords"),
                description=complete_job.get("description"),
                date=complete_job.get("date"),
                organizationid=complete_job.get("organizationid"),
                organization_name=complete_job.get("organization_name"),
                numberofopenings=complete_job.get("numberofopenings"),
                industryname=complete_job.get("industryname"),
                sectorname=complete_job.get("sectorname"),
                functionalareaname=complete_job.get("functionalareaname"),
                functionalrolename=complete_job.get("functionalrolename"),
                aveexp=complete_job.get("aveexp"),
                avewage=complete_job.get("avewage"),
                gendercode=complete_job.get("gendercode"),
                highestqualification=complete_job.get("highestqualification"),
                statename=complete_job.get("statename"),
                districtname=complete_job.get("districtname"),
                keywords_matched=job_data.get("keywords_matched"),
                keywords_unmatched=job_data.get("keywords_unmatched"),
                user_skills_matched=job_data.get("user_skills_matched"),
                keyword_match_score=job_data.get("keyword_matches")
            )
            job_results.append(job_result)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        logger.info(f"Job search completed: {len(job_results)} jobs returned in {processing_time_ms}ms")
        
        return JobSearchResponse(
            jobs=job_results,
            query_skills=request.skills,
            total_found=len(ranked_jobs),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job search failed: {e}")
        raise HTTPException(status_code=500, detail="Job search failed")
    
@app.post("/recommend_courses", response_model=CourseRecommendationResponse)
async def recommend_courses(request: CourseRecommendationRequest) -> CourseRecommendationResponse:
    """
    Get course recommendations for unmatched keywords
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Course recommendation request for {len(request.keywords_unmatched)} keywords: {request.keywords_unmatched}")
        
        # Get course recommendations from GPT
        recommendations_data = await course_service.get_course_recommendations(request.keywords_unmatched)
        
        # Convert to response format
        course_recommendations = []
        for course_data in recommendations_data:
            try:
                course_rec = CourseRecommendation(
                    course_name=course_data.get("course_name", "Unknown Course"),
                    platform=course_data.get("platform", "Unknown Platform"),
                    duration=course_data.get("duration", "Unknown Duration"),
                    link=course_data.get("link", ""),
                    educator=course_data.get("educator", "Unknown Educator"),
                    skill_covered=course_data.get("skill_covered", ""),
                    difficulty_level=course_data.get("difficulty_level"),
                    rating=course_data.get("rating")
                )
                course_recommendations.append(course_rec)
            except Exception as e:
                logger.error(f"Error processing course recommendation: {e}")
                continue
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        logger.info(f"Course recommendations completed: {len(course_recommendations)} courses returned in {processing_time_ms}ms")
        
        return CourseRecommendationResponse(
            recommendations=course_recommendations,
            keywords_processed=request.keywords_unmatched,
            total_recommendations=len(course_recommendations),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Course recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Course recommendation failed")


#New Endpoint added

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Enhanced chat endpoint with location awareness"""
    try:
        response = await enhanced_chat_service.handle_chat_message(request)
        return response
    except Exception as e:
        logger.error(f"Enhanced chat endpoint failed: {e}")
        return ChatResponse(
            response="I'm here to help you find jobs! You can tell me your skills or ask about jobs in specific locations like 'Mumbai jobs'.",
            message_type="text",
            chat_phase="profile_building",
            suggestions=["My skills are...", "Jobs in Mumbai", "Remote work", "Entry level"]
        )

@app.post("/upload_cv", response_model=CVAnalysisResponse)
async def upload_cv_enhanced(cv_file: UploadFile = File(...)) -> CVAnalysisResponse:
    """Enhanced CV upload with complete analysis and job matching"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.png', '.jpg', '.jpeg'}
        file_ext = '.' + cv_file.filename.lower().split('.')[-1]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"Processing CV upload: {cv_file.filename}")
        
        # Read file content
        file_content = await cv_file.read()
        
        # Process CV using enhanced processor
        cv_profile = await cv_processor.process_cv(file_content, cv_file.filename)
        
        # Get job search text
        search_text = cv_processor.get_job_search_text(cv_profile)
        
        jobs_found = []
        total_jobs = 0
        
        # Perform job search if we have meaningful skills
        if cv_profile.skills and len(cv_profile.skills) >= 2:
            try:
                # Generate embedding for combined profile text
                profile_embedding = await embedding_service.get_embedding(search_text)
                
                # Search similar jobs
                similar_jobs = await vector_store.search_similar_jobs(
                    profile_embedding, 
                    top_k=30
                )
                
                if similar_jobs:
                    # Re-rank jobs using GPT with extracted skills
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    
                    # Get complete job details for top matches
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[:10]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    # Format job results
                    for job_data in ranked_jobs[:10]:
                        complete_job = next(
                            (j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), 
                            {}
                        )
                        
                        if complete_job:
                            jobs_found.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0),
                                "keywords": complete_job.get("keywords", ""),
                                "functionalrolename": complete_job.get("functionalrolename", ""),
                                "industryname": complete_job.get("industryname", ""),
                                "skills_matched": job_data.get("keywords_matched", []),
                                "similarity_score": job_data.get("similarity_used", 0)
                            })
                    
                    total_jobs = len(ranked_jobs)
                    
            except Exception as job_search_error:
                logger.error(f"Job search failed during CV processing: {job_search_error}")
                # Continue without job results
        
        # Generate processing recommendations
        recommendations = _generate_cv_recommendations(cv_profile)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Success response
        success_message = f"✅ Successfully processed your CV! "
        
        if jobs_found:
            success_message += f"Found {len(jobs_found)} matching jobs with {len(cv_profile.skills)} extracted skills."
        else:
            success_message += f"Extracted {len(cv_profile.skills)} skills. Try refining your CV for better job matches."
        
        logger.info(f"CV processing completed: {cv_profile.confidence_score} confidence, "
                   f"{len(jobs_found)} jobs, {processing_time_ms}ms")
        
        return CVAnalysisResponse(
            success=True,
            message=success_message,
            profile=cv_processor.to_dict(cv_profile),
            jobs=jobs_found,
            total_jobs_found=total_jobs,
            processing_time_ms=processing_time_ms,
            confidence_score=cv_profile.confidence_score,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV upload processing failed: {e}")
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return CVAnalysisResponse(
            success=False,
            message=f"Failed to process CV: {str(e)}",
            processing_time_ms=processing_time_ms
        )

def _generate_cv_recommendations(profile: CVProfile) -> List[str]:
    """Generate recommendations based on CV analysis"""
    recommendations = []
    
    # Skills recommendations
    if len(profile.skills) < 5:
        recommendations.append("💡 Add more technical skills to improve job matching")
    
    # Contact information
    if not profile.email:
        recommendations.append("📧 Add contact email for better profile completeness")
    
    if not profile.phone:
        recommendations.append("📱 Include phone number in your CV")
    
    # Experience recommendations
    if len(profile.experience) == 0:
        recommendations.append("💼 Add work experience details for better job matching")
    elif len(profile.experience) < 2:
        recommendations.append("💼 Include more work experience entries if available")
    
    # Education recommendations
    if len(profile.education) == 0:
        recommendations.append("🎓 Add education background to strengthen your profile")
    
    # Summary recommendations
    if not profile.summary:
        recommendations.append("📝 Add a professional summary to highlight your strengths")
    
    # Confidence-based recommendations
    if profile.confidence_score < 0.5:
        recommendations.append("⚡ Consider adding more detailed information to improve CV quality")
    elif profile.confidence_score < 0.7:
        recommendations.append("📈 Good CV structure! Add a few more details for optimal results")
    
    # Skills gap analysis
    high_demand_skills = [
        "Python", "JavaScript", "React", "Node.js", "AWS", "Docker", 
        "Kubernetes", "SQL", "MongoDB", "Git", "CI/CD"
    ]
    
    missing_skills = [skill for skill in high_demand_skills 
                     if skill.lower() not in [s.lower() for s in profile.skills]]
    
    if missing_skills:
        recommendations.append(f"🚀 Consider learning in-demand skills: {', '.join(missing_skills[:3])}")
    
    return recommendations[:5]  # Limit to top 5 recommendations

# Add this new endpoint for CV analysis without job search
@app.post("/analyze_cv", response_model=CVAnalysisResponse)
async def analyze_cv_only(cv_file: UploadFile = File(...)) -> CVAnalysisResponse:
    """Analyze CV structure and extract data without job matching"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.png', '.jpg', '.jpeg'}
        file_ext = '.' + cv_file.filename.lower().split('.')[-1]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"Analyzing CV: {cv_file.filename}")
        
        # Read and process CV
        file_content = await cv_file.read()
        cv_profile = await cv_processor.process_cv(file_content, cv_file.filename)
        
        # Generate recommendations
        recommendations = _generate_cv_recommendations(cv_profile)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        analysis_message = f"📊 CV Analysis Complete! Extracted {len(cv_profile.skills)} skills, "
        analysis_message += f"{len(cv_profile.experience)} experience entries, "
        analysis_message += f"confidence score: {cv_profile.confidence_score:.1%}"
        
        return CVAnalysisResponse(
            success=True,
            message=analysis_message,
            profile=cv_processor.to_dict(cv_profile),
            processing_time_ms=processing_time_ms,
            confidence_score=cv_profile.confidence_score,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"CV analysis failed: {e}")
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return CVAnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            processing_time_ms=processing_time_ms
        )

@app.post("/upload_cv_chat", response_model=ChatResponse)
async def upload_cv_for_chat(cv_file: UploadFile = File(...)) -> ChatResponse:
    """Upload CV and get chat-style response with job matches"""
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Process CV using the enhanced processor
        file_content = await cv_file.read()
        cv_profile = await cv_processor.process_cv(file_content, cv_file.filename)
        
        # Generate chat response with job matches using CV service
        chat_response = await cv_chat_service.handle_cv_upload_chat(cv_profile)
        
        return chat_response
        
    except Exception as e:
        logger.error(f"CV chat upload failed: {e}")
        return ChatResponse(
            response="I had trouble processing your CV. Let's build your profile by chatting about your skills!",
            message_type="text",
            chat_phase="profile_building"
        )

@app.post("/chat_with_cv", response_model=ChatResponse)
async def chat_with_cv_context(request: ChatWithCVRequest) -> ChatResponse:
    """Handle chat with CV context for follow-up questions"""
    try:
        if request.cv_profile_data:
            # Convert dict back to CVProfile if needed
            cv_profile = CVProfile(**request.cv_profile_data)
            
            # Create a ChatRequest for the CV service
            chat_request = ChatRequest(
                message=request.message,
                chat_phase=request.chat_phase,
                user_profile=request.user_profile,
                conversation_history=request.conversation_history
            )
            
            return await enhanced_chat_service.handle_cv_followup_chat(request, cv_profile)
        else:
            # Fall back to regular chat
            chat_request = ChatRequest(
                message=request.message,
                chat_phase=request.chat_phase,
                user_profile=request.user_profile,
                conversation_history=request.conversation_history
            )
            response = await simple_chat_service.handle_chat_message(chat_request)
        
        return response
    except Exception as e:
        logger.error(f"Chat with CV context failed: {e}")
        return ChatResponse(
            response="I can help you find jobs. What would you like to do?",
            message_type="text",
            chat_phase="job_searching"
        )
    

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8888))
    host = os.getenv("HOST", "0.0.0.0")
 
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
