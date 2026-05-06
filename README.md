# HR AI Recommendation Service

An intelligent AI-powered service for analyzing HR activities and matching candidates with optimal opportunities. Built with FastAPI and leveraging LLM capabilities for enhanced HR decision-making.

## Overview

The HR AI Recommendation Service is designed to:

- **Analyze Activities**: Classify HR activities (training, certifications, projects, etc.) by scenario and domain
- **Match Candidates**: Evaluate candidate fit for specific activities with detailed insights
- **Provide Recommendations**: Generate actionable recommendations based on AI analysis and rule-based heuristics
- **Ensure Quality**: Flag risks and identify development areas for candidates

## Features

### 1. Activity Analysis (`/analyze-activity`)
Analyzes internal company activities to determine:
- **Scenario Classification**: Identifies the business scenario using zero-shot classification and rule-based analysis
  - UPSKILLING_TRAINING
  - CERTIFICATION_PREPARATION
  - PROJECT_STAFFING
  - EXPERT_MISSION
  - AUDIT_SUPPORT
  - COMPLIANCE_TRAINING

- **Domain Classification**: Detects the primary business domain
  - Cloud DevOps
  - Frontend Development
  - Backend Development
  - Data AI
  - Finance
  - Human Resources
  - Sales
  - Operations
  - Cybersecurity

- **Skill Profiling**: Separates required and preferred skills
- **Enrichment Data**: Uses LLM to extract:
  - Activity goals and objectives
  - Target seniority level
  - Risk level assessment
  - Ideal candidate profile
  - Quality warnings

### 2. Candidate Fit Analysis (`/analyze-candidate-fit`)
Evaluates how well a candidate matches an activity by analyzing:
- **Skill Alignment**: Matches candidate skills against activity requirements
- **Experience Relevance**: Considers years of experience and current position
- **Department Fit**: Assesses cross-departmental alignment
- **Risk Assessment**: Identifies potential risks and gaps
- **Development Opportunities**: Suggests areas for professional growth
- **Recommendation**: Provides action guidance:
  - RECOMMEND
  - CONSIDER_WITH_GAP
  - BACKUP_ONLY
  - NOT_RECOMMENDED

### 3. Health Checks
- `/health`: Service availability check
- `/llm-health`: LLM (Ollama) availability and connectivity check

## Architecture

### Components

- **FastAPI Framework**: RESTful API with automatic documentation
- **Pydantic Models**: Type-safe request/response validation
- **Zero-Shot Classification**: Hugging Face transformer pipeline for scenario and domain classification
- **LLM Integration**: Ollama-based language model for enriched insights
- **Rule-Based Engine**: Fallback logic for classification when confidence is low

### Processing Pipeline

```
1. Input Validation (Pydantic)
   ↓
2. Rule-Based Analysis
   ↓
3. Zero-Shot Classification
   ↓
4. LLM Enrichment
   ↓
5. Output Validation & Return
```

## Technical Stack

- **Runtime**: Python 3.8+
- **Framework**: FastAPI
- **ML Models**: 
  - Transformers (valhalla/distilbart-mnli-12-3)
  - Ollama with Llama3.2
- **HTTP Client**: httpx (async)
- **Type System**: Pydantic

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama running locally on `http://localhost:11434`
- Required Python packages

### Setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

2. **Install dependencies**:
```bash
pip install fastapi pydantic httpx transformers torch
```

3. **Install and run Ollama**:
   - Download from [ollama.ai](https://ollama.ai)
   - Run the Ollama service: `ollama serve`
   - Pull the required model: `ollama pull llama3.2`

## Usage

### Starting the Service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

### API Documentation

Access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example: Analyze Activity

```bash
curl -X POST "http://localhost:8000/analyze-activity?debug=false" \
  -H "Content-Type: application/json" \
  -d {
    "title": "Cloud Architecture Training",
    "description": "Advanced training on Kubernetes and Docker",
    "type": "TRAINING",
    "context": "UPSKILLING",
    "department": "Engineering",
    "skills": ["Docker", "Kubernetes", "Linux Advanced", "DevOps Practices"]
  }
```

### Example: Analyze Candidate Fit

```bash
curl -X POST "http://localhost:8000/analyze-candidate-fit?debug=false" \
  -H "Content-Type: application/json" \
  -d {
    "activity": {
      "title": "Cloud Architecture Training",
      "scenario": "UPSKILLING_TRAINING",
      "mainDomain": "Cloud DevOps",
      "type": "TRAINING",
      "context": "UPSKILLING",
      "requiredSkills": ["Docker", "Kubernetes"],
      "preferredSkills": ["Linux", "DevOps Practices"]
    },
    "candidate": {
      "fullName": "John Doe",
      "departmentName": "Engineering",
      "position": "Backend Developer",
      "experienceYears": 5,
      "skills": [
        {"name": "Docker", "level": "Expert", "dynamicScore": 0.9},
        {"name": "Python", "level": "Expert", "dynamicScore": 0.95}
      ],
      "features": {
        "readinessLabel": "READY",
        "skillCoverage": 0.8,
        "mandatorySkillCoverage": 1.0,
        "departmentFit": 0.9,
        "profileQuality": 0.85
      }
    }
  }
```

## Configuration

### Environment Variables

- `OLLAMA_URL`: Ollama API endpoint (default: `http://localhost:11434/api/generate`)
- `OLLAMA_MODEL`: Model to use (default: `llama3.2`)

### Model Configuration

- **Zero-Shot Classifier**: `valhalla/distilbart-mnli-12-3`
- **Device**: CPU by default (device=-1), configure in `get_classifier()` for GPU

## Response Examples

### Activity Analysis Response

```json
{
  "scenario": "UPSKILLING_TRAINING",
  "aiScenario": "UPSKILLING_TRAINING",
  "aiScenarioConfidence": 0.892,
  "mainDomain": "Cloud DevOps",
  "aiDomain": "Cloud DevOps",
  "aiDomainConfidence": 0.956,
  "requiredSkills": ["Docker", "Kubernetes"],
  "preferredSkills": ["Linux", "DevOps Practices"],
  "flags": {
    "availabilityRequired": true,
    "departmentStrict": false,
    "mandatorySkillCheck": true,
    "goodForJuniors": true,
    "experienceImportant": false
  },
  "activityGoal": "Equip team with modern containerization and orchestration skills",
  "seniorityTarget": "JUNIOR_TO_MID",
  "riskLevel": "LOW",
  "recommendedCandidateProfile": "Developer with interest in DevOps and cloud technologies",
  "qualityWarnings": [],
  "llmStatus": "ok"
}
```

### Candidate Fit Response

```json
{
  "fitSummary": "Strong candidate with excellent Docker expertise and relevant DevOps background",
  "riskFlags": [],
  "developmentAreas": ["Kubernetes advanced patterns"],
  "recommendedAction": "RECOMMEND",
  "aiStatus": "ok"
}
```

## Error Handling

The service provides graceful error handling:

- **Validation Errors** (400): Invalid request format
- **Processing Errors** (500): Issues with classification or LLM
- **Fallback Mechanism**: If LLM is unavailable, returns basic analysis with `llmStatus: "fallback"`
- **Debug Mode**: Use `?debug=true` to get detailed error information

## Development

### Project Structure

```
hr-ai-recommendation-service/
├── main.py                 # Main application file with all endpoints
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Adding New Features

1. **New Scenario Types**: Add to `SCENARIO_LABELS` and update `rule_based_analysis()`
2. **New Domains**: Add to `DOMAIN_LABELS` and `domain_keywords` in `infer_domain_by_rules()`
3. **New Endpoints**: Follow the FastAPI patterns for POST endpoints with Pydantic models

## Performance Considerations

- **Caching**: The zero-shot classifier is cached in memory after first use
- **Timeouts**: LLM calls default to 60-second timeout
- **Async Processing**: All I/O operations are async for better concurrency
- **Batch Processing**: Consider batch analysis for large candidate pools

## Troubleshooting

### LLM Connection Issues
```bash
# Check Ollama service status
curl http://localhost:11434/api/tags

# Ensure Llama3.2 is installed
ollama pull llama3.2
```

### Import Errors
```bash
# Reinstall transformers and dependencies
pip install --upgrade transformers torch
```

### Zero-Shot Classifier Not Loading
- Ensure internet connection for model download on first run
- Check disk space for model storage (2-3 GB required)

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze-activity` | Analyze an HR activity |
| POST | `/analyze-candidate-fit` | Evaluate candidate fit for an activity |
| GET | `/health` | Service health check |
| GET | `/llm-health` | LLM connectivity check |
| GET | `/docs` | Swagger UI documentation |
| GET | `/redoc` | ReDoc documentation |

## Future Enhancements

- [ ] Batch processing endpoint for multiple candidates
- [ ] Caching layer for frequently analyzed activities
- [ ] Custom scoring models per domain
- [ ] Integration with external HR systems
- [ ] WebSocket support for real-time updates
- [ ] Multi-language support

## License

[Add your license information here]

## Support

For issues or questions, please contact the development team or create an issue in the repository.
