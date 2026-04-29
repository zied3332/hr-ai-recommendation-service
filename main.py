from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import re
import traceback
import httpx

app = FastAPI(title="HR AI Layer Service")

classifier = None

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

SCENARIO_LABELS = [
    "UPSKILLING_TRAINING",
    "CERTIFICATION_PREPARATION",
    "PROJECT_STAFFING",
    "EXPERT_MISSION",
    "AUDIT_SUPPORT",
    "COMPLIANCE_TRAINING",
]

DOMAIN_LABELS = [
    "Cloud DevOps",
    "Frontend Development",
    "Backend Development",
    "Data AI",
    "Finance",
    "Human Resources",
    "Sales",
    "Operations",
    "Cybersecurity",
]


class ActivityInput(BaseModel):
    title: str
    description: Optional[str] = ""
    type: str
    context: Optional[str] = ""
    department: Optional[str] = ""
    skills: List[str] = Field(default_factory=list)


class CandidateFitActivityInput(BaseModel):
    title: str
    scenario: str
    mainDomain: str
    type: Optional[str] = ""
    context: Optional[str] = ""
    requiredSkills: List[str] = Field(default_factory=list)
    preferredSkills: List[str] = Field(default_factory=list)


class CandidateFitCandidateInput(BaseModel):
    fullName: str
    departmentName: Optional[str] = ""
    position: Optional[str] = ""
    experienceYears: Optional[float] = None
    skills: List[Dict[str, Any]] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)


class CandidateFitInput(BaseModel):
    activity: CandidateFitActivityInput
    candidate: CandidateFitCandidateInput


def get_classifier():
    global classifier

    if classifier is None:
        from transformers import pipeline

        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1,
        )

    return classifier


def build_activity_text(activity: ActivityInput) -> str:
    return f"""
    Internal company activity.

    Title: {activity.title}
    Description: {activity.description}
    Type: {activity.type}
    Context: {activity.context}
    Department: {activity.department}
    Skills: {", ".join(activity.skills)}

    Identify the business scenario and domain.
    """


def rule_based_analysis(activity: ActivityInput) -> Dict[str, Any]:
    text = f"{activity.title} {activity.description} {activity.type} {activity.context}".lower()

    activity_type = activity.type.upper().strip()
    activity_context = activity.context.upper().strip()

    flags = {
        "availabilityRequired": True,
        "departmentStrict": False,
        "mandatorySkillCheck": True,
        "goodForJuniors": False,
        "experienceImportant": False,
    }

    scenario = None

    if activity_type == "TRAINING" and activity_context == "UPSKILLING":
        scenario = "UPSKILLING_TRAINING"
        flags["goodForJuniors"] = True
    elif activity_type == "CERTIFICATION":
        scenario = "CERTIFICATION_PREPARATION"
    elif activity_type in ["PROJECT", "MISSION"]:
        scenario = "PROJECT_STAFFING"
        flags["experienceImportant"] = True

    if "audit" in text:
        scenario = "AUDIT_SUPPORT"

    if "compliance" in text or "policy" in text:
        scenario = "COMPLIANCE_TRAINING"

    return {"ruleScenario": scenario, "flags": flags}


def infer_domain_by_rules(activity: ActivityInput) -> Optional[str]:
    text = f"""
    {activity.title}
    {activity.description}
    {activity.department}
    {" ".join(activity.skills)}
    """.lower()

    finance_keywords = [
        "finance", "financial", "investment", "investments", "portfolio",
        "risk assessment", "risk management", "financial statements",
        "accounting", "budget", "tax", "revenue", "cost", "profit", "loss",
        "excel advanced",
    ]

    if any(keyword in text for keyword in finance_keywords):
        return "Finance"

    domain_keywords = {
        "Cloud DevOps": ["docker", "kubernetes", "linux", "devops", "cloud", "deployment", "container", "aws", "azure", "gcp"],
        "Frontend Development": ["react", "angular", "vue", "html", "css", "javascript", "typescript", "frontend"],
        "Backend Development": ["node", "nestjs", "spring", "java", "api", "database", "sql", "mongodb", "backend"],
        "Data AI": ["python", "machine learning", "artificial intelligence", "analytics", "forecasting", "tensorflow", "pandas", "model"],
        "Finance": finance_keywords,
        "Human Resources": ["hr", "human resources", "recruitment", "employee onboarding", "performance review"],
        "Sales": ["sales", "crm", "customer", "lead", "prospect", "negotiation"],
        "Operations": ["operations", "logistics", "process", "supply chain", "workflow"],
        "Cybersecurity": ["security", "cybersecurity", "firewall", "vulnerability", "authentication", "encryption"],
    }

    scores = {domain: 0 for domain in domain_keywords}

    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in text:
                scores[domain] += 1

    best_domain = max(scores, key=scores.get)

    if scores[best_domain] > 0:
        return best_domain

    return None


def split_skills(skills: List[str]) -> Dict[str, List[str]]:
    required = []
    preferred = []

    for skill in skills:
        clean = skill.strip()
        if not clean:
            continue

        lowered = clean.lower()

        if any(word in lowered for word in ["advanced", "expert", "required", "mandatory"]):
            required.append(clean)
        else:
            preferred.append(clean)

    if not required and preferred:
        required = preferred[:2]
        preferred = preferred[2:]

    return {
        "requiredSkills": required,
        "preferredSkills": preferred,
    }


def parse_json_safely(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    if not raw:
        raise ValueError("Ollama returned empty response")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:300]}")

    return json.loads(match.group(0))


def validate_activity_llm_output(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed_seniority = {
        "JUNIOR",
        "JUNIOR_TO_MID",
        "MID",
        "MID_TO_SENIOR",
        "SENIOR",
        "ANY",
    }
    allowed_risk = {"LOW", "MEDIUM", "HIGH"}

    seniority = str(data.get("seniorityTarget", "ANY")).strip().upper()
    risk = str(data.get("riskLevel", "LOW")).strip().upper()

    if seniority not in allowed_seniority:
        seniority = "ANY"

    if risk not in allowed_risk:
        risk = "LOW"

    warnings = data.get("qualityWarnings", [])
    if not isinstance(warnings, list):
        warnings = []

    return {
        "activityGoal": str(data.get("activityGoal", "")).strip(),
        "seniorityTarget": seniority,
        "riskLevel": risk,
        "recommendedCandidateProfile": str(data.get("recommendedCandidateProfile", "")).strip(),
        "qualityWarnings": [str(w).strip() for w in warnings if str(w).strip()][:5],
    }


def validate_candidate_fit_output(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed_actions = {
        "RECOMMEND",
        "CONSIDER_WITH_GAP",
        "BACKUP_ONLY",
        "NOT_RECOMMENDED",
    }

    action = str(data.get("recommendedAction", "CONSIDER_WITH_GAP")).strip().upper()
    if action not in allowed_actions:
        action = "CONSIDER_WITH_GAP"

    def clean_list(value):
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()][:5]

    return {
        "fitSummary": str(data.get("fitSummary", "")).strip(),
        "riskFlags": clean_list(data.get("riskFlags", [])),
        "developmentAreas": clean_list(data.get("developmentAreas", [])),
        "recommendedAction": action,
    }


async def call_ollama_json(prompt: str, timeout: float = 60.0) -> Dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        raw = response.json().get("response", "")
        return parse_json_safely(raw)


async def extract_llm_enrichment(
    activity: ActivityInput,
    scenario: str,
    domain: str,
) -> Dict[str, Any]:
    prompt = f"""
You are an HR recommendation assistant.

Return ONLY valid JSON. No markdown. No explanation.

Analyze this activity:

Title: {activity.title}
Description: {activity.description}
Type: {activity.type}
Context: {activity.context}
Department: {activity.department}
Detected scenario: {scenario}
Detected domain: {domain}
Skills: {", ".join(activity.skills)}

Return this exact JSON structure:
{{
  "activityGoal": "short clear goal",
  "seniorityTarget": "JUNIOR | JUNIOR_TO_MID | MID | MID_TO_SENIOR | SENIOR | ANY",
  "riskLevel": "LOW | MEDIUM | HIGH",
  "recommendedCandidateProfile": "short description of ideal employee profile",
  "qualityWarnings": ["warning 1", "warning 2"]
}}

Rules:
- qualityWarnings should mention weak or unrelated skills.
- If everything is good, qualityWarnings should be [].
- Do not change the scenario or domain.
"""

    parsed = await call_ollama_json(prompt)
    return validate_activity_llm_output(parsed)


async def extract_candidate_fit_insights(payload: CandidateFitInput) -> Dict[str, Any]:
    activity = payload.activity
    candidate = payload.candidate

    skill_lines = []
    for skill in candidate.skills[:12]:
        skill_lines.append(
            f"- {skill.get('name', 'Unknown')} | level={skill.get('level', '')} | score={skill.get('dynamicScore', 0)}"
        )

    features = candidate.features or {}

    prompt = f"""
You are an HR AI assistant analyzing candidate fit.

Return ONLY valid JSON. No markdown. No explanation.

Activity:
- Title: {activity.title}
- Scenario: {activity.scenario}
- Domain: {activity.mainDomain}
- Type: {activity.type}
- Context: {activity.context}
- Required skills: {", ".join(activity.requiredSkills)}
- Preferred skills: {", ".join(activity.preferredSkills)}

Candidate:
- Name: {candidate.fullName}
- Department: {candidate.departmentName}
- Position: {candidate.position}
- Experience years: {candidate.experienceYears}
- Skills:
{chr(10).join(skill_lines)}

Computed numeric features:
- readinessLabel: {features.get("readinessLabel")}
- skillCoverage: {features.get("skillCoverage")}
- mandatorySkillCoverage: {features.get("mandatorySkillCoverage")}
- departmentFit: {features.get("departmentFit")}
- profileQuality: {features.get("profileQuality")}
- matchedMandatorySkills: {features.get("matchedMandatorySkills")}
- missingMandatorySkills: {features.get("missingMandatorySkills")}
- matchedPreferredSkills: {features.get("matchedPreferredSkills")}
- missingPreferredSkills: {features.get("missingPreferredSkills")}

Return this exact JSON structure:
{{
  "fitSummary": "short explanation of candidate fit",
  "riskFlags": ["risk 1", "risk 2"],
  "developmentAreas": ["area 1", "area 2"],
  "recommendedAction": "RECOMMEND | CONSIDER_WITH_GAP | BACKUP_ONLY | NOT_RECOMMENDED"
}}

Rules:
- Do not invent skills the candidate does not have.
- If mandatorySkillCoverage is 0, do not recommend.
- If readinessLabel is INSUFFICIENT_DATA, recommendedAction should be NOT_RECOMMENDED.
- If missingMandatorySkills is not empty, mention them in riskFlags.
- Keep answers short and useful for HR.
"""

    parsed = await call_ollama_json(prompt, timeout=60.0)
    return validate_candidate_fit_output(parsed)


@app.post("/analyze-activity")
async def analyze_activity(activity: ActivityInput, debug: bool = Query(False)):
    text = build_activity_text(activity)

    rules = rule_based_analysis(activity)
    rule_domain = infer_domain_by_rules(activity)
    skill_profile = split_skills(activity.skills)

    try:
        model = get_classifier()

        scenario_prediction = model(text, candidate_labels=SCENARIO_LABELS)
        domain_prediction = model(text, candidate_labels=DOMAIN_LABELS)

        ai_scenario = scenario_prediction["labels"][0]
        ai_scenario_confidence = float(scenario_prediction["scores"][0])

        ai_domain = domain_prediction["labels"][0]
        ai_domain_confidence = float(domain_prediction["scores"][0])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Zero-shot classifier failed: {type(e).__name__}: {str(e)}",
        )

    if rules["ruleScenario"]:
        final_scenario = rules["ruleScenario"]
    elif ai_scenario_confidence >= 0.5:
        final_scenario = ai_scenario
    else:
        final_scenario = "UNKNOWN"

    if rule_domain:
        main_domain = rule_domain
    elif ai_domain_confidence >= 0.5:
        main_domain = ai_domain
    else:
        main_domain = "UNKNOWN"

    llm_status = "skipped"
    llm_error = None
    llm_enrichment = {
        "activityGoal": "",
        "seniorityTarget": "ANY",
        "riskLevel": "LOW",
        "recommendedCandidateProfile": "",
        "qualityWarnings": [],
    }

    try:
        llm_enrichment = await extract_llm_enrichment(activity, final_scenario, main_domain)
        llm_status = "ok"
    except Exception as e:
        print("LLM ERROR:", traceback.format_exc())
        llm_status = "fallback"
        llm_error = f"{type(e).__name__}: {str(e)}"

    response = {
        "scenario": final_scenario,
        "aiScenario": ai_scenario,
        "aiScenarioConfidence": round(ai_scenario_confidence, 3),
        "mainDomain": main_domain,
        "aiDomain": ai_domain,
        "aiDomainConfidence": round(ai_domain_confidence, 3),
        "requiredSkills": skill_profile["requiredSkills"],
        "preferredSkills": skill_profile["preferredSkills"],
        "flags": rules["flags"],
        "activityGoal": llm_enrichment["activityGoal"],
        "seniorityTarget": llm_enrichment["seniorityTarget"],
        "riskLevel": llm_enrichment["riskLevel"],
        "recommendedCandidateProfile": llm_enrichment["recommendedCandidateProfile"],
        "qualityWarnings": llm_enrichment["qualityWarnings"],
        "llmStatus": llm_status,
    }

    if debug:
        response["debug"] = {
            "ruleScenario": rules["ruleScenario"],
            "ruleDomain": rule_domain,
            "scenarioRanking": scenario_prediction,
            "domainRanking": domain_prediction,
            "llmError": llm_error,
        }

    return response


@app.post("/analyze-candidate-fit")
async def analyze_candidate_fit(payload: CandidateFitInput, debug: bool = Query(False)):
    ai_status = "skipped"
    ai_error = None

    fallback = {
        "fitSummary": "AI insight unavailable. Use numeric Layer 2 features for decision.",
        "riskFlags": [],
        "developmentAreas": payload.candidate.features.get("missingMandatorySkills", []),
        "recommendedAction": "CONSIDER_WITH_GAP",
        "aiStatus": "fallback",
    }

    try:
        result = await extract_candidate_fit_insights(payload)
        result["aiStatus"] = "ok"

        if debug:
            result["debug"] = {
                "activityTitle": payload.activity.title,
                "candidateName": payload.candidate.fullName,
            }

        return result

    except Exception as e:
        print("CANDIDATE FIT LLM ERROR:", traceback.format_exc())
        ai_status = "fallback"
        ai_error = f"{type(e).__name__}: {str(e)}"

        if debug:
            fallback["debug"] = {
                "aiStatus": ai_status,
                "aiError": ai_error,
            }

        return fallback


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/llm-health")
async def llm_health():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": "Return JSON only: {\"status\":\"ok\"}",
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()

            raw = response.json().get("response", "")
            parsed = parse_json_safely(raw)

        return {"status": "ok", "model": OLLAMA_MODEL, "response": parsed}

    except Exception as e:
        return {
            "status": "unavailable",
            "error": f"{type(e).__name__}: {str(e)}",
        }