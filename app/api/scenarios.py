"""
Scenario API endpoints for conversation practice scenarios.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.database import get_db
from app.auth import get_current_user
from app.schemas.scenario import (
    GenerateFromProjectsRequest,
    ManualCreateRequest,
    ScenarioResponse,
    ModifyWithChatRequest,
    ModifyWithChatResponse
)
from app.services.scenario_service import ScenarioService
from app.models.scenario import Scenario

logger = logging.getLogger(__name__)

router = APIRouter()
scenario_service = ScenarioService()
security = HTTPBearer()


@router.post("/generate-from-projects", response_model=dict)
async def generate_from_projects(
    request: GenerateFromProjectsRequest,
    user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Generate scenarios from projects and schedules using GPT-4o.

    This endpoint:
    1. Fetches project and schedule information from Java backend
    2. Uses GPT-4o to generate realistic business conversation scenarios
    3. Saves scenarios to PostgreSQL database

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])
        jwt_token = credentials.credentials

        logger.info(f"🚀 Scenario generation request: user={user_id}, projects={len(request.projectIds)}, schedules={len(request.scheduleIds)}, documents={len(request.documentIds)}")

        scenarios = await scenario_service.generate_from_projects(
            project_ids=request.projectIds,
            schedule_ids=request.scheduleIds,
            document_ids=request.documentIds,
            language=request.language,
            difficulty=request.difficulty,
            count=request.count,
            user_id=user_id,
            jwt_token=jwt_token,
            db=db
        )

        return {
            "success": True,
            "message": f"Successfully generated {len(scenarios)} scenarios",
            "data": scenarios
        }

    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Scenario generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario generation failed: {str(e)}")


@router.post("/create", response_model=dict)
async def create_manual_scenario(
    request: ManualCreateRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a manual scenario.

    User provides all scenario details manually without AI generation.

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])

        logger.info(f"📝 Manual scenario creation: user={user_id}, title={request.title}")

        scenario = await scenario_service.create_manual(
            user_id=user_id,
            title=request.title,
            description=request.description,
            scenario_text=request.scenarioText,
            category=request.category,
            roles=request.roles.dict(),
            required_terminology=request.requiredTerminology,
            language=request.language,
            difficulty=request.difficulty,
            project_id=request.projectId,
            schedule_id=request.scheduleId,
            db=db
        )

        return {
            "success": True,
            "message": "Scenario created successfully",
            "data": scenario
        }

    except Exception as e:
        logger.error(f"❌ Manual scenario creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario creation failed: {str(e)}")


@router.get("", response_model=dict)
async def get_scenarios(
    user: dict = Depends(get_current_user),
    language: Optional[str] = Query(None, description="Filter by language (en, ko, zh, ja)"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (beginner, intermediate, advanced)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    schedule_ids: Optional[str] = Query(None, description="Comma-separated schedule IDs"),
    project_ids: Optional[str] = Query(None, description="Comma-separated project IDs"),
    limit: int = Query(20, ge=1, le=100, description="Number of scenarios to return"),
    offset: int = Query(0, ge=0, description="Number of scenarios to skip"),
    db: Session = Depends(get_db)
):
    """
    Get user's scenarios with optional filters.

    Returns scenarios ordered by creation date (newest first).

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])

        # Build query
        query = db.query(Scenario).filter(Scenario.user_id == user_id)

        # Apply filters
        if language:
            query = query.filter(Scenario.language == language)
        if difficulty:
            query = query.filter(Scenario.difficulty == difficulty)
        if category:
            query = query.filter(Scenario.category == category)

        # Multiple schedule IDs (OR condition)
        if schedule_ids:
            schedule_id_list = [sid.strip() for sid in schedule_ids.split(',')]
            from sqlalchemy import or_
            filters = [Scenario.schedule_ids.contains([sid]) for sid in schedule_id_list]
            query = query.filter(or_(*filters))

        # Multiple project IDs (OR condition)
        if project_ids:
            project_id_list = [pid.strip() for pid in project_ids.split(',')]
            from sqlalchemy import or_
            filters = [Scenario.project_ids.contains([pid]) for pid in project_id_list]
            query = query.filter(or_(*filters))

        # Get total count
        total = query.count()

        # Apply pagination and ordering
        scenarios = query.order_by(Scenario.created_at.desc()).limit(limit).offset(offset).all()

        # Convert to response format
        result = []
        for s in scenarios:
            result.append({
                "id": str(s.id),
                "title": s.title,
                "description": s.description,
                "scenarioText": s.scenario_text,
                "language": s.language,
                "difficulty": s.difficulty,
                "category": s.category,
                "roles": s.roles,
                "requiredTerminology": s.required_terminology,
                "projectIds": s.project_ids,
                "scheduleIds": s.schedule_ids,
                "autoGenerated": s.auto_generated,
                "createdAt": s.created_at.isoformat()
            })

        logger.info(f"✅ Retrieved {len(result)} scenarios for user {user_id}")

        return {
            "success": True,
            "message": f"Retrieved {len(result)} scenarios",
            "data": {
                "scenarios": result,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        }

    except Exception as e:
        logger.error(f"❌ Failed to retrieve scenarios: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scenarios: {str(e)}")


@router.get("/{scenario_id}", response_model=dict)
async def get_scenario(
    scenario_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific scenario by ID.

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])

        # Query scenario
        scenario = db.query(Scenario).filter(
            Scenario.id == scenario_id,
            Scenario.user_id == user_id
        ).first()

        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")

        # Convert to response format
        result = {
            "id": str(scenario.id),
            "title": scenario.title,
            "description": scenario.description,
            "scenarioText": scenario.scenario_text,
            "language": scenario.language,
            "difficulty": scenario.difficulty,
            "category": scenario.category,
            "roles": scenario.roles,
            "requiredTerminology": scenario.required_terminology,
            "projectIds": scenario.project_ids,
            "scheduleIds": scenario.schedule_ids,
            "autoGenerated": scenario.auto_generated,
            "createdAt": scenario.created_at.isoformat()
        }

        return {
            "success": True,
            "message": "Scenario retrieved successfully",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve scenario: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scenario: {str(e)}")


@router.put("/{scenario_id}", response_model=dict)
async def update_scenario(
    scenario_id: str,
    request: ManualCreateRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a scenario by ID.

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])

        # Query scenario
        scenario = db.query(Scenario).filter(
            Scenario.id == scenario_id,
            Scenario.user_id == user_id
        ).first()

        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")

        # Update scenario fields
        scenario.title = request.title
        scenario.description = request.description
        scenario.scenario_text = request.scenarioText
        scenario.language = request.language
        scenario.difficulty = request.difficulty
        scenario.category = request.category
        scenario.roles = request.roles.dict()
        scenario.required_terminology = request.requiredTerminology

        db.commit()
        db.refresh(scenario)

        logger.info(f"✏️  Updated scenario {scenario_id} for user {user_id}")

        # Convert to response format
        result = {
            "id": str(scenario.id),
            "title": scenario.title,
            "description": scenario.description,
            "scenarioText": scenario.scenario_text,
            "language": scenario.language,
            "difficulty": scenario.difficulty,
            "category": scenario.category,
            "roles": scenario.roles,
            "requiredTerminology": scenario.required_terminology,
            "projectIds": scenario.project_ids,
            "scheduleIds": scenario.schedule_ids,
            "autoGenerated": scenario.auto_generated,
            "createdAt": scenario.created_at.isoformat()
        }

        return {
            "success": True,
            "message": "Scenario updated successfully",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update scenario: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update scenario: {str(e)}")


@router.delete("/{scenario_id}", response_model=dict)
async def delete_scenario(
    scenario_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a scenario by ID.

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])

        # Query scenario
        scenario = db.query(Scenario).filter(
            Scenario.id == scenario_id,
            Scenario.user_id == user_id
        ).first()

        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")

        # Delete scenario
        db.delete(scenario)
        db.commit()

        logger.info(f"🗑️  Deleted scenario {scenario_id} for user {user_id}")

        return {
            "success": True,
            "message": "Scenario deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete scenario: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete scenario: {str(e)}")


@router.post("/modify-with-chat", response_model=dict)
async def modify_scenario_with_chat(
    request: ModifyWithChatRequest,
    user: dict = Depends(get_current_user)
):
    """
    채팅을 통한 시나리오 수정

    사용자의 자연어 요청을 받아 GPT-4o를 사용하여 시나리오를 수정합니다.
    현재 시나리오 상태를 컨텍스트로 전달하고, 수정된 필드만 반환합니다.

    **Authentication required**: JWT token from Java backend
    """
    try:
        user_id = str(user["user_id"])

        logger.info(f"💬 Chat modification request: user={user_id}, message='{request.userMessage[:50]}...'")

        result = await scenario_service.modify_with_chat(
            current_scenario=request.currentScenario,
            user_message=request.userMessage,
            language=request.language,
            difficulty=request.difficulty
        )

        return {
            "success": True,
            "message": "Scenario modification successful",
            "data": result
        }

    except Exception as e:
        logger.error(f"❌ Scenario modification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario modification failed: {str(e)}")
