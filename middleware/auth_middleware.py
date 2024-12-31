from fastapi import Request, HTTPException

async def get_current_user(request: Request):
    """
    Middleware to handle authentication.
    Replace this with your actual Clerk authentication logic.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # TODO: Implement actual Clerk authentication
    # This is just a mock implementation
    return {"userId": "mock_user_id"}
