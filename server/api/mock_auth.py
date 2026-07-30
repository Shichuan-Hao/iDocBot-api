"""
本地 Mock 登录接口（替代前端硬编码的 mengxuegu mock 登录）。
支持任意用户名/密码登录，方便调试。
"""
import time
from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Optional


router = APIRouter(prefix="/api/v1", tags=["mock-auth"])


class LoginRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    phone: Optional[str] = None
    code: Optional[str] = None


class UserInfo(BaseModel):
    id: str = "1"
    username: str = "user"
    nickname: str = "Fufan User"
    avatar: str = ""


@router.post("/users/login")
async def login(data: LoginRequest):
    """Mock 登录：任意用户名密码都返回成功"""
    username = data.username or data.phone or "guest"
    return {
        "code": 200,
        "msg": "登录成功",
        "data": {
            "token": f"mock-token-{int(time.time())}",
            "userInfo": {
                "id": "1",
                "username": username,
                "nickname": username,
                "avatar": "",
            },
        },
    }


@router.post("/users/logout")
async def logout():
    return {"code": 200, "msg": "退出成功", "data": None}


@router.get("/users/info")
async def get_user_info():
    """获取用户信息"""
    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "id": "1",
            "username": "guest",
            "nickname": "Fufan User",
            "avatar": "",
        },
    }


@router.post("/users/sendCode")
async def send_code(data: LoginRequest):
    """发送验证码（mock）"""
    return {"code": 200, "msg": "验证码已发送（mock: 1234）", "data": None}