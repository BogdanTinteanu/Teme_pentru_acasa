import requests
import httpx
import sys
import pytest

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

# Done: Test pentru endpoint-ul root
def test_read_root():
    """Verifica daca serverul este pornit si raspunde la ruta principala."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    # Verificam daca primim un mesaj de bun venit (ex: "Bine ai venit in "Asistentul tau personal de fitness"!)
    assert "message" in response.json()

# Done: Scenariu testare pentru endpoint-ul /chat/
@pytest.mark.asyncio
async def test_chat_success():
    """Verifica daca agentul raspunde corect la o intrebare valida."""
    payload = {"message": "Ce exercitii pot face pentru brate?"}
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/chat/", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Verificam daca raspunsul nu este gol
    assert len(data["response"]) > 0

# Done: Test negativ pentru endpoint-ul /chat/
@pytest.mark.asyncio
async def test_chat_empty_message():
    """Verifica comportamentul serverului cand primeste un mesaj gol."""
    payload = {"message": ""}
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/chat/", json=payload)
    # De exemplu ar trebui ss dea o eroare 400 (bad request) sau 422 (unprocessable entity)
    assert response.status_code in [400, 422]
