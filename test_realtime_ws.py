import asyncio
import websockets

async def test_websocket():
    uri = "ws://localhost:8000/api/ai/voice/realtime"

    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
