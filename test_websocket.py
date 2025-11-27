import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/ai/voice/stt/stream"

    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")

            # Send initial config
            config = {"selected_languages": ["ko-KR", "en-US"]}
            await websocket.send(json.dumps(config))
            print(f"Sent config: {config}")

            # Try to receive a response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received: {response}")
            except asyncio.TimeoutError:
                print("⏱️ No immediate response (this is normal for STT)")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())
