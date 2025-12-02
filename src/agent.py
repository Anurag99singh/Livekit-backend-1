import logging
import threading
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from livekit.plugins import sarvam
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        # Base instructions
        base_instructions = (
            "Greet user as his name is known from the profile,like hii <name>! I am Talkypie, How can I assist you today?.Dont take name of user multiple times in a single response. "
            "You are a kid assistant, who helps engage kids in a fun playful manner. "
            "Please be concise in your responses. Use very simple language that kids can understand and use short sentences. "
            "You eagerly assist users with their questions by providing information from your extensive knowledge. "
            "Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols. "
            "You are curious, friendly, and have a sense of humor."
        )

        # If a profile file exists (written by the frontend via the profile endpoint), include it in the assistant instructions
        profile_instructions = ""
        try:
            profile_path = os.path.join(os.getcwd(), "profile.json")
            if os.path.exists(profile_path):
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                parts = []
                name = profile.get("name")
                if name:
                    parts.append(f"Child name: {name}.")
                age = profile.get("age")
                if age:
                    parts.append(f"Age: {age}.")
                gender = profile.get("gender")
                if gender:
                    parts.append(f"Gender: {gender}.")
                likes = profile.get("likes")
                if likes:
                    parts.append(f"Likes/Dislikes: {likes}")
                learning = profile.get("learning")
                if learning:
                    parts.append(f"Current learning: {learning}")

                if parts:
                    profile_instructions = (
                        "Use the following parent-provided profile to tailor your responses for the child: "
                        + " ".join(parts)
                    )
        except Exception:
            # best-effort only; don't crash the agent if profile can't be read
            profile_instructions = ""

        final_instructions = base_instructions + (" " + profile_instructions if profile_instructions else "")

        super().__init__(
            instructions=final_instructions,
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en")
        stt=sarvam.STT(
                        language="hi-IN",
                        model="saarika:v2.5",
                    ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/

        # tts=inference.TTS(
        #     model="cartesia/sonic-3", voice="f31cc6a7-c1e8-4764-980c-60a361443dd1"
        # ),
        tts=sarvam.TTS(
                    target_language_code="hi-IN",
                    speaker="manisha",
                ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


def _start_profile_server(host: str = "0.0.0.0", port: int = 8080):
    """Start a tiny HTTP server that accepts POST /profile to save a JSON profile to disk.

    This is intentionally dependency-free and minimal. The server will respond with CORS headers
    so the frontend can POST from localhost:3000 during development.
    """

    class Handler(BaseHTTPRequestHandler):
        def _set_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            # Allow GET as well so frontend can poll GET /profile from another origin
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._set_cors_headers()
            self.end_headers()

        def do_POST(self):
            if self.path != "/profile":
                self.send_response(404)
                self._set_cors_headers()
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                self.send_response(400)
                self._set_cors_headers()
                self.end_headers()
                return

            try:
                profile_path = os.path.join(os.getcwd(), "profile.json")
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")
            except Exception:
                self.send_response(500)
                self._set_cors_headers()
                self.end_headers()

        def do_GET(self):
            # Serve saved profile.json for client-side auto-start
            if self.path != "/profile":
                self.send_response(404)
                self._set_cors_headers()
                self.end_headers()
                return

            try:
                profile_path = os.path.join(os.getcwd(), "profile.json")
                if not os.path.exists(profile_path):
                    # No profile saved yet
                    self.send_response(204)
                    self._set_cors_headers()
                    self.end_headers()
                    return

                with open(profile_path, "r", encoding="utf-8") as f:
                    data = f.read()

                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data.encode("utf-8"))
            except Exception:
                self.send_response(500)
                self._set_cors_headers()
                self.end_headers()

    try:
        server = HTTPServer((host, port), Handler)
    except Exception:
        logger.exception("Profile server failed to start on %s:%s", host, port)
        return

    def serve():
        logger.info("Profile endpoint listening on http://%s:%s/profile", host, port)
        try:
            server.serve_forever()
        except Exception:
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()


# Start the profile server as soon as this module is imported so the frontend can POST before
# a session begins. This is a best-effort convenience for local development.
try:
    _start_profile_server()
except Exception:
    logger.exception("Failed to initialize profile HTTP endpoint")



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
